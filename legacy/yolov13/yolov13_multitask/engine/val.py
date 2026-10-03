"""Multi-task validator: detection mAP + drivable-area mIoU + lane mIoU."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import DetMetrics, box_iou

from yolov13_multitask.utils.metrics import SegMetrics


class MultiTaskValidator:
    """Lean validator. Built to be called by the trainer with a model+loader."""

    def __init__(
        self,
        data: Dict,
        device: torch.device,
        da_classes: int,
        ll_classes: int,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        max_det: int = 300,
        save_dir: Path | str = ".",
    ) -> None:
        self.data = data
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.save_dir = Path(save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

        self.det_metrics = DetMetrics(save_dir=self.save_dir, names={i: n for i, n in enumerate(data.get("names", []))})
        self.det_metrics.names = {i: n for i, n in enumerate(data.get("names", []))}

        self.da_metrics = SegMetrics(da_classes, names=list(data.get("da_names", [])))
        self.ll_metrics = SegMetrics(ll_classes, names=list(data.get("ll_names", [])))

    # --------------------------------------------------------------- internals
    def _match(self, pred_cls: torch.Tensor, gt_cls: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
        correct = np.zeros((pred_cls.shape[0], self.niou), dtype=bool)
        if pred_cls.numel() == 0:
            return torch.tensor(correct, dtype=torch.bool, device=pred_cls.device)
        correct_class = gt_cls[:, None] == pred_cls
        iou = (iou * correct_class).cpu().numpy()
        for i, t in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= t)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_cls.device)

    # ------------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, model, loader) -> Dict:
        model.eval()
        model = model.to(self.device)
        # Stash export/inplace flags so we don't mutate model permanently
        # Detect head will return (decoded_preds, feats) tuple in eval mode; we
        # need decoded_preds to feed into NMS.
        stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.da_metrics.reset()
        self.ll_metrics.reset()

        for batch in loader:
            img = batch["img"].to(self.device)
            preds = model(img)
            det_out = preds["det"] if isinstance(preds, dict) else preds[0]
            da_logits = preds["da"] if isinstance(preds, dict) else preds[1]
            ll_logits = preds["ll"] if isinstance(preds, dict) else preds[2]

            # Decoded boxes ------------------------------------------------
            if isinstance(det_out, (list, tuple)):
                # Eval mode in upstream Detect returns (y, x). Pick decoded y.
                decoded = det_out[0] if isinstance(det_out, tuple) else det_out[0]
            else:
                decoded = det_out
            nms_out = ops.non_max_suppression(
                decoded, self.conf_thres, self.iou_thres, max_det=self.max_det
            )

            H, W = img.shape[-2:]
            cls = batch["cls"].to(self.device).view(-1)
            bboxes_xywh = batch["bboxes"].to(self.device)  # normalized
            bidx = batch["batch_idx"].to(self.device)

            for si, pred in enumerate(nms_out):
                idx = (bidx == si)
                gt_cls_si = cls[idx]
                gt_box_si = bboxes_xywh[idx]
                if gt_box_si.numel():
                    # convert normalized xywh to pixel xyxy at the input resolution
                    gt_xyxy = ops.xywh2xyxy(gt_box_si) * torch.tensor((W, H, W, H), device=self.device)
                else:
                    gt_xyxy = torch.zeros((0, 4), device=self.device)

                npr = 0 if pred is None else pred.shape[0]
                stat_tp = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
                stat_conf = torch.zeros(0, device=self.device)
                stat_pred_cls = torch.zeros(0, device=self.device)
                if npr:
                    stat_conf = pred[:, 4]
                    stat_pred_cls = pred[:, 5]
                    if gt_xyxy.numel():
                        iou = box_iou(gt_xyxy, pred[:, :4])
                        stat_tp = self._match(pred[:, 5], gt_cls_si, iou)
                stats["tp"].append(stat_tp)
                stats["conf"].append(stat_conf)
                stats["pred_cls"].append(stat_pred_cls)
                stats["target_cls"].append(gt_cls_si)
                stats["target_img"].append(gt_cls_si.unique())

            # Segmentation -------------------------------------------------
            da_target = batch["da_mask"].to(self.device)
            ll_target = batch["ll_mask"].to(self.device)
            if da_logits.shape[-2:] != da_target.shape[-2:]:
                da_logits = F.interpolate(da_logits, size=da_target.shape[-2:], mode="bilinear", align_corners=False)
            if ll_logits.shape[-2:] != ll_target.shape[-2:]:
                ll_logits = F.interpolate(ll_logits, size=ll_target.shape[-2:], mode="bilinear", align_corners=False)
            has_da = batch["has_da"].to(self.device).bool()
            has_ll = batch["has_ll"].to(self.device).bool()
            if has_da.any():
                self.da_metrics.update(da_logits[has_da], da_target[has_da])
            if has_ll.any():
                self.ll_metrics.update(ll_logits[has_ll], ll_target[has_ll])

        # ------------------------------------------------------------------- agg
        cat = lambda k: torch.cat(stats[k], 0).cpu().numpy() if stats[k] else np.zeros(0)
        det_results = {}
        try:
            tp = cat("tp")
            if tp.size:
                conf = cat("conf")
                pred_cls = cat("pred_cls")
                target_cls = cat("target_cls")
                self.det_metrics.process(tp=tp, conf=conf, pred_cls=pred_cls, target_cls=target_cls)
                p, r, map50, map75, map5095 = self.det_metrics.mean_results()
                det_results = {"precision": float(p), "recall": float(r), "map50": float(map50),
                               "map75": float(map75), "map50_95": float(map5095)}
        except Exception as e:  # pragma: no cover
            LOGGER.warning("DetMetrics aggregation failed: %s", e)
            det_results = {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map75": 0.0, "map50_95": 0.0}

        da_summary = self.da_metrics.summary()
        ll_summary = self.ll_metrics.summary()
        out = {
            **det_results,
            "mIoU_da": da_summary["mIoU"],
            "mIoU_ll": ll_summary["mIoU"],
            "iou_da_per_class": da_summary["per_class_iou"],
            "iou_ll_per_class": ll_summary["per_class_iou"],
            "pixel_acc_da": da_summary["pixel_accuracy"],
            "pixel_acc_ll": ll_summary["pixel_accuracy"],
        }

        LOGGER.info(
            "VAL  mAP50=%.4f  mAP50-95=%.4f  mIoU_da=%.4f  mIoU_ll=%.4f",
            out.get("map50", 0.0), out.get("map50_95", 0.0), out["mIoU_da"], out["mIoU_ll"],
        )
        LOGGER.info("DA per-class IoU: %s", da_summary["per_class_iou"])
        LOGGER.info("LL per-class IoU: %s", ll_summary["per_class_iou"])
        return out
