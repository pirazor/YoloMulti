"""Per-source evaluators, all in the native image frame of each dataset.

* :class:`DetEvaluator`: COCO-style mAP50 / mAP50-95 (greedy IoU matching per class, Ultralytics' ``ap_per_class``)
  for road users (8 classes), elements (2) and the BDD100K 10-class union.
* :class:`ElementEvaluator`: light-colour accuracy, fine-sign top-1 accuracy on elements matched to the ground
  truth by IoU >= 0.5 (classification given a detection, like the sign classifier baseline).
* :class:`SegEvaluator`: drivable-area IoUs in the network frame (``adas_mt.engine.metrics.SegConfusion``).
* :class:`LaneEvaluator`: CULane-style F1: lanes rendered 30 px wide (at 1640 px width, scaled with the image),
  IoU >= 0.5 is a hit, greedy matching; plus order / direction accuracy on the matched lanes where labelled.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from adas_mt.engine.metrics import SegConfusion
from sparc.data.dataset import class_to_order
from sparc.data.schema import (AGENT_CLASSES, BDD_CLASSES, DA_CLASSES, EL_LIGHT, EL_SIGN, ELEMENT_CLASSES,
                               LANE_ORDER_UNLABELLED, Sample)

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match_tp(pred_boxes: np.ndarray, pred_cls: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray,
             thresholds: np.ndarray = IOU_THRESHOLDS) -> np.ndarray:
    """(N, T) bool: prediction i is a true positive at threshold t (same class, greedy by IoU, each GT used once).
    Predictions must be sorted by descending confidence."""
    tp = np.zeros((len(pred_boxes), len(thresholds)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    iou = _iou(pred_boxes, gt_boxes) * (pred_cls[:, None] == gt_cls[None, :])
    for t, thr in enumerate(thresholds):
        used = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(pred_boxes)):  # confidence order
            cand = np.where((iou[i] >= thr) & ~used)[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                tp[i, t] = True
    return tp


class DetEvaluator:
    def __init__(self, names: Sequence[str]):
        self.names = list(names)
        self.tp, self.conf, self.pcls, self.tcls = [], [], [], []

    def update(self, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls) -> None:
        pred_boxes, pred_scores, pred_cls = (np.asarray(x) for x in (pred_boxes, pred_scores, pred_cls))
        order = np.argsort(-pred_scores)
        pred_boxes, pred_scores, pred_cls = pred_boxes[order], pred_scores[order], pred_cls[order]
        self.tp.append(match_tp(pred_boxes.reshape(-1, 4), pred_cls, np.asarray(gt_boxes).reshape(-1, 4), np.asarray(gt_cls)))
        self.conf.append(pred_scores)
        self.pcls.append(pred_cls)
        self.tcls.append(np.asarray(gt_cls))

    def results(self, prefix: str) -> Dict[str, float]:
        from ultralytics.utils.metrics import ap_per_class

        tcls = np.concatenate(self.tcls) if self.tcls else np.zeros(0)
        if len(tcls) == 0:
            return {f"{prefix}mAP50": 0.0, f"{prefix}mAP50-95": 0.0}
        tp = np.concatenate(self.tp) if self.tp else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
        conf = np.concatenate(self.conf) if self.conf else np.zeros(0)
        pcls = np.concatenate(self.pcls) if self.pcls else np.zeros(0)
        if len(conf) == 0:
            return {f"{prefix}mAP50": 0.0, f"{prefix}mAP50-95": 0.0}
        res = ap_per_class(tp, conf, pcls, tcls, names={i: n for i, n in enumerate(self.names)})
        ap = res[5]  # (nc_present, 10)
        classes = res[6] if len(res) > 6 else np.unique(tcls)
        out = {f"{prefix}mAP50": float(ap[:, 0].mean()), f"{prefix}mAP50-95": float(ap.mean())}
        for c, row in zip(classes, ap):
            out[f"{prefix}AP50_{self.names[int(c)]}"] = float(row[0])
        return out


class ElementEvaluator:
    """Attribute / fine-class accuracy of elements matched to GT elements (IoU >= 0.5, best score)."""

    def __init__(self):
        self.light = [0, 0]
        self.sign = [0, 0]
        self.recall = [0, 0]

    def update(self, pred_boxes, pred_scores, pred_cls, pred_attr, pred_sign, gt_boxes, gt_cls, gt_attr, gt_sign) -> None:
        gt_boxes = np.asarray(gt_boxes).reshape(-1, 4)
        self.recall[1] += len(gt_boxes)
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return
        order = np.argsort(-np.asarray(pred_scores))
        pb, pc, pa, ps = (np.asarray(x)[order] for x in (pred_boxes, pred_cls, pred_attr, pred_sign))
        iou = _iou(pb.reshape(-1, 4), gt_boxes)
        used = np.zeros(len(pb), dtype=bool)
        for j in range(len(gt_boxes)):
            cand = np.where((iou[:, j] >= 0.5) & ~used & (pc == gt_cls[j]))[0]
            if len(cand) == 0:
                continue
            i = cand[0]  # highest score among candidates
            used[i] = True
            self.recall[0] += 1
            if gt_cls[j] == EL_LIGHT and gt_attr[j] >= 0:
                self.light[1] += 1
                self.light[0] += int(pa[i] == gt_attr[j])
            if gt_cls[j] == EL_SIGN and gt_sign[j] >= 0:
                self.sign[1] += 1
                self.sign[0] += int(ps[i] == gt_sign[j])

    def results(self, prefix: str) -> Dict[str, float]:
        out = {f"{prefix}el_recall50": self.recall[0] / max(self.recall[1], 1)}
        if self.light[1]:
            out[f"{prefix}light_color_acc"] = self.light[0] / self.light[1]
        if self.sign[1]:
            out[f"{prefix}sign_top1"] = self.sign[0] / self.sign[1]
        return out


class SegEvaluator:
    def __init__(self, device="cpu"):
        self.conf = SegConfusion(len(DA_CLASSES), DA_CLASSES, device)

    def update(self, da_logits: torch.Tensor, da_target: torch.Tensor) -> None:
        self.conf.update(da_logits.argmax(1), da_target)

    def results(self, prefix: str) -> Dict[str, float]:
        return {f"{prefix}{k.split('/', 1)[-1]}": v for k, v in self.conf.results("da").items()}


def render_lanes(polys: Sequence[np.ndarray], hw, width_px: int) -> List[np.ndarray]:
    out = []
    for p in polys:
        m = np.zeros(hw, np.uint8)
        pts = np.round(np.asarray(p, dtype=np.float64)).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(m, [pts], False, 1, int(width_px))
        out.append(m)
    return out


class LaneEvaluator:
    """CULane protocol: 30 px line width for a 1640 px wide image, scaled with the width; IoU >= 0.5 is a hit."""

    def __init__(self, iou_thr: float = 0.5, width_at_1640: float = 30.0):
        self.iou_thr, self.w1640 = iou_thr, width_at_1640
        self.tp = self.fp = self.fn = 0
        self.order = [0, 0]
        self.direction = [0, 0]

    def update(self, pred: List[Dict], gt: Sample) -> None:
        hw = gt.hw
        w = max(1, int(round(self.w1640 * hw[1] / 1640)))
        gm = render_lanes([l.points for l in gt.lanes], hw, w)
        pm = render_lanes([p["points"] for p in pred], hw, w)
        if not gm or not pm:
            self.fp += len(pm)
            self.fn += len(gm)
            return
        iou = np.zeros((len(pm), len(gm)))
        for i, a in enumerate(pm):
            for j, b in enumerate(gm):
                inter = np.logical_and(a, b).sum()
                iou[i, j] = inter / max(np.logical_or(a, b).sum(), 1)
        used_p, used_g = set(), set()
        for i, j in sorted(((i, j) for i in range(len(pm)) for j in range(len(gm))), key=lambda ij: -iou[ij]):
            if iou[i, j] < self.iou_thr:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            self.tp += 1
            lane = gt.lanes[j]
            if lane.order != LANE_ORDER_UNLABELLED and not lane.weak:
                self.order[1] += 1
                self.order[0] += int(class_to_order(pred[i]["order"]) == lane.order)
            if lane.direction >= 0 and not lane.weak:
                self.direction[1] += 1
                self.direction[0] += int(pred[i]["dir"] == lane.direction)
        self.fp += len(pm) - len(used_p)
        self.fn += len(gm) - len(used_g)

    def results(self, prefix: str) -> Dict[str, float]:
        p = self.tp / max(self.tp + self.fp, 1)
        r = self.tp / max(self.tp + self.fn, 1)
        out = {f"{prefix}lane_F1": 2 * p * r / max(p + r, 1e-9), f"{prefix}lane_P": p, f"{prefix}lane_R": r}
        if self.order[1]:
            out[f"{prefix}lane_order_acc"] = self.order[0] / self.order[1]
        if self.direction[1]:
            out[f"{prefix}lane_dir_acc"] = self.direction[0] / self.direction[1]
        return out


class SourceEvaluator:
    """All evaluators for one validation source; ``update`` takes post-processed predictions + the raw samples."""

    def __init__(self, source: str, device="cpu"):
        self.source = source
        self.agents = DetEvaluator(AGENT_CLASSES)
        self.elements = DetEvaluator(ELEMENT_CLASSES)
        self.bdd = DetEvaluator(BDD_CLASSES)
        self.attrs = ElementEvaluator()
        self.seg = SegEvaluator(device)
        self.lanes = LaneEvaluator()
        self.flags = {"agents": False, "elements": False, "da": False, "lanes": False}

    def update(self, preds: List[Dict], samples: List[Sample], batch: Dict, da_logits: Optional[torch.Tensor],
               da_target: Optional[torch.Tensor]) -> None:
        for b, (p, s) in enumerate(zip(preds, samples)):
            lm = s.label_mask
            a, e = p["agents"], p["elements"]
            if lm.agents:
                self.flags["agents"] = True
                vocab = np.isin(a["cls"], list(lm.cls_vocab))  # predictions of classes the source does not label are not errors
                self.agents.update(a["boxes"][vocab], a["scores"][vocab], a["cls"][vocab], s.boxes2d, s.cls)
            if lm.elements:
                self.flags["elements"] = True
                vocab = np.isin(e["cls"], list(lm.el_vocab))
                self.elements.update(e["boxes"][vocab], e["scores"][vocab], e["cls"][vocab], s.el_boxes, s.el_cls)
                gt_sign = np.array([batch["vocab"].id(n) for n in s.el_sign.tolist()]) if "vocab" in batch else np.full(len(s.el_cls), -1)
                self.attrs.update(e["boxes"][vocab], e["scores"][vocab], e["cls"][vocab], e["attr"][vocab], e["sign"][vocab],
                                  s.el_boxes, s.el_cls, s.el_attr, gt_sign)
            if lm.agents and lm.elements:
                va = np.isin(a["cls"], list(lm.cls_vocab))
                ve = np.isin(e["cls"], list(lm.el_vocab))
                boxes = np.concatenate([a["boxes"][va], e["boxes"][ve]])
                scores = np.concatenate([a["scores"][va], e["scores"][ve]])
                cls = np.concatenate([a["cls"][va], e["cls"][ve] + len(AGENT_CLASSES)])
                self.bdd.update(boxes, scores, cls, np.concatenate([s.boxes2d, s.el_boxes]),
                                np.concatenate([s.cls, s.el_cls + len(AGENT_CLASSES)]))
            if lm.lane_geom:
                self.flags["lanes"] = True
                self.lanes.update(p["lanes"], s)
        if da_logits is not None and da_target is not None:
            self.flags["da"] = True
            self.seg.update(da_logits, da_target)

    def results(self) -> Dict[str, float]:
        pre = f"{self.source}/"
        out: Dict[str, float] = {}
        if self.flags["agents"]:
            out.update(self.agents.results(pre + "agents_"))
        if self.flags["elements"]:
            out.update(self.elements.results(pre + "elements_"))
            out.update(self.attrs.results(pre))
        if self.flags["agents"] and self.flags["elements"]:
            out.update({k: v for k, v in self.bdd.results(pre + "bdd10_").items() if "AP50_" not in k})
        if self.flags["da"]:
            out.update(self.seg.results(pre))
        if self.flags["lanes"]:
            out.update(self.lanes.results(pre))
        return out
