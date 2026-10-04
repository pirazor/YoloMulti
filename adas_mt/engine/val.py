"""Validator: detection mAP (NMS-free head) + drivable-area and lane IoU on the deployed 384x640 geometry.

Metrics are computed at the model's input resolution on non-ignored pixels, so letterbox padding and tasks
that are unannotated for an image never count. Lane IoU is measured on the training masks (lane lines about
4 px wide at 640 px, i.e. the "8 px at 720p" protocol), so it is NOT comparable with the 2 px-test numbers of
YOLOP-style papers; compare runs of this repository with each other.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER

from adas_mt.data.masks import unpack_masks

from .config import MTConfig
from .metrics import SegConfusion


class MultiTaskValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None, mt: MTConfig | None = None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.mt = mt or MTConfig()
        self._seg: tuple | None = None

    # ------------------------------------------------------------------ data
    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Used when validating standalone (the trainer passes its own loader): same geometry as training."""
        from adas_mt.data.dataset import MultiTaskDataset

        return MultiTaskDataset(
            img_path=img_path, data=self.data, imgsz=self.mt.imgsz, augment=False, hyp=self.args, batch_size=batch,
            cache=self.args.cache or False, rect=False, stride=32, pad=0.0, single_cls=self.args.single_cls or False,
            classes=self.args.classes, fraction=self.args.fraction, prefix=f"{mode}: ",
        )

    # ------------------------------------------------------------------ setup
    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        if not getattr(self, "end2end", False):
            LOGGER.warning(
                "validating the one-to-many head with NMS, which is NOT the deployed model. "
                "Set nms=False to validate the NMS-free head."
            )
        # standalone validation wraps the model in AutoBackend; exported engines have no attributes at all,
        # so fall back to data.yaml (the packed mask is decoded with these counts)
        base = model.model if getattr(model, "format", None) == "pt" else model
        self.da_classes = int(getattr(base, "da_classes", None) or self.data["da_classes"])
        self.ll_classes = int(getattr(base, "ll_classes", None) or self.data["ll_classes"])
        self.da = SegConfusion(self.da_classes, getattr(base, "da_names", None) or self.data.get("da_names", []), self.device)
        self.ll = SegConfusion(self.ll_classes, getattr(base, "ll_names", None) or self.data.get("ll_names", []), self.device)

    # ---------------------------------------------------------------- per batch
    def postprocess(self, preds: Dict[str, Any]):  # type: ignore[override]
        det = preds["det"]
        det = det[0] if isinstance(det, (tuple, list)) else det  # eval output is (decoded, raw)
        self._seg = (preds["da"], preds["ll"])
        return super().postprocess(det)

    def update_metrics(self, preds, batch) -> None:  # type: ignore[override]
        super().update_metrics(preds, batch)
        da_t, ll_t = unpack_masks(batch["semantic_mask"], self.da_classes, self.ll_classes)
        da, ll = self._seg
        self.da.update(da.argmax(1), da_t)
        self.ll.update(ll.argmax(1), ll_t)

    def gather_stats(self) -> None:
        super().gather_stats()
        self.da.reduce()
        self.ll.reduce()

    # ------------------------------------------------------------------- results
    def get_stats(self) -> Dict[str, float]:
        stats = dict(super().get_stats())
        da, ll = self.da.results("da"), self.ll.results("ll")
        stats.update({f"metrics/{k}": v for k, v in {**da, **ll}.items()})
        w = self.mt.fitness
        stats["fitness"] = (
            w["det"] * float(stats.get("metrics/mAP50-95(B)", 0.0))
            + w["da"] * da["da_mIoU"]
            + w["ll"] * ll["ll_IoU_fg"]
        )
        return stats

    def print_results(self) -> None:
        super().print_results()
        s = self._last_seg_summary()
        LOGGER.info(s)

    def _last_seg_summary(self) -> str:
        da, ll = self.da.results("da"), self.ll.results("ll")
        return (
            f"{'drivable area':>22}  mIoU {da['da_mIoU']:.3f}  IoU(fg) {da['da_IoU_fg']:.3f}  "
            + " ".join(f"{n} {da[f'da_IoU_{n}']:.3f}" for n in self.da.names)
            + f"\n{'lane':>22}  IoU(fg) {ll['ll_IoU_fg']:.3f}  recall {ll['ll_recall_fg']:.3f}  mIoU {ll['ll_mIoU']:.3f}  "
            + " ".join(f"{n} {ll[f'll_IoU_{n}']:.3f}" for n in self.ll.names)
        )
