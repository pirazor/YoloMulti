"""Per-class IoU + pixel accuracy for semantic segmentation tasks.

Uses a streaming confusion-matrix accumulator so we don't keep predictions in
memory across the validation loop.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class SegMetrics:
    """Multi-class semantic-segmentation metrics."""

    def __init__(self, num_classes: int, names: List[str] | None = None):
        self.num_classes = int(num_classes)
        self.names = names or [f"class_{i}" for i in range(num_classes)]
        self.confmat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confmat[:] = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """preds: (N, C, H, W) logits OR (N, H, W) class ids; target: (N, H, W)."""
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)
        p = preds.detach().to(torch.int64).flatten().cpu().numpy()
        t = target.detach().to(torch.int64).flatten().cpu().numpy()
        valid = (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
        idx = self.num_classes * t[valid] + p[valid]
        bc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confmat += bc.reshape(self.num_classes, self.num_classes)

    # ------------------------------------------------------------------ readout
    def per_class_iou(self) -> np.ndarray:
        cm = self.confmat
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        denom = tp + fp + fn
        iou = np.zeros_like(tp)
        valid = denom > 0
        iou[valid] = tp[valid] / denom[valid]
        return iou

    def mean_iou(self, ignore_bg: bool = True) -> float:
        iou = self.per_class_iou()
        if ignore_bg and len(iou) > 1:
            iou = iou[1:]
        return float(iou.mean()) if len(iou) else 0.0

    def pixel_accuracy(self) -> float:
        cm = self.confmat
        denom = cm.sum()
        return float(np.diag(cm).sum() / denom) if denom else 0.0

    def summary(self, ignore_bg: bool = True) -> Dict:
        iou = self.per_class_iou()
        per_class = {self.names[i]: float(iou[i]) for i in range(self.num_classes)}
        return {
            "per_class_iou": per_class,
            "mIoU": self.mean_iou(ignore_bg=ignore_bg),
            "pixel_accuracy": self.pixel_accuracy(),
        }
