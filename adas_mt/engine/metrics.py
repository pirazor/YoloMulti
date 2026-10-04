"""Streaming semantic-segmentation metrics for the DA and lane heads (confusion matrix on the device)."""

from __future__ import annotations

from typing import Dict, Sequence

import torch


class SegConfusion:
    """``update(pred, target)``: ``pred`` (B,H,W) class ids, ``target`` (B,H,W) with 255 = ignore."""

    def __init__(self, nc: int, names: Sequence[str], device: torch.device | str = "cpu"):
        self.nc = int(nc)
        self.names = list(names)[: self.nc] + [f"class{i}" for i in range(len(names), self.nc)]
        self.mat = torch.zeros(self.nc, self.nc, dtype=torch.long, device=device)  # rows: GT, cols: prediction

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        valid = target < self.nc  # also drops 255 (padding / unannotated)
        t, p = target[valid].long(), pred[valid].long().clamp(max=self.nc - 1)
        self.mat += torch.bincount(t * self.nc + p, minlength=self.nc * self.nc).reshape(self.nc, self.nc).to(self.mat)

    def reduce(self) -> None:
        """Sum across DDP ranks (no-op when not distributed)."""
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.mat)

    def results(self, prefix: str) -> Dict[str, float]:
        """IoUs are averaged over classes that have ground truth; background (class 0) is reported separately."""
        m = self.mat.double().cpu()
        tp = m.diag()
        gt = m.sum(1)
        pr = m.sum(0)
        union = gt + pr - tp
        iou = torch.where(union > 0, tp / union.clamp(min=1), torch.zeros_like(tp))
        has_gt = gt > 0
        fg = has_gt.clone()
        fg[0] = False
        out: Dict[str, float] = {f"{prefix}_mIoU": float(iou[fg].mean()) if fg.any() else 0.0}
        for i in range(self.nc):
            out[f"{prefix}_IoU_{self.names[i]}"] = float(iou[i])
        # binary "any foreground class" IoU (drivable area as one region / any lane marking), YOLOP-style
        if self.nc > 1:
            g_fg, p_fg = m[1:].sum(), m[:, 1:].sum()
            inter = m[1:, 1:].sum()
            u = g_fg + p_fg - inter
            out[f"{prefix}_IoU_fg"] = float(inter / u) if u > 0 else 0.0
            out[f"{prefix}_recall_fg"] = float(inter / g_fg) if g_fg > 0 else 0.0  # "lane accuracy"
        out[f"{prefix}_pixel_acc"] = float(tp.sum() / m.sum()) if m.sum() > 0 else 0.0
        return out
