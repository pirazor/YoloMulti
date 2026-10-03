"""Joint loss: Ultralytics E2E detection loss + masked segmentation losses.

Segmentation targets come from the packed ``batch["semantic_mask"]`` (see data/masks.py); pixels that
are padding or belong to an unannotated task are 255 and are excluded from CE and from the region
losses, so they contribute exactly zero gradient.

Returned like upstream criteria: ``(loss_vector * batch_size, items)`` where ``loss_vector`` is
``[box, cls, dfl|l1, da, ll]`` and the trainer sums it.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

from adas_mt.data.masks import IGNORE, unpack_masks

AUX_WEIGHT = 0.4


def masked_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean CE over non-ignored pixels; exactly 0 (with a live graph) when every pixel is ignored."""
    valid = target != IGNORE
    n = valid.sum()
    ce = F.cross_entropy(logits, target.clamp(max=logits.shape[1] - 1), reduction="none")
    return (ce * valid).sum() / n.clamp(min=1)


def _region_stats(logits: torch.Tensor, target: torch.Tensor):
    valid = (target != IGNORE).unsqueeze(1)  # (N,1,H,W)
    probs = logits.softmax(1) * valid
    onehot = F.one_hot(target.clamp(max=logits.shape[1] - 1), logits.shape[1]).permute(0, 3, 1, 2).to(probs.dtype) * valid
    dims = (0, 2, 3)
    tp = (probs * onehot).sum(dims)
    fp = (probs * (1 - onehot) * valid).sum(dims)
    fn = ((1 - probs) * onehot * valid).sum(dims)
    return tp, fp, fn


def masked_dice(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice over foreground classes (class 0 = background excluded); ignored pixels excluded."""
    tp, fp, fn = _region_stats(logits, target)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    return 1.0 - dice[1:].mean() if dice.numel() > 1 else 1.0 - dice.mean()


def masked_focal_tversky(
    logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Focal Tversky over foreground classes (beta > alpha favours recall for thin lanes)."""
    tp, fp, fn = _region_stats(logits, target)
    t = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    t = t[1:] if t.numel() > 1 else t
    # d/dx x**gamma is infinite at 0 (reached when no valid pixel / perfect overlap) -> clamp, and make
    # an all-ignored batch exactly zero so it cannot produce NaN gradients.
    loss = ((1.0 - t).clamp_min(1e-6) ** gamma).mean()
    return loss * (target != IGNORE).any()


class MultiTaskLoss:
    """Callable criterion for :class:`adas_mt.nn.model.MultiTaskModel`."""

    def __init__(self, model, w_da: float | None = None, w_ll: float | None = None):
        det = model.model[-1]
        self.det = E2ELoss(model) if getattr(det, "one2one_cv2", None) is not None else v8DetectionLoss(model)
        self.device = next(model.parameters()).device
        self.da_classes, self.ll_classes = model.da_classes, model.ll_classes
        args = getattr(model, "args", None)
        self.w_da = float(w_da if w_da is not None else getattr(args, "da_gain", 1.0))
        self.w_ll = float(w_ll if w_ll is not None else getattr(args, "ll_gain", 1.0))

    def update(self) -> None:
        """Per-epoch hook (E2ELoss decays its one-to-many weight)."""
        if hasattr(self.det, "update"):
            self.det.update()

    def seg_losses(self, preds: Dict[str, torch.Tensor], batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        da_t, ll_t = unpack_masks(batch["semantic_mask"].to(preds["da"].device), self.da_classes, self.ll_classes)
        da, ll = preds["da"].float(), preds["ll"].float()
        l_da = masked_ce(da, da_t) + masked_dice(da, da_t)
        if "da_aux" in preds:
            l_da = l_da + AUX_WEIGHT * masked_ce(preds["da_aux"].float(), da_t)
        l_ll = masked_ce(ll, ll_t) + masked_focal_tversky(ll, ll_t)
        return l_da, l_ll

    def __call__(self, preds, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        det_loss, det_items = self.det(preds["det"], batch)  # (3,) * batch_size, dict
        bs = batch["img"].shape[0]
        l_da, l_ll = self.seg_losses(preds, batch)
        loss = torch.cat([det_loss.reshape(-1), (self.w_da * l_da * bs).reshape(1), (self.w_ll * l_ll * bs).reshape(1)])
        items = dict(det_items)
        items["da_loss"], items["ll_loss"] = l_da.detach(), l_ll.detach()
        return loss, items
