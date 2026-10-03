"""Multi-task loss = v8 detection loss + drivable-area seg loss + lane seg loss.

L_total = w_det * L_det + w_da * L_da + w_ll * L_ll

with:
    L_det = upstream v8DetectionLoss (returns scalar already-summed-with-internal-gains)
    L_da  = mean( CE(da_logits, da_gt) + Dice(...) )    over samples with has_da=True
    L_ll  = mean( CE(ll_logits, ll_gt) + FocalTversky(...) )   over samples with has_ll=True

Optional Kendall-style learnable uncertainty weighting (toggled via
``model.use_uncertainty_weighting``):
    L_total = sum_i 0.5 * exp(-2*sigma_i) * L_i + sigma_i

Loss-items vector is length 5: (box, cls, dfl, da, ll) — matching the
``loss_names`` declared by ``MultiTaskTrainer``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class DiceLoss(nn.Module):
    """Multi-class Dice loss on softmax logits.

    Background class (id 0) is included in the average by default; pass
    ``ignore_bg=True`` to skip it (useful when bg dominates pixel area).
    """

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W); target: (N, H, W) long
        probs = logits.softmax(dim=1)
        target_oh = F.one_hot(target.clamp(0, self.num_classes - 1), self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).to(probs.dtype)
        dims = (0, 2, 3)
        inter = (probs * target_oh).sum(dim=dims)
        denom = probs.sum(dim=dims) + target_oh.sum(dim=dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        if self.ignore_bg and self.num_classes > 1:
            dice = dice[1:]
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky for sparse classes (e.g. lanes).

    alpha penalises FP, beta penalises FN. With alpha < beta the loss is
    biased towards higher recall (preferred for thin lane lines).
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
        ignore_bg: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        target_oh = F.one_hot(target.clamp(0, self.num_classes - 1), self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).to(probs.dtype)
        dims = (0, 2, 3)
        tp = (probs * target_oh).sum(dim=dims)
        fp = (probs * (1 - target_oh)).sum(dim=dims)
        fn = ((1 - probs) * target_oh).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if self.ignore_bg and self.num_classes > 1:
            tversky = tversky[1:]
        return ((1.0 - tversky) ** self.gamma).mean()


class MultiTaskLoss:
    """Callable criterion for ``YOLOv13MultiTask``.

    ``__call__(preds, batch) -> (scalar_loss * batch_size, items_detached)``
    matching the upstream contract of ``v8DetectionLoss``.
    """

    def __init__(self, model):
        self.model = model
        self.det_loss = v8DetectionLoss(model)
        self.device = self.det_loss.device

        self.da_classes = int(getattr(model, "da_classes", 3))
        self.ll_classes = int(getattr(model, "ll_classes", 3))

        # Loss weights from model.args.loss_weights if present, else (1,1,1).
        lw = getattr(getattr(model, "args", None), "loss_weights", None) or {}
        if isinstance(lw, dict):
            self.w_det = float(lw.get("det", 1.0))
            self.w_da = float(lw.get("da", 1.0))
            self.w_ll = float(lw.get("ll", 1.0))
        else:
            self.w_det = self.w_da = self.w_ll = 1.0

        self.ce_da = nn.CrossEntropyLoss()
        self.dice_da = DiceLoss(self.da_classes, ignore_bg=True)
        self.ce_ll = nn.CrossEntropyLoss()
        self.ftv_ll = FocalTverskyLoss(self.ll_classes, ignore_bg=True)

        self.use_uncertainty = bool(getattr(model, "use_uncertainty_weighting", False))
        if self.use_uncertainty:
            self.log_sigma = nn.Parameter(torch.zeros(3, device=self.device))
            # Register as a true model parameter so it is part of optim state.
            model.register_parameter("multitask_log_sigma", self.log_sigma)

    # ---------------------------------------------------------------- internals
    def _seg_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        has_mask: torch.Tensor,
        ce: nn.CrossEntropyLoss,
        secondary: nn.Module,
    ) -> torch.Tensor:
        """Compute per-task segmentation loss, masking out samples without GT."""
        if logits.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        if has_mask is None:
            valid_logits, valid_target = logits, target
        else:
            valid = has_mask.bool()
            if not valid.any():
                return torch.zeros((), device=logits.device, dtype=logits.dtype)
            valid_logits = logits[valid]
            valid_target = target[valid]
        # Match logits spatial size to target if needed (e.g., during ablations).
        if valid_logits.shape[-2:] != valid_target.shape[-2:]:
            valid_logits = F.interpolate(
                valid_logits, size=valid_target.shape[-2:], mode="bilinear", align_corners=False
            )
        ce_l = ce(valid_logits, valid_target.long())
        sec_l = secondary(valid_logits, valid_target.long())
        return ce_l + sec_l

    # ------------------------------------------------------------------- callable
    def __call__(self, preds, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        det_preds = preds["det"] if isinstance(preds, dict) else preds[0]
        da_logits = preds["da"] if isinstance(preds, dict) else preds[1]
        ll_logits = preds["ll"] if isinstance(preds, dict) else preds[2]

        loss_det_scalar, items_det = self.det_loss(det_preds, batch)
        # v8DetectionLoss returns ``loss.sum() * batch_size`` -- already scalar.

        da_target = batch["da_mask"].to(da_logits.device)
        ll_target = batch["ll_mask"].to(ll_logits.device)
        has_da = batch.get("has_da")
        has_ll = batch.get("has_ll")
        if has_da is not None:
            has_da = has_da.to(da_logits.device)
        if has_ll is not None:
            has_ll = has_ll.to(ll_logits.device)

        l_da = self._seg_loss(da_logits, da_target, has_da, self.ce_da, self.dice_da)
        l_ll = self._seg_loss(ll_logits, ll_target, has_ll, self.ce_ll, self.ftv_ll)

        # The det loss is already multiplied by batch_size; scale our seg losses
        # the same way so total stays comparable across batches.
        bs = batch["img"].shape[0]
        l_da_scaled = l_da * bs
        l_ll_scaled = l_ll * bs

        if self.use_uncertainty:
            s = self.log_sigma
            total = (
                0.5 * torch.exp(-2 * s[0]) * loss_det_scalar
                + 0.5 * torch.exp(-2 * s[1]) * l_da_scaled
                + 0.5 * torch.exp(-2 * s[2]) * l_ll_scaled
                + s.sum()
            )
        else:
            total = self.w_det * loss_det_scalar + self.w_da * l_da_scaled + self.w_ll * l_ll_scaled

        items = torch.cat(
            [
                items_det.detach(),
                l_da.detach().reshape(1),
                l_ll.detach().reshape(1),
            ]
        )
        return total, items
