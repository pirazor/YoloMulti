"""Dense feature distillation: frozen foundation-model patch tokens -> YOLO neck features.

Student side: neck P3 (avg-pooled to stride 16) concatenated with neck P4 -> 1x1-BN-SiLU-1x1 projector
(``model.kd_proj``, a normal submodule so any optimizer/EMA picks it up; removed by
``MultiTaskModel.strip_training_only()`` before export) -> token grid of the teacher.
Loss: per-token cosine distance + a token-affinity (relational) term on a random subset of tokens, scaled
by a cosine-decayed weight.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .teacher import FrozenTeacher


class KDProjector(nn.Module):
    def __init__(self, c_in: int, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(256, min(dim, 512))
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(), nn.Conv2d(hidden, dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def distill_terms(z: torch.Tensor, t: torch.Tensor, n_aff: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """z, t: (B, N, D). Returns (cosine distance, affinity MSE)."""
    cos = (1.0 - F.cosine_similarity(z, t, dim=-1)).mean()
    n = z.shape[1]
    idx = torch.randperm(n, device=z.device)[: min(n_aff, n)]
    zs, ts = F.normalize(z[:, idx], dim=-1), F.normalize(t[:, idx], dim=-1)
    sz, st = zs @ zs.transpose(1, 2), ts @ ts.transpose(1, 2)
    # Relational term: compare the *structure* of the token-affinity matrices. Centering removes the global
    # offset (DINO tokens share a strong common component; the cosine term already covers it) and dividing
    # by the teacher's centred power makes the term scale-free (~1 for an unrelated student, 0 when
    # matched). A raw MSE would be ~0.02 (negligible); dividing the uncentred MSE by the variance explodes.
    sz = sz - sz.mean((1, 2), keepdim=True)
    st = st - st.mean((1, 2), keepdim=True)
    aff = F.mse_loss(sz, st) / st.pow(2).mean().clamp_min(1e-2)  # floor: weak-structure teachers cannot blow it up
    return cos, aff


class Distiller:
    """Criterion-side helper (not an ``nn.Module``): owns the teacher, registers ``model.kd_proj``."""

    def __init__(
        self,
        model,
        teacher: FrozenTeacher,
        w_cos: float = 1.0,
        w_aff: float = 1.0,
        weight: float = 1.0,
        weight_end: float = 0.1,
        every: int = 1,
        n_aff_tokens: int = 256,
    ):
        self.teacher = teacher
        self.w_cos, self.w_aff = w_cos, w_aff
        self.w0, self.w1 = float(weight), float(weight_end)
        self.every = max(1, int(every))
        self.n_aff = n_aff_tokens
        self.progress = 0.0
        self._calls = 0
        self.model = model
        if not hasattr(model, "kd_proj"):
            raise RuntimeError(
                "model has no kd_proj: build the model with kd_dim=teacher.dim (or call "
                "model.attach_kd_projector) BEFORE creating the optimizer/EMA/DDP wrapper; a projector added "
                "later is never optimised, never EMA'd and drifts across DDP ranks"
            )
        if model.kd_proj.net[-1].out_channels != teacher.dim:
            raise ValueError("model.kd_proj width does not match the teacher")
        self.teacher.to(next(model.parameters()).device)

    # lambda(t): cosine decay weight -> weight_end over training progress in [0, 1]
    def set_progress(self, p: float) -> None:
        self.progress = min(max(float(p), 0.0), 1.0)

    @property
    def lam(self) -> float:
        return self.w1 + 0.5 * (self.w0 - self.w1) * (1.0 + math.cos(math.pi * self.progress))

    def student_tokens(self, p3: torch.Tensor, p4: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        s = torch.cat([F.adaptive_avg_pool2d(p3, p4.shape[-2:]), p4], 1)
        z = self.model.kd_proj(s)
        if tuple(z.shape[-2:]) != tuple(grid):
            z = F.interpolate(z, size=grid, mode="bilinear", align_corners=False) if (
                grid[0] * grid[1] > z.shape[-1] * z.shape[-2]
            ) else F.adaptive_avg_pool2d(z, grid)
        return z.flatten(2).transpose(1, 2).float()

    def loss_from_feats(self, p3, p4, img) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``img``: (B,3,H,W) RGB, uint8 or float in [0,1]. Returns (weighted loss, items)."""
        self._calls += 1
        if (self._calls - 1) % self.every:  # skipped step: no teacher forward
            # keep the projector in the autograd graph: DDP raises on parameters that received no gradient
            zero = self.model.kd_proj(torch.cat([F.adaptive_avg_pool2d(p3, p4.shape[-2:]), p4], 1)).sum() * 0.0
            return zero, {"kd_loss": zero.detach(), "kd_cos": zero.detach()}
        img = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
        if next(self.teacher.parameters()).device != img.device:  # e.g. model moved after the distiller was built
            self.teacher.to(img.device)
        t, grid = self.teacher(img)
        z = self.student_tokens(p3, p4, grid)
        cos, aff = distill_terms(z, t, self.n_aff)
        loss = self.w_cos * cos + self.w_aff * aff
        return self.lam * loss, {"kd_loss": loss.detach(), "kd_cos": (1.0 - cos).detach()}

    def __call__(self, preds: Dict, batch: Dict):
        p3, p4 = preds["feats"]
        return self.loss_from_feats(p3, p4, batch["img"])
