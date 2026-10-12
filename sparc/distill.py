"""DINOv3 teacher on the encoder's stride-16 map (training only; reuses adas_mt's teacher and loss terms).

The frozen teacher sees the same RGB batch; its patch tokens (patch 16 -> the 24x40 grid at 384x640) are compared
with a projection of the encoder's P4 map by the cosine + affinity terms of ``adas_mt.distill.loss``. The weight
decays with a cosine schedule over training (RT-DETRv4 / DEIMv2 style: the student is pulled towards foundation
features early, then left to the task losses).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adas_mt.distill.loss import KDProjector, distill_terms
from adas_mt.distill.teacher import FrozenTeacher


class DistillHook(nn.Module):
    def __init__(self, teacher: str = "dinov3_b", checkpoint: Optional[str] = None, c_student: int = 256,
                 weight: float = 1.0, affinity: float = 1.0, pretrained: bool = True, dtype: str = "auto"):
        super().__init__()
        self.teacher = FrozenTeacher(teacher, pretrained=pretrained, checkpoint=checkpoint, dtype=dtype)
        self.proj = KDProjector(c_student, self.teacher.dim)
        self.weight, self.affinity = weight, affinity

    def schedule(self, progress: float) -> float:
        """Cosine decay of the weight with training progress in [0, 1]."""
        return self.weight * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    def forward(self, image01: torch.Tensor, p4: torch.Tensor, progress: float) -> Dict[str, torch.Tensor]:
        """``image01`` (B, 3, H, W) in [0, 1]; ``p4`` the encoder's stride-16 map (B, C, H/16, W/16)."""
        with torch.no_grad():
            t, (gh, gw) = self.teacher(image01)
        z = self.proj(p4)
        if z.shape[-2:] != (gh, gw):
            z = F.interpolate(z, size=(gh, gw), mode="bilinear", align_corners=False)
        z = z.flatten(2).transpose(1, 2)
        if not torch.isfinite(t).all():
            return {"kd": z.sum() * 0.0, "kd_cos": z.sum() * 0.0, "kd_weight": torch.tensor(0.0)}
        cos, aff = distill_terms(z.float(), t)
        w = self.schedule(progress)
        return {"kd": w * (cos + self.affinity * aff), "kd_cos": cos.detach(), "kd_aff": aff.detach(),
                "kd_weight": torch.tensor(w)}
