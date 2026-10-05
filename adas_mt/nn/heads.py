"""Light segmentation heads for drivable area (DA) and lane (LL) on a YOLO26 neck.

Design (see docs/architecture.md):
  * DA   : large smooth regions -> stride-8 head (P3 + upsampled P4), logits upsampled to input
           resolution. Training-only auxiliary classifier on P4 (deep supervision, dropped at export).
  * Lane : thin structures -> stride-4 head fusing the backbone P2 skip with the neck P3/P4 (top-down
           sum), sub-pixel (PixelShuffle x4) classifier so thin lines stay sharp at full resolution.
Everything is depthwise-separable 3x3 + 1x1, so the whole segmentation side costs ~1 GFLOP at 384x640.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, DWConv


def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


class DAHead(nn.Module):
    """Drivable-area head: ``(p3, p4) -> logits (B, nc, H, W)`` (+ aux logits while training)."""

    def __init__(self, ch: Sequence[int], nc: int, mid: int | None = None):
        super().__init__()
        c3, c4 = ch
        self.nc = nc
        mid = mid or max(32, c3 // 2)
        self.lat3 = Conv(c3, mid, 1)
        self.lat4 = Conv(c4, mid, 1)
        self.mix = nn.Sequential(DWConv(mid, mid, 3), Conv(mid, mid, 1))
        self.cls = nn.Conv2d(mid, nc, 1)
        self.aux = nn.Sequential(Conv(c4, mid, 1), nn.Conv2d(mid, nc, 1))  # training only

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, size: Sequence[int]):
        x = self.lat3(p3)
        x = x + _up(self.lat4(p4), x)
        logits = F.interpolate(self.cls(self.mix(x)), size=tuple(size), mode="bilinear", align_corners=False)
        if self.training and self.aux is not None:
            aux = F.interpolate(self.aux(p4), size=tuple(size), mode="bilinear", align_corners=False)
            return logits, aux
        return logits, None


class LaneHead(nn.Module):
    """Lane head: ``(p2, p3, p4) -> logits (B, nc, H, W)`` through a stride-4 fusion + PixelShuffle(4)."""

    SCALE = 4

    def __init__(self, ch: Sequence[int], nc: int, mid: int | None = None, bg_prior: float = 0.99):
        super().__init__()
        c2, c3, c4 = ch
        self.nc = nc
        mid = mid or max(32, c3 // 2)
        self.lat2 = Conv(c2, mid, 1)
        self.lat3 = Conv(c3, mid, 1)
        self.lat4 = Conv(c4, mid, 1)
        self.mix = nn.Sequential(DWConv(mid, mid, 3), Conv(mid, mid, 1))
        r2 = self.SCALE**2
        self.cls = nn.Conv2d(mid, nc * r2, 1)
        self.shuffle = nn.PixelShuffle(self.SCALE)
        self._init_bias(bg_prior)

    def _init_bias(self, bg_prior: float) -> None:
        """Start from 'almost everything is background' so lane pixels (~1%) do not dominate early loss."""
        r2 = self.SCALE**2
        bias = torch.zeros(self.nc * r2)
        if self.nc > 1:
            bias[:r2] = math.log(bg_prior * (self.nc - 1) / (1.0 - bg_prior))  # class 0 = background
        with torch.no_grad():
            self.cls.bias.copy_(bias)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, size: Sequence[int]):
        x = self.lat4(p4)
        x = self.lat3(p3) + _up(x, p3)
        x = self.lat2(p2) + _up(x, p2)
        logits = self.shuffle(self.cls(self.mix(x)))
        if tuple(logits.shape[-2:]) != tuple(size):  # input not a multiple of 4 -> match exactly
            logits = F.interpolate(logits, size=tuple(size), mode="bilinear", align_corners=False)
        return logits
