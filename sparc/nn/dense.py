"""Dense heads on the high-resolution path: drivable area (deployed) and the lane-marking mask (training aux)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.nn.common import ConvBNAct


class DenseHead(nn.Module):
    """Encoder stride-8 map + backbone P2 (stride 4) -> class logits at the input resolution."""

    def __init__(self, n_cls: int, c_enc: int = 256, c_p2: int = 64, c: int = 32):
        super().__init__()
        self.lat = ConvBNAct(c_enc, c, 1)
        self.fuse = nn.Sequential(ConvBNAct(c + c_p2, c, 3), ConvBNAct(c, c, 3))
        self.out = nn.Conv2d(c, n_cls, 1)

    def forward(self, p2, p3):
        x = F.interpolate(self.lat(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.out(self.fuse(torch.cat([x, p2], 1)))
        return F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)
