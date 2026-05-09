"""FPN-fused segmentation decoder shared by drivable-area and lane heads.

Takes P3/P4/P5 from the YOLOv13 FPN, fuses them at P3 resolution, then
upsamples 3x back to the input resolution, producing per-class logits.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class FPNFusedSegDecoder(nn.Module):
    """Multi-scale FCN decoder.

    Args:
        ch_in: tuple of three input channel counts (C3, C4, C5).
        num_classes: number of output classes (including background).
        mid_ch: width of fused intermediate features. Default: 128.
    """

    def __init__(self, ch_in: Sequence[int], num_classes: int, mid_ch: int = 128):
        super().__init__()
        c3, c4, c5 = ch_in
        # 1x1 reductions to a common width before fusion
        self.lat_p3 = Conv(c3, mid_ch, 1, 1)
        self.lat_p4 = Conv(c4, mid_ch, 1, 1)
        self.lat_p5 = Conv(c5, mid_ch, 1, 1)
        # Fuse and decode: stride 8 -> stride 1
        self.fuse = Conv(mid_ch * 3, mid_ch, 3, 1)
        # 3 upsample blocks (each 2x), with channels mid -> mid/2 -> mid/2 -> mid/4
        self.up1 = Conv(mid_ch, mid_ch // 2, 3, 1)
        self.up2 = Conv(mid_ch // 2, mid_ch // 2, 3, 1)
        self.up3 = Conv(mid_ch // 2, max(mid_ch // 4, 16), 3, 1)
        self.classifier = nn.Conv2d(max(mid_ch // 4, 16), num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        p3, p4, p5 = feats
        x3 = self.lat_p3(p3)
        x4 = F.interpolate(self.lat_p4(p4), size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x5 = F.interpolate(self.lat_p5(p5), size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([x3, x4, x5], dim=1))  # (B, mid, H/8, W/8)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up1(x)  # H/4
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up2(x)  # H/2
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up3(x)  # H/1
        return self.classifier(x)
