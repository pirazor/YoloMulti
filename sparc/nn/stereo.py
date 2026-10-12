"""In-network stereo depth: the right eye costs a light CNN stem and a correlation volume, not a second network.

Why it exists: on a stereo rig, metric depth is not free. A separate foundation stereo network (e.g. a
Fast-FoundationStereo engine) costs tens of milliseconds per frame on an Orin, more than the whole SPARC budget.
This head shares the frame budget with everything else:

1. both rectified eyes -> the same 3-conv stem (stride 4, 64 ch);
2. group-wise correlation volume over ``max_disp / 4`` disparity levels (built with shifts, static shapes);
3. 2D cost aggregation (the disparity dimension as channels, as in LightStereo) -> soft-argmin disparity at
   stride 4;
4. learnt convex upsampling (RAFT) to the input resolution -> sub-pixel disparity in network pixels.

Depth = ``fx * baseline / disparity`` with ``fx`` from the ``K`` input and the baseline as an input, so the engine is
calibration-independent. It is trained against offline depth from a large stereo model on the full-resolution
recordings (see ``docs/sparc/plan.md``); Phase 0 only measures its cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.nn.common import ConvBNAct


class StereoStem(nn.Module):
    def __init__(self, c: int = 48):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(3, 24, 3, 2), ConvBNAct(24, c, 3, 2), ConvBNAct(c, c, 3), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return self.net(x)


def gwc_volume(fl: torch.Tensor, fr: torch.Tensor, levels: int, groups: int) -> torch.Tensor:
    """Group-wise correlation: (B, C, h, w) x2 -> (B, groups * levels, h, w); level d compares left x with right x-d."""
    b, c, h, w = fl.shape
    cg = c // groups
    out = []
    for d in range(levels):
        if d == 0:
            prod = fl * fr
        else:
            prod = F.pad(fl[..., d:] * fr[..., :-d], (d, 0, 0, 0))
        out.append(prod.view(b, groups, cg, h, w).mean(2))
    return torch.stack(out, 2).flatten(1, 2)


def convex_upsample(x: torch.Tensor, mask: torch.Tensor, factor: int) -> torch.Tensor:
    """RAFT convex upsampling of (B, 1, h, w) with mask (B, 9*f*f, h, w); 3x3 neighbourhood by pad + slices."""
    b, _, h, w = x.shape
    m = mask.view(b, 9, factor * factor, h, w).softmax(1)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    nb = torch.stack([xp[:, 0, i:i + h, j:j + w] for i in range(3) for j in range(3)], 1)  # (B, 9, h, w)
    up = (m * nb[:, :, None]).sum(1)  # (B, f*f, h, w)
    return F.pixel_shuffle(up, factor)


class StereoHead(nn.Module):
    def __init__(self, max_disp: int = 192, groups: int = 8, c: int = 48, agg: int = 64):
        super().__init__()
        if max_disp % 4:
            raise ValueError("max_disp must be a multiple of 4")
        self.levels, self.groups = max_disp // 4, groups
        self.stem = StereoStem(c)
        self.agg_in = ConvBNAct(groups * self.levels, agg, 1)
        self.down = nn.Sequential(ConvBNAct(agg, 96, 3, 2), ConvBNAct(96, 96, 3))
        self.up = ConvBNAct(96, agg, 1)
        self.agg_out = nn.Sequential(ConvBNAct(agg + c, agg, 1), nn.Conv2d(agg, self.levels, 3, padding=1))
        self.mask = nn.Conv2d(agg + c, 9 * 16, 1)

    def forward(self, left, right):
        """Normalised rectified images (B, 3, H, W) -> disparity (B, 1, H, W) in network pixels."""
        fl, fr = self.stem(torch.cat([left, right], 0)).chunk(2, 0)
        cost = self.agg_in(gwc_volume(fl, fr, self.levels, self.groups))
        y = self.down(cost)
        y = cost + F.interpolate(self.up(y), size=cost.shape[-2:], mode="bilinear", align_corners=False)
        yc = torch.cat([y, fl], 1)
        logits = self.agg_out(yc)
        disp_levels = torch.arange(self.levels, dtype=logits.dtype, device=logits.device).view(1, -1, 1, 1)
        disp4 = (logits.softmax(1) * disp_levels).sum(1, keepdim=True)  # stride-4 pixels
        return convex_upsample(disp4 * 4.0, self.mask(yc), 4)

    @staticmethod
    def to_depth(disp, K, baseline, min_disp: float = 0.25):
        """Depth (m), 0 where the disparity is below ``min_disp`` network pixels (too far / invalid)."""
        bf = (K[:, 0, 0] * baseline.view(-1)).view(-1, 1, 1, 1)
        return torch.where(disp > min_disp, bf / disp.clamp(min=min_disp), torch.zeros_like(disp))
