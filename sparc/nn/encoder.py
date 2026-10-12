"""RT-DETR hybrid encoder: one transformer layer on P5 (AIFI) + a CNN top-down / bottom-up fusion (CCFM).

In: backbone P3..P5 (any channels). Out: three 256-channel maps at strides 8 / 16 / 32.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.nn.common import ConvBNAct, CSPBlock, sincos_2d


class AIFI(nn.Module):
    def __init__(self, d: int = 256, heads: int = 8, ffn: int = 1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        pos = sincos_2d(h, w, c).to(x)
        q = t + pos
        t = self.n1(t + self.attn(q, q, t, need_weights=False)[0])
        t = self.n2(t + self.ffn(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class HybridEncoder(nn.Module):
    def __init__(self, in_channels: Sequence[int], d: int = 256, depth: int = 1, expansion: float = 0.5):
        super().__init__()
        self.proj = nn.ModuleList(ConvBNAct(c, d, 1, act=False) for c in in_channels)
        self.aifi = AIFI(d)
        self.lat = nn.ModuleList(ConvBNAct(d, d, 1) for _ in range(2))
        self.td = nn.ModuleList(CSPBlock(2 * d, d, depth, expansion) for _ in range(2))
        self.down = nn.ModuleList(ConvBNAct(d, d, 3, 2) for _ in range(2))
        self.bu = nn.ModuleList(CSPBlock(2 * d, d, depth, expansion) for _ in range(2))
        self.d = d

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = (p(f) for p, f in zip(self.proj, feats))
        p5 = self.aifi(p5)
        # top-down
        l5 = self.lat[0](p5)
        t4 = self.td[0](torch.cat([F.interpolate(l5, scale_factor=2.0, mode="nearest"), p4], 1))
        l4 = self.lat[1](t4)
        o3 = self.td[1](torch.cat([F.interpolate(l4, scale_factor=2.0, mode="nearest"), p3], 1))
        # bottom-up
        o4 = self.bu[0](torch.cat([self.down[0](o3), l4], 1))
        o5 = self.bu[1](torch.cat([self.down[1](o4), l5], 1))
        return [o3, o4, o5]
