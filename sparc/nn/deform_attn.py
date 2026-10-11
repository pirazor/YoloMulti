"""Multi-scale deformable attention (Deformable DETR) with ``grid_sample`` sampling.

This is RT-DETR's pure-PyTorch path: it exports to ONNX ``GridSample`` (opset >= 16), which the TensorRT ONNX
parser maps to its native grid-sample layer, so no plugin is needed. Phase 0 probes this on the device
(``bakeoff`` op probes); ``sample_mode="gather"`` is the fallback that does the bilinear interpolation with
``GatherElements`` only.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bilinear_gather(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """``grid_sample(value, grid, bilinear, zeros, align_corners=False)`` with gathers only.

    value (N, C, H, W), grid (N, Q, P, 2) in [-1, 1] -> (N, C, Q, P)."""
    n, c, h, w = value.shape
    _, q, p, _ = grid.shape
    x = (grid[..., 0] + 1) * (w * 0.5) - 0.5
    y = (grid[..., 1] + 1) * (h * 0.5) - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    flat = value.flatten(2)  # (N, C, H*W)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            wgt = (1 - (x - xi).abs()) * (1 - (y - yi).abs())
            valid = ((xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)).to(value.dtype)
            idx = (yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).long().flatten(1)  # (N, Q*P)
            g = torch.gather(flat, 2, idx[:, None, :].expand(n, c, q * p)).view(n, c, q, p)
            out = out + g * (wgt * valid)[:, None]
    return out


def ms_deform_attn(value: torch.Tensor, shapes: Sequence[Tuple[int, int]], loc: torch.Tensor, attn: torch.Tensor,
                   mode: str = "grid_sample") -> torch.Tensor:
    """value (B, S, H, C) over all levels, loc (B, Q, H, L, P, 2) in [0, 1], attn (B, Q, H, L, P) -> (B, Q, H*C)."""
    b, _, heads, c = value.shape
    _, q, _, levels, points, _ = loc.shape
    splits = [h * w for h, w in shapes]
    vals = value.split(splits, dim=1)
    grids = 2 * loc - 1
    sampled = []
    for lvl, (h, w) in enumerate(shapes):
        v = vals[lvl].flatten(2).transpose(1, 2).reshape(b * heads, c, h, w)
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)  # (B*H, Q, P, 2)
        if mode == "grid_sample":
            s = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        else:
            s = _bilinear_gather(v, g)
        sampled.append(s)  # (B*H, C, Q, P)
    a = attn.transpose(1, 2).reshape(b * heads, 1, q, levels * points)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * a).sum(-1)  # (B*H, C, Q)
    return out.view(b, heads * c, q).transpose(1, 2)


class MSDeformAttn(nn.Module):
    def __init__(self, d: int = 256, heads: int = 8, levels: int = 3, points: int = 4, mode: str = "grid_sample"):
        super().__init__()
        self.d, self.heads, self.levels, self.points, self.mode = d, heads, levels, points, mode
        self.offsets = nn.Linear(d, heads * levels * points * 2)
        self.weights = nn.Linear(d, heads * levels * points)
        self.value = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self._reset()

    def _reset(self):
        nn.init.zeros_(self.offsets.weight)
        th = torch.arange(self.heads, dtype=torch.float32) * (2 * math.pi / self.heads)
        grid = torch.stack([th.cos(), th.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.heads, 1, 1, 2).repeat(1, self.levels, self.points, 1)
        for i in range(self.points):
            grid[:, :, i] *= i + 1
        with torch.no_grad():
            self.offsets.bias.copy_(grid.flatten())
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def project_value(self, memory):
        b, s, _ = memory.shape
        return self.value(memory).view(b, s, self.heads, self.d // self.heads)

    def forward(self, query, ref, memory, shapes):
        """query (B, Q, d); ref (B, Q, 4) boxes cxcywh or (B, Q, 2) points, normalised; memory (B, S, d)."""
        return self.attend(query, ref, self.project_value(memory), shapes)

    def attend(self, query, ref, v, shapes):
        """Like ``forward`` with the value projection ``v`` (B, S, heads, d/heads) computed once and shared."""
        b, q, _ = query.shape
        off = self.offsets(query).view(b, q, self.heads, self.levels, self.points, 2)
        w = self.weights(query).view(b, q, self.heads, self.levels * self.points).softmax(-1)
        w = w.view(b, q, self.heads, self.levels, self.points)
        if ref.shape[-1] == 4:  # offsets scaled by box size (RT-DETR)
            loc = ref[:, :, None, None, None, :2] + off / self.points * ref[:, :, None, None, None, 2:] * 0.5
        else:
            norm = torch.tensor([[w_, h_] for h_, w_ in shapes], dtype=query.dtype, device=query.device)
            loc = ref[:, :, None, None, None, :] + off / norm[None, None, None, :, None, :]
        return self.out(ms_deform_attn(v, shapes, loc, w, self.mode))
