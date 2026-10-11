"""BEV head: image features -> ego-centred grid, grounded by depth where there is depth, by the ground plane
everywhere else.

Grid (``BEVGrid``): rows = forward distance from ``x_max`` (row 0) down to 0 m, columns = lateral from
``+y_max`` (left, column 0) to ``-y_max``; default 48 m x 32 m at 0.2 m -> 240 x 160 cells. Ego frame: x forward,
y left, z up, origin on the ground below the camera (``cam2ego`` carries the camera height / pitch / roll).

* :class:`DepthSplat` back-projects points on a stride-4 grid (stride-8 features upsampled x2; each point takes the
  mean of the valid depths in its 4x4 block) through ``K`` and ``cam2ego`` and averages the features landing in each
  BEV cell with ``scatter_add`` -> ONNX ``ScatterElements(reduction="add")``. All shapes are static; only the
  cell index is data-dependent. Invalid points go to a dump cell that is dropped. (Taking the *nearest* depth of
  a stride-8 block instead would pull ground points towards the camera by ~1 m at 10 m.)
* :class:`IPMSampler` does the reverse for every BEV cell: project the cell's ground point into the image and
  ``grid_sample`` the features (covers cells without depth: beyond stereo range, sky-free road far ahead).
* :class:`BEVDecoder` fuses both (plus their validity) and predicts 7 maps: drivable (3 logits), lane line (1),
  lane heading (sin, cos), agent occupancy (1, auxiliary).

``K`` is at network pixels (letterbox included) and is an input, as are ``cam2ego`` and ``depth``: one engine
serves any calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.data.schema import BEV_HW, BEV_RES_M
from sparc.nn.common import ConvBNAct, ResBlock

BEV_OUT = ("da_bg", "da_direct", "da_alt", "lane", "heading_sin", "heading_cos", "agent")


@dataclass(frozen=True)
class BEVGrid:
    x_max: float = BEV_HW[0] * BEV_RES_M
    y_max: float = BEV_HW[1] * BEV_RES_M / 2
    res: float = BEV_RES_M
    z_min: float = -1.0
    z_max: float = 3.0

    @property
    def hw(self):
        return int(round(self.x_max / self.res)), int(round(2 * self.y_max / self.res))

    def cell_of(self, x: torch.Tensor, y: torch.Tensor):
        """(row, col) float indices (floor them) of ego points."""
        return torch.floor((self.x_max - x) / self.res), torch.floor((self.y_max - y) / self.res)

    def centers(self) -> torch.Tensor:
        """(3, H*W) ego coordinates of the cell centres on the ground (z = 0)."""
        h, w = self.hw
        r, c = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        x = self.x_max - (r + 0.5) * self.res
        y = self.y_max - (c + 0.5) * self.res
        return torch.stack([x.flatten(), y.flatten(), torch.zeros(h * w)])


def _pixel_centres(h: int, w: int, stride: int, device, dtype):
    """Network-pixel coordinates of the centres of an (h, w) feature grid with the given stride."""
    v = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride - 0.5
    u = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return uu.flatten(), vv.flatten()


class DepthSplat(nn.Module):
    def __init__(self, c_in: int = 256, c: int = 64, feat_stride: int = 8, point_stride: int = 4,
                 grid: BEVGrid = BEVGrid()):
        super().__init__()
        if feat_stride % point_stride:
            raise ValueError("feat_stride must be a multiple of point_stride")
        self.reduce = ConvBNAct(c_in, c, 1)
        self.c, self.up, self.stride, self.grid = c, feat_stride // point_stride, point_stride, grid

    def cell_index(self, depth, K, cam2ego, pts_hw):
        """(B, N) flat BEV cell index per point of the stride-``point_stride`` grid (dump index H*W when invalid)."""
        g, s = self.grid, self.stride
        h, w = pts_hw
        valid_px = (depth > 0).to(depth.dtype)
        cnt = F.avg_pool2d(valid_px, s, s)
        d = (F.avg_pool2d(depth * valid_px, s, s) / cnt.clamp(min=1e-6)).flatten(1)  # mean valid depth per block
        valid = cnt.flatten(1) > 0
        u, v = _pixel_centres(h, w, s, depth.device, depth.dtype)
        fx, fy, cx, cy = K[:, 0, 0:1], K[:, 1, 1:2], K[:, 0, 2:3], K[:, 1, 2:3]
        p = torch.stack([(u[None] - cx) / fx * d, (v[None] - cy) / fy * d, d], 1)  # (B, 3, N) camera frame
        e = cam2ego[:, :3, :3] @ p + cam2ego[:, :3, 3:4]
        row, col = g.cell_of(e[:, 0], e[:, 1])
        H, W = g.hw
        valid = valid & (row >= 0) & (row < H) & (col >= 0) & (col < W) & (e[:, 2] >= g.z_min) & (e[:, 2] <= g.z_max)
        return torch.where(valid, row * W + col, torch.full_like(row, H * W)).long()

    def forward(self, feat, depth, K, cam2ego):
        f = self.reduce(feat)
        if self.up > 1:
            f = F.interpolate(f, scale_factor=float(self.up), mode="nearest")
        b, c, h, w = f.shape
        H, W = self.grid.hw
        idx = self.cell_index(depth, K, cam2ego, (h, w))
        src = torch.cat([f.flatten(2), torch.ones_like(f[:, :1].flatten(2))], 1)  # (B, C+1, N)
        acc = torch.zeros(b, c + 1, H * W + 1, dtype=f.dtype, device=f.device)
        acc = acc.scatter_add(2, idx[:, None].expand(b, c + 1, h * w), src)[:, :, : H * W]
        cnt = acc[:, c:]
        mean = acc[:, :c] / cnt.clamp(min=1.0)
        return torch.cat([mean, torch.log1p(cnt)], 1).view(b, c + 1, H, W)


class IPMSampler(nn.Module):
    def __init__(self, c_in: int = 256, c: int = 64, stride: int = 8, grid: BEVGrid = BEVGrid()):
        super().__init__()
        self.reduce = ConvBNAct(c_in, c, 1)
        self.stride, self.grid = stride, grid
        self.register_buffer("centres", grid.centers(), persistent=False)

    def forward(self, feat, K, cam2ego):
        f = self.reduce(feat)
        b, c, h, w = f.shape
        H, W = self.grid.hw
        img_h, img_w = h * self.stride, w * self.stride
        R, t = cam2ego[:, :3, :3], cam2ego[:, :3, 3:4]
        p = R.transpose(1, 2) @ (self.centres[None] - t)  # (B, 3, HW) camera frame
        z = p[:, 2]
        zs = z.clamp(min=0.1)
        u = K[:, 0, 0:1] * p[:, 0] / zs + K[:, 0, 2:3]
        v = K[:, 1, 1:2] * p[:, 1] / zs + K[:, 1, 2:3]
        gx = (u + 0.5) / img_w * 2 - 1
        gy = (v + 0.5) / img_h * 2 - 1
        valid = (z > 0.5) & (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
        s = F.grid_sample(f, torch.stack([gx, gy], -1)[:, None], mode="bilinear", padding_mode="zeros",
                          align_corners=False)[:, :, 0]  # (B, C, HW)
        m = valid.to(f.dtype)[:, None]
        return torch.cat([s * m, m], 1).view(b, c + 1, H, W)


class BEVDecoder(nn.Module):
    """Small U-shape: 240x160 -> 120x80 (64 ch) -> 60x40 (96 ch, residual blocks) -> back up with a skip."""

    def __init__(self, c_in: int, n_out: int = len(BEV_OUT), c1: int = 64, c2: int = 96, blocks: int = 3):
        super().__init__()
        self.stem = ConvBNAct(c_in, c1, 3, 2)
        self.down = ConvBNAct(c1, c2, 3, 2)
        self.blocks = nn.Sequential(*(ResBlock(c2) for _ in range(blocks)))
        self.fuse = ConvBNAct(c2 + c1, c1, 3)
        self.out = nn.Conv2d(c1, n_out, 1)

    def forward(self, x):
        s = self.stem(x)
        y = self.blocks(self.down(s))
        y = self.fuse(torch.cat([F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return F.interpolate(self.out(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


class BEVHead(nn.Module):
    def __init__(self, c_in: int = 256, c: int = 64, grid: BEVGrid = BEVGrid(), splat: bool = True, ipm: bool = True):
        super().__init__()
        if not (splat or ipm):
            raise ValueError("the BEV head needs the depth splat, the IPM branch or both")
        self.splat = DepthSplat(c_in, c, grid=grid) if splat else None
        self.ipm = IPMSampler(c_in, c, grid=grid) if ipm else None
        self.decoder = BEVDecoder((c + 1) * (int(splat) + int(ipm)))
        self.grid = grid

    def forward(self, feat, depth, K, cam2ego):
        parts = []
        if self.splat is not None:
            parts.append(self.splat(feat, depth, K, cam2ego))
        if self.ipm is not None:
            parts.append(self.ipm(feat, K, cam2ego))
        return self.decoder(torch.cat(parts, 1))
