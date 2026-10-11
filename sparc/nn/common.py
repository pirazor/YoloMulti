"""Small building blocks shared by the SPARC modules (all TensorRT-friendly: Conv/BN/SiLU/MatMul/Softmax)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True, groups: int = 1):
        super().__init__(nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(c2),
                         nn.SiLU(inplace=True) if act else nn.Identity())


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.a, self.b = ConvBNAct(c, c, 3), ConvBNAct(c, c, 3, act=False)

    def forward(self, x):
        return F.silu(x + self.b(self.a(x)))


class CSPBlock(nn.Module):
    """CSP block (RT-DETR's RepC3 without re-parameterisation): split, n residual 3x3 blocks, merge."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 1.0):
        super().__init__()
        h = int(c2 * e)
        self.cv1, self.cv2 = ConvBNAct(c1, h, 1), ConvBNAct(c1, h, 1)
        self.m = nn.Sequential(*(ResBlock(h) for _ in range(n)))
        self.cv3 = ConvBNAct(h, c2, 1) if h != c2 else nn.Identity()

    def forward(self, x):
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class MLP(nn.Module):
    def __init__(self, c_in: int, c_hidden: int, c_out: int, n: int):
        super().__init__()
        dims = [c_in] + [c_hidden] * (n - 1) + [c_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x / |x|`` along the last dim with Mul/ReduceSum/Sqrt only (``Tensor.norm`` exports to ReduceL2)."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True)).clamp_min(eps)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def sincos_2d(h: int, w: int, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """(1, h*w, dim) fixed 2D sin-cos position embedding (RT-DETR AIFI)."""
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    d = dim // 4
    omega = 1.0 / temperature ** (torch.arange(d, dtype=torch.float32) / d)
    ox, oy = gx.flatten()[:, None] * omega, gy.flatten()[:, None] * omega
    return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)[None]


def ln2d(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    """RGB in [0, 1] -> ImageNet-normalised (inside the graph, so the device preprocessing stays /255 only)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


def bias_init_with_prob(p: float) -> float:
    return float(-math.log((1 - p) / p))
