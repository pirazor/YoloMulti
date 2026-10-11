"""Box utilities (normalised cxcywh <-> xyxy, IoU, GIoU), batched, differentiable."""

from __future__ import annotations

import torch


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)


def box_area(b: torch.Tensor) -> torch.Tensor:
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xyxy (N, 4) x (M, 4) -> (N, M) IoU."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / union.clamp(min=1e-9)


def pairwise_giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xyxy (N, 4) x (M, 4) -> (N, M) generalised IoU."""
    iou = pairwise_iou(a, b)
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb - lt).clamp(min=0).prod(-1)
    inter = iou * (box_area(a)[:, None] + box_area(b)[None, :]) / (1 + iou)  # inter = iou * union, union = A + B - inter
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return iou - (hull - union) / hull.clamp(min=1e-9)


def elementwise_giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xyxy (N, 4) and (N, 4) -> (N,) GIoU of corresponding boxes."""
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(a) + box_area(b) - inter
    iou = inter / union.clamp(min=1e-9)
    hl = torch.min(a[:, :2], b[:, :2])
    hr = torch.max(a[:, 2:], b[:, 2:])
    hull = (hr - hl).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-9)


def elementwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    return inter / (box_area(a) + box_area(b) - inter).clamp(min=1e-9)
