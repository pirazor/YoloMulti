"""Overlay helpers for multi-task predictions."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


# Drivable-area colors (RGB), index = class id (0=bg ignored)
DA_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (0, 200, 0),     # direct -> green
    2: (40, 80, 220),   # alternative -> blue-ish
}

# Lane palette; cycled through if more classes than colors.
LL_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 255, 0),    # solid -> yellow
    (255, 80, 255),   # dashed -> magenta
    (80, 255, 255),
    (255, 160, 80),
    (180, 255, 100),
    (255, 100, 100),
    (100, 100, 255),
)


def _color_for_lane(class_id: int) -> Tuple[int, int, int]:
    if class_id <= 0:
        return (0, 0, 0)
    return LL_PALETTE[(class_id - 1) % len(LL_PALETTE)]


def overlay_drivable_area(
    image_rgb: np.ndarray,
    da_mask: np.ndarray,
    alpha: float = 0.4,
    palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Blend a translucent colored overlay for the drivable-area mask."""
    palette = palette or DA_COLORS
    out = image_rgb.copy()
    color = np.zeros_like(out)
    for cls_id, rgb in palette.items():
        color[da_mask == cls_id] = rgb
    mask_present = (da_mask > 0).astype(np.float32)[..., None]
    out = (out.astype(np.float32) * (1 - alpha * mask_present)
           + color.astype(np.float32) * (alpha * mask_present)).clip(0, 255).astype(np.uint8)
    return out


def overlay_lane_mask(
    image_rgb: np.ndarray,
    ll_mask: np.ndarray,
    thickness: int = 2,
    use_contours: bool = True,
) -> np.ndarray:
    """Draw lane pixels on the image. Contours are preferred for clean lines;
    fall back to per-pixel colouring when contour extraction fails."""
    out = image_rgb.copy()
    classes = sorted(int(c) for c in np.unique(ll_mask) if c > 0)
    for cls_id in classes:
        m = (ll_mask == cls_id).astype(np.uint8)
        if use_contours:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                cv2.polylines(out, [cnt], isClosed=False, color=_color_for_lane(cls_id), thickness=thickness)
        else:
            out[m > 0] = _color_for_lane(cls_id)
    return out


def draw_boxes(
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: Sequence[int],
    confs: Sequence[float],
    names: Sequence[str],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = image_rgb.copy()
    for (x1, y1, x2, y2), c, conf in zip(boxes_xyxy, classes, confs):
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{names[int(c)] if 0 <= int(c) < len(names) else int(c)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_multitask(
    image_rgb: np.ndarray,
    da_mask: Optional[np.ndarray] = None,
    ll_mask: Optional[np.ndarray] = None,
    boxes_xyxy: Optional[np.ndarray] = None,
    cls: Optional[Sequence[int]] = None,
    conf: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    out = image_rgb
    if da_mask is not None:
        out = overlay_drivable_area(out, da_mask)
    if ll_mask is not None:
        out = overlay_lane_mask(out, ll_mask)
    if boxes_xyxy is not None and cls is not None:
        confs = list(conf) if conf is not None else [1.0] * len(boxes_xyxy)
        out = draw_boxes(out, np.asarray(boxes_xyxy), cls, confs, names or [])
    return out
