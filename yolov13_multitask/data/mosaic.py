"""4-image mosaic that composites images, bboxes, and TWO raster masks together.

Geometry mirrors ``ultralytics.data.augment.Mosaic._mosaic4`` (line 658) but
operates directly on raster masks (PNG uint8) which v8's pipeline does not
handle natively. We use this as a pre-Albumentations stage; the resulting
2*imgsz canvas is then fed to the standard albu Compose (which itself includes
a final crop + resize).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import numpy as np


def _xyxy_offset(bboxes: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if bboxes.size == 0:
        return bboxes
    out = bboxes.astype(np.float32, copy=True)
    out[:, [0, 2]] += dx
    out[:, [1, 3]] += dy
    return out


def mosaic4(
    samples: List[Dict],
    imgsz: int,
    rng: random.Random,
    pad_value: int = 114,
) -> Dict:
    """Build a 2*imgsz x 2*imgsz mosaic from 4 samples.

    Each ``sample`` dict must contain:
        img     : HxWx3 uint8
        bboxes  : (N, 4) float xyxy in pixel coords on this sample's image
        cls     : (N,) int
        da_mask : HxW uint8 (zeros if absent)
        ll_mask : HxW uint8 (zeros if absent)
        has_da, has_ll : bool
    Returns a dict with keys img, bboxes, cls, da_mask, ll_mask, has_da, has_ll.
    """
    assert len(samples) == 4, "mosaic4 requires exactly 4 samples"
    s = imgsz
    canvas = np.full((2 * s, 2 * s, 3), pad_value, dtype=np.uint8)
    da = np.zeros((2 * s, 2 * s), dtype=np.uint8)
    ll = np.zeros((2 * s, 2 * s), dtype=np.uint8)

    cx = rng.randint(int(s * 0.5), int(s * 1.5))
    cy = rng.randint(int(s * 0.5), int(s * 1.5))

    out_boxes: List[np.ndarray] = []
    out_cls: List[np.ndarray] = []

    has_da_any = False
    has_ll_any = False

    # Layout: [top-left, top-right, bot-left, bot-right]
    for i, sample in enumerate(samples):
        img = sample["img"]
        h, w = img.shape[:2]
        # scale image to fit in s x s
        scale = min(s / h, s / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        if (nh, nw) != (h, w):
            import cv2

            img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            da_r = cv2.resize(sample["da_mask"], (nw, nh), interpolation=cv2.INTER_NEAREST) if sample["has_da"] else np.zeros((nh, nw), dtype=np.uint8)
            ll_r = cv2.resize(sample["ll_mask"], (nw, nh), interpolation=cv2.INTER_NEAREST) if sample["has_ll"] else np.zeros((nh, nw), dtype=np.uint8)
        else:
            img_r = img
            da_r = sample["da_mask"] if sample["has_da"] else np.zeros((nh, nw), dtype=np.uint8)
            ll_r = sample["ll_mask"] if sample["has_ll"] else np.zeros((nh, nw), dtype=np.uint8)
        # rescale boxes
        if sample["bboxes"].size:
            boxes = sample["bboxes"].astype(np.float32, copy=True) * scale
        else:
            boxes = sample["bboxes"]

        # Determine paste region in the 2s x 2s canvas, with cropping if quadrant
        # is too small to fit the whole resized sample.
        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(cx - nw, 0), max(cy - nh, 0), cx, cy
            x1b, y1b, x2b, y2b = nw - (x2a - x1a), nh - (y2a - y1a), nw, nh
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = cx, max(cy - nh, 0), min(cx + nw, 2 * s), cy
            x1b, y1b, x2b, y2b = 0, nh - (y2a - y1a), x2a - x1a, nh
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(cx - nw, 0), cy, cx, min(2 * s, cy + nh)
            x1b, y1b, x2b, y2b = nw - (x2a - x1a), 0, nw, y2a - y1a
        else:  # bottom-right
            x1a, y1a, x2a, y2a = cx, cy, min(cx + nw, 2 * s), min(2 * s, cy + nh)
            x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a

        # Paste image
        canvas[y1a:y2a, x1a:x2a] = img_r[y1b:y2b, x1b:x2b]
        if sample["has_da"]:
            da[y1a:y2a, x1a:x2a] = da_r[y1b:y2b, x1b:x2b]
            has_da_any = True
        if sample["has_ll"]:
            ll[y1a:y2a, x1a:x2a] = ll_r[y1b:y2b, x1b:x2b]
            has_ll_any = True

        # Translate boxes to mosaic coords and clip to its quadrant.
        if isinstance(boxes, np.ndarray) and boxes.size:
            dx = x1a - x1b
            dy = y1a - y1b
            b = _xyxy_offset(boxes, dx, dy)
            # clip
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, 2 * s)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, 2 * s)
            keep = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
            if keep.any():
                out_boxes.append(b[keep])
                out_cls.append(sample["cls"][keep])

    bboxes = np.concatenate(out_boxes, axis=0) if out_boxes else np.zeros((0, 4), dtype=np.float32)
    cls = np.concatenate(out_cls, axis=0) if out_cls else np.zeros((0,), dtype=np.int64)

    return {
        "img": canvas,
        "bboxes": bboxes,
        "cls": cls,
        "da_mask": da,
        "ll_mask": ll,
        "has_da": has_da_any,
        "has_ll": has_ll_any,
    }
