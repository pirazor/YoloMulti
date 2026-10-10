"""Rectangular-input augmentation pipeline built from Ultralytics 8.4 transforms.

Upstream Mosaic is square-only. ``RectMosaic`` generalises the 4-image mosaic to an
``(h, w)`` target (e.g. 384x640): a (2h, 2w) canvas is assembled, then
``RandomPerspective(size=(w, h))`` crops/zooms the central window, exactly like the
square case. Image, boxes and the packed semantic mask all go through upstream code
paths, so they stay aligned (nearest interpolation, 255 padding for masks).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from ultralytics.data.augment import (
    BaseMixTransform,
    Compose,
    LetterBox,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.utils import LOGGER

from .masks import IGNORE


class RectMosaic(Mosaic):
    """4-image mosaic on a (2h, 2w) canvas for a rectangular (h, w) target."""

    def __init__(self, dataset, hw: tuple[int, int], p: float = 1.0):
        super().__init__(dataset, imgsz=max(hw), p=p, n=4)
        self.h, self.w = int(hw[0]), int(hw[1])
        self.border = (-self.w // 2, -self.h // 2)  # (x, y) margins, as upstream

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        params = BaseMixTransform.get_params(self, labels)
        assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        h_t, w_t = self.h, self.w
        xc = int(random.uniform(-self.border[0], 2 * w_t + self.border[0]))
        yc = int(random.uniform(-self.border[1], 2 * h_t + self.border[1]))
        layout = []
        for i in range(4):
            patch = labels if i == 0 else labels["mix_labels"][i - 1]
            h, w = patch.get("resized_shape", patch["img"].shape[:2])
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w_t * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h_t * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w_t * 2), min(h_t * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            layout.append(
                dict(labels_patch=patch, x1a=x1a, y1a=y1a, x2a=x2a, y2a=y2a, x1b=x1b, y1b=y1b, x2b=x2b, y2b=y2b,
                     padw=x1a - x1b, padh=y1a - y1b, img_shape=(h, w))
            )
        params["layout"] = layout
        return params

    def apply_image(self, labels, params=None):
        canvas = np.full((self.h * 2, self.w * 2, labels["img"].shape[2]), 114, dtype=np.uint8)
        for it in params["layout"]:
            canvas[it["y1a"] : it["y2a"], it["x1a"] : it["x2a"]] = it["labels_patch"]["img"][
                it["y1b"] : it["y2b"], it["x1b"] : it["x2b"]
            ]
        labels["img"] = canvas
        return labels

    def apply_semantic(self, labels, params=None):
        if labels.get("semantic_mask") is None and all(m.get("semantic_mask") is None for m in labels.get("mix_labels", [])):
            return labels
        canvas = np.full((self.h * 2, self.w * 2), IGNORE, dtype=np.uint8)
        for it in params["layout"]:
            m = it["labels_patch"].get("semantic_mask")
            if m is None:
                continue
            canvas[it["y1a"] : it["y2a"], it["x1a"] : it["x2a"]] = m[it["y1b"] : it["y2b"], it["x1b"] : it["x2b"]]
        labels["semantic_mask"] = canvas
        return labels

    def _cat_labels(self, mosaic_labels):
        final = super()._cat_labels(mosaic_labels)  # clips to a (2*imgsz)^2 square; redo for (2h, 2w)
        if not final:
            return final
        final["resized_shape"] = (self.h * 2, self.w * 2)
        final["instances"].clip(self.w * 2, self.h * 2)
        good = final["instances"].remove_zero_area_boxes()
        final["cls"] = final["cls"][good]
        return final


UNSUPPORTED_AUG = ("mixup", "cutmix", "copy_paste", "flipud")  # upstream keys this pipeline does not implement


def build_train_transforms(dataset, hw: tuple[int, int], hyp) -> Compose:
    """Rectangular training pipeline: [RectMosaic] -> RandomPerspective -> HSV -> flip."""
    ignored = [k for k in UNSUPPORTED_AUG if getattr(hyp, k, 0)]
    if ignored:
        LOGGER.warning(f"{ignored} > 0 but the multi-task pipeline implements only mosaic / affine / HSV / fliplr: ignored")
    h, w = hw
    affine = RandomPerspective(
        degrees=hyp.degrees, translate=hyp.translate, scale=hyp.scale, shear=hyp.shear,
        perspective=hyp.perspective, size=(w, h),
    )
    steps = []
    if hyp.mosaic > 0:
        steps.append(RectMosaic(dataset, hw, p=hyp.mosaic))
    steps += [
        affine,
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(p=hyp.fliplr, direction="horizontal"),
    ]
    return Compose(steps)


def build_val_transforms(hw: tuple[int, int]) -> Compose:
    """Centered letterbox to (h, w); masks are padded with 255 (ignore)."""
    return Compose([LetterBox(new_shape=tuple(hw), scaleup=False)])
