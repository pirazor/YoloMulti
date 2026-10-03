"""Albumentations-based augmentation pipeline for the multi-task dataset.

Image + boxes + da_mask + ll_mask are kept synchronized by Albumentations'
``additional_targets={'da_mask': 'mask', 'll_mask': 'mask'}``. Geometric
transforms apply nearest-neighbor interpolation to masks (Albumentations does
this automatically for the 'mask' target type).
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

try:  # albumentations is a project requirement (pinned in requirements.txt)
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:  # pragma: no cover - surfaced at first use
    A = None
    ToTensorV2 = None


_ADDITIONAL_TARGETS = {"da_mask": "mask", "ll_mask": "mask"}
_BBOX_PARAMS_KW = dict(format="pascal_voc", label_fields=["cls"], min_visibility=0.1)


def _ensure_albu():
    if A is None:  # pragma: no cover
        raise RuntimeError(
            "albumentations is required for yolov13_multitask augmentation. "
            "Install with `pip install albumentations` or `pip install -r requirements.txt`."
        )


def build_train_transform(imgsz: int = 640, hyp: Dict | None = None):
    _ensure_albu()
    hyp = hyp or {}
    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=imgsz,
                min_width=imgsz,
                border_mode=cv2.BORDER_CONSTANT,
                fill=114,
                fill_mask=0,
                position="top_left",
            ),
            A.HorizontalFlip(p=float(hyp.get("fliplr", 0.5))),
            A.Affine(
                scale=(0.5, 1.5),
                translate_percent=(-0.1, 0.1),
                rotate=(-10, 10),
                shear={"x": (-2, 2), "y": (-2, 2)},
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=114,
                fill_mask=0,
                p=float(hyp.get("affine_p", 0.5)),
            ),
            A.HueSaturationValue(p=float(hyp.get("hsv_p", 0.5))),
            A.RandomBrightnessContrast(p=float(hyp.get("bc_p", 0.3))),
            A.MotionBlur(p=float(hyp.get("blur_p", 0.1))),
            A.GaussNoise(p=float(hyp.get("noise_p", 0.1))),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(**_BBOX_PARAMS_KW),
        additional_targets=_ADDITIONAL_TARGETS,
        seed=hyp.get("seed", None),
    )


def build_val_transform(imgsz: int = 640):
    _ensure_albu()
    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=imgsz,
                min_width=imgsz,
                border_mode=cv2.BORDER_CONSTANT,
                fill=114,
                fill_mask=0,
                position="top_left",
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(**_BBOX_PARAMS_KW),
        additional_targets=_ADDITIONAL_TARGETS,
    )


def apply(transform, sample: Dict) -> Dict:
    """Run Albumentations on a sample dict produced by ``MultiTaskDataset``.

    Returns a tensor-ready dict.
    """
    bboxes = sample["bboxes"]
    if bboxes is None or len(bboxes) == 0:
        bboxes_in: list = []
        cls_in: list = []
    else:
        # clip to valid pascal_voc range to avoid albu's strict validator
        h, w = sample["img"].shape[:2]
        b = np.asarray(bboxes, dtype=np.float32).copy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h - 1)
        keep = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
        b = b[keep]
        cls = np.asarray(sample["cls"]).astype(np.int64)[keep]
        bboxes_in = b.tolist()
        cls_in = cls.tolist()

    out = transform(
        image=sample["img"],
        da_mask=sample["da_mask"],
        ll_mask=sample["ll_mask"],
        bboxes=bboxes_in,
        cls=cls_in,
    )
    return out
