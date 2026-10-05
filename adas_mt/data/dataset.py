"""Detection + drivable-area + lane dataset on top of Ultralytics' ``YOLODataset``.

Layout produced by ``convert_supervisely.py``::

    root/images/{train,val}/*.jpg
    root/labels_det/{train,val}/*.txt     YOLO boxes
    root/labels_da/{train,val}/*.png      uint8 class map (0=bg, 1=direct, 2=alternative); optional per image
    root/labels_ll/{train,val}/*.png      uint8 class map (0=bg, 1..N lane classes);       optional per image

A missing PNG means "this task is not annotated for this image" (ignored in the loss),
not "all background". DA + LL are packed into a single ``semantic_mask`` (see masks.py)
so upstream augmentation keeps boxes and both masks aligned.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from ultralytics.data.augment import Compose, Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths

from .masks import pack_masks
from .transforms import build_train_transforms, build_val_transforms


class MultiTaskFormat(Format):
    """Upstream detection ``Format`` that also tensorises the packed semantic mask."""

    def apply_image(self, labels, params=None):
        labels = super().apply_image(labels, params)
        m = labels.get("semantic_mask")
        if m is not None:
            labels["semantic_mask"] = torch.from_numpy(np.ascontiguousarray(m)).to(torch.int32)
        return labels


def _read_png(path: str) -> np.ndarray | None:
    import os

    if not os.path.isfile(path):
        return None  # genuinely unannotated
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:  # present but unreadable: fail loudly instead of silently training it as "unlabelled"
        raise OSError(f"corrupt mask file: {path}")
    return m


class MultiTaskDataset(YOLODataset):
    """Boxes + packed (DA, LL) semantic mask; rectangular ``imgsz=(h, w)`` aware."""

    format_class = MultiTaskFormat

    def __init__(self, *args, data: dict, imgsz: int | tuple[int, int] = (384, 640), **kwargs):
        if kwargs.get("rect"):
            # Ultralytics' DetectionTrainer.build_dataset passes rect=True for val, which letterboxes to
            # per-batch shapes (e.g. 384x672) instead of the 384x640 the deployed model sees.
            raise ValueError("MultiTaskDataset does not support rect=True; build val datasets with rect=False")
        hw = (int(imgsz), int(imgsz)) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
        self.imgsz_hw = hw
        self.da_classes = int(data["da_classes"])
        self.ll_classes = int(data["ll_classes"])
        super().__init__(*args, data=data, imgsz=max(hw), task="detect", **kwargs)  # load_image: long side -> max(hw)

    # ---- label files -----------------------------------------------------------------
    def get_label_files(self) -> list[str]:
        self.label_files = img2label_paths(self.im_files, label_dir="labels_det")
        return self.label_files

    # ---- transforms ------------------------------------------------------------------
    def build_transforms(self, hyp=None) -> Compose:
        hyp = hyp or self.hyp
        if self.augment:
            transforms = build_train_transforms(self, self.imgsz_hw, hyp)
        else:
            transforms = build_val_transforms(self.imgsz_hw)
        transforms.append(
            self.format_class(
                bbox_format="xywh", normalize=True, return_mask=False, return_keypoint=False, return_obb=False,
                batch_idx=True, mask_ratio=hyp.mask_ratio, mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    # ---- masks -----------------------------------------------------------------------
    def load_packed_mask(self, index: int, hw: tuple[int, int]) -> np.ndarray:
        """Packed mask resized (nearest) to the loaded image size ``hw``."""
        im_file = self.labels[index]["im_file"]  # label list may be shorter than the scanned file list
        da = _read_png(img2label_paths([im_file], label_dir="labels_da", suffix=".png")[0])
        ll = _read_png(img2label_paths([im_file], label_dir="labels_ll", suffix=".png")[0])
        packed = pack_masks(da, ll, self.da_classes, self.ll_classes, shape=_native_shape(da, ll, hw))
        if packed.shape != tuple(hw):
            packed = cv2.resize(packed, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        return packed

    def get_image_and_label(self, index: int) -> dict:
        label = super().get_image_and_label(index)
        label["semantic_mask"] = self.load_packed_mask(index, label["img"].shape[:2])
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return YOLODataset.collate_fn(batch)  # stacks img + semantic_mask, concatenates boxes/cls


def _native_shape(da, ll, fallback):
    for m in (da, ll):
        if m is not None:
            return m.shape[:2]
    return tuple(fallback)
