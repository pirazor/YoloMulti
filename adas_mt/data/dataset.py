"""Detection + drivable-area + lane dataset on top of Ultralytics' ``YOLODataset``.

Layout produced by ``convert_supervisely.py``::

    root/images/{train,val}/*.jpg
    root/labels_det/{train,val}/*.txt     YOLO boxes
    root/labels_da/{train,val}/*.png      uint8 class map (0=bg, 1=direct, 2=alternative); optional per image
    root/labels_ll/{train,val}/*.png      uint8 class map (0=bg, 1..N lane classes);       optional per image

A missing PNG means "this task is not annotated for this image" (ignored in the loss),
not "all background". DA + LL are packed into a single ``semantic_mask`` (see masks.py)
so upstream augmentation keeps boxes and both masks aligned.

Cost: decoding two 720p PNGs and packing them is ~35 ms, more than the JPEG itself, and a mosaic sample
needs four of them while Ultralytics serves three of its four images from the RAM buffer. The packed mask is
therefore (a) built at the loaded image size (4x fewer pixels to pack at 384x640), (b) cached next to the
buffered images (same eviction) and (c) cached for the whole split with ``cache='ram'`` in the same shared
tensor type Ultralytics uses for the images, so dataloader workers do not duplicate it.
"""

from __future__ import annotations

import os
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import Compose, Format
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

from .masks import IGNORE, pack_masks
from .transforms import build_train_transforms, build_val_transforms

MASK_CHECK_SAMPLE = 32  # masks per task and split whose class ids are validated when the dataset is built


class MultiTaskFormat(Format):
    """Upstream detection ``Format`` that also tensorises the packed semantic mask."""

    def apply_image(self, labels, params=None):
        labels = super().apply_image(labels, params)
        m = labels.get("semantic_mask")
        if m is not None:
            labels["semantic_mask"] = torch.from_numpy(np.ascontiguousarray(m)).to(torch.int32)
        return labels


def _read_png(path: str) -> np.ndarray | None:
    """Class-id mask as uint8, or None when the file does not exist (= task unannotated for this image).

    The PNG header decides how the file is decoded: ``cv2.IMREAD_GRAYSCALE`` would silently turn a palette PNG into
    luminance values, a 1-bit PNG into 0/255 (255 = ignore) and a 16-bit PNG into zeros, and the packing would then
    train those pixels as "unlabelled" or as the wrong class without any error."""
    if not os.path.isfile(path):
        return None  # genuinely unannotated
    try:
        with Image.open(path) as im:
            mode = im.mode  # header only; pixels are not decoded unless asked
            if mode in {"P", "1"}:  # palette indices / 1-bit: only PIL returns the class ids
                return np.asarray(im, dtype=np.uint8)
    except (OSError, ValueError) as e:  # present but unreadable: fail loudly instead of training it as "unlabelled"
        raise OSError(f"corrupt mask file: {path}") from e
    if mode == "L":
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # fast path for the converter's own output
    elif mode.startswith("I"):  # 16-bit / 32-bit integer grayscale
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is not None:
            if m.ndim != 2:
                m = None
            elif m.max() > 255:
                raise ValueError(f"{path}: {m.dtype} mask with ids up to {int(m.max())}; class ids must be < 255")
            else:
                m = m.astype(np.uint8)
    else:
        raise ValueError(f"{path}: PNG mode {mode!r} is not a class-id mask (expected 8-bit grayscale or palette)")
    if m is None:
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
        self._packed: dict = {}  # packed masks of the images in BaseDataset's mosaic buffer (same eviction)
        self._packed_ram = None  # every packed mask of the split when cache='ram' (shared _ImageCache)
        super().__init__(*args, data=data, imgsz=max(hw), task="detect", **kwargs)  # load_image: long side -> max(hw)
        self.check_masks()

    def check_masks(self, sample: int = MASK_CHECK_SAMPLE) -> None:
        """Count the DA / LL masks of the split and validate the class ids of a sample of them.

        The label cache only verifies ``labels_det``; without this a misnamed mask directory (``labels_DA``), a wrong
        ``path:`` in data.yaml or masks with other class ids (a 0/255 binary mask, ids up to 8 with ``ll_classes: 3``)
        would train that task as "unlabelled" everywhere, in silence. A split that genuinely has no labels for a task
        lists it under ``optional_masks: [da|ll]`` in data.yaml."""
        optional = {str(t).lower() for t in (self.data.get("optional_masks") or [])}
        for task, n in (("da", self.da_classes), ("ll", self.ll_classes)):
            files = img2label_paths(self.im_files, label_dir=f"labels_{task}", suffix=".png")
            present = [f for f in files if os.path.isfile(f)]
            LOGGER.info(f"{self.prefix}{len(present)}/{len(files)} images have {task.upper()} masks (labels_{task})")
            if not present:
                if files and task not in optional:
                    raise FileNotFoundError(
                        f"{self.prefix}none of the {len(files)} images has a labels_{task}/*.png mask (looked for "
                        f"{files[0]}): check the directory name and data.yaml's path; if this split really has no "
                        f"{task.upper()} labels, add `optional_masks: [{task}]` to data.yaml"
                    )
                continue
            for f in present[:: max(1, len(present) // sample)][:sample]:
                vals = np.unique(_read_png(f))
                bad = vals[(vals >= n) & (vals != IGNORE)]
                if bad.size:
                    raise ValueError(
                        f"{f}: class ids {bad.tolist()} but data.yaml says {task}_classes={n} (valid ids 0..{n - 1}, "
                        f"255 = ignore); they would train as 'unlabelled'"
                    )
                if IGNORE in vals and set(vals.tolist()) <= {0, IGNORE}:
                    LOGGER.warning(f"{self.prefix}{f} holds only 0 and 255: 255 is 'ignore', so this binary mask trains "
                                   "as unlabelled; use 1 for the foreground class")

    # ---- caching ---------------------------------------------------------------------
    def check_cache_ram(self, safety_margin: float = 1.0) -> bool:
        # the packed mask (1 byte/px) is cached next to the 3-byte/px image: require a third more
        return super().check_cache_ram(safety_margin + (1.0 + safety_margin) / 3.0)

    def cache_images(self) -> None:
        super().cache_images()
        if self.cache != "ram":
            return  # 'disk' caches the images as .npy only; masks are then cached through the mosaic buffer
        masks = [None] * self.ni
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(lambda i: self._read_packed_mask(i, self.im_hw[i]), range(self.ni))
            for i, m in TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0, desc=f"{self.prefix}Caching masks (RAM)"):
                masks[i] = m
        self._packed_ram = BaseDataset._ImageCache(masks)

    # ---- label files -----------------------------------------------------------------
    def get_label_files(self) -> list[str]:
        self.label_files = img2label_paths(self.im_files, label_dir="labels_det")
        return self.label_files

    # ---- transforms ------------------------------------------------------------------
    def build_transforms(self, hyp=None) -> Compose:
        hyp = hyp if hyp is not None else getattr(self, "hyp", None)  # BaseDataset / YOLODataset never store hyp
        if hyp is None:
            raise ValueError("build_transforms needs hyp (the dataset was built without one)")
        self.hyp = hyp
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
    def _read_packed_mask(self, index: int, hw: tuple[int, int]) -> np.ndarray:
        """Read the DA / LL PNGs of image ``index`` and pack them at size ``hw`` (no caching)."""
        im_file = self.labels[index]["im_file"]  # label list may be shorter than the scanned file list
        da = _read_png(img2label_paths([im_file], label_dir="labels_da", suffix=".png")[0])
        ll = _read_png(img2label_paths([im_file], label_dir="labels_ll", suffix=".png")[0])
        # Resize each class map to the loaded image size BEFORE packing: nearest interpolation picks the same
        # source pixel for both maps, so the result equals packing at native size and resizing afterwards, at a
        # quarter of the pixels for 720p -> 384x640 (also lets the two PNGs have different native sizes).
        return pack_masks(_resize_nearest(da, hw), _resize_nearest(ll, hw), self.da_classes, self.ll_classes, shape=tuple(hw))

    def load_packed_mask(self, index: int, hw: tuple[int, int]) -> np.ndarray:
        """Packed mask at the loaded image size ``hw``, from the RAM cache / mosaic-buffer cache when available."""
        hw = tuple(hw)
        if self._packed_ram is not None:
            return _resize_nearest(self._packed_ram[index], hw)
        packed = self._packed.get(index)
        if packed is None:
            packed = self._read_packed_mask(index, hw)
            if self.augment and self.max_buffer_length > 0:  # same condition under which load_image buffers the image
                self._packed[index] = packed
        return _resize_nearest(packed, hw)

    def get_image_and_label(self, index: int) -> dict:
        label = super().get_image_and_label(index)  # load_image may evict an image from the mosaic buffer
        label["semantic_mask"] = self.load_packed_mask(index, label["img"].shape[:2])
        if len(self._packed) > len(self.buffer):  # keep exactly the masks of the images still in RAM
            keep = set(self.buffer)
            self._packed = {i: m for i, m in self._packed.items() if i in keep}
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return YOLODataset.collate_fn(batch)  # stacks img + semantic_mask, concatenates boxes/cls


def _resize_nearest(m: np.ndarray | None, hw: tuple[int, int]) -> np.ndarray | None:
    if m is None or m.shape[:2] == tuple(hw):
        return m
    return cv2.resize(m, (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_NEAREST)
