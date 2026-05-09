"""Multi-task dataset returning image + det + drivable-area mask + lane mask.

Loads from a YOLO-format directory written by ``convert_supervisely.py``.

Returned per-sample tensor dict::

    img      : float32 (3, H, W) in [0, 1]
    bboxes   : float32 (N, 4) xyxy in pixel coords on (H, W)
    cls      : int64   (N,)
    da_mask  : int64   (H, W)
    ll_mask  : int64   (H, W)
    has_da   : bool tensor scalar
    has_ll   : bool tensor scalar
    im_file  : str (kept for debug; stripped by collate_fn)
    ori_shape: (h0, w0)

``collate_fn`` produces the v8-style det targets (``batch_idx``, ``cls``,
``bboxes``) so the upstream ``v8DetectionLoss`` can be reused unchanged.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .mosaic import mosaic4
from .transforms import apply as apply_transform
from .transforms import build_train_transform, build_val_transform

LOGGER = logging.getLogger("yolov13_multitask.dataset")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _load_image(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path, h: int, w: int) -> Tuple[np.ndarray, bool]:
    if not path.exists():
        return np.zeros((h, w), dtype=np.uint8), False
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((h, w), dtype=np.uint8), False
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m, True


def _load_yolo_txt(path: Path, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (bboxes_xyxy_pixels, cls)."""
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    rows: List[List[float]] = []
    cls: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        cx, cy, bw, bh = (float(p) for p in parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        rows.append([x1, y1, x2, y2])
        cls.append(c)
    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(rows, dtype=np.float32), np.asarray(cls, dtype=np.int64)


class MultiTaskDataset(Dataset):
    """Image + detection + drivable-area + lane segmentation."""

    def __init__(
        self,
        data_yaml: str | Path,
        split: str = "train",
        imgsz: int = 640,
        augment: bool = True,
        hyp: Optional[Dict] = None,
        mosaic_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.data_yaml_path = Path(data_yaml).resolve()
        with self.data_yaml_path.open("r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)
        self.root = Path(self.data.get("path", self.data_yaml_path.parent))
        if not self.root.is_absolute():
            self.root = (self.data_yaml_path.parent / self.root).resolve()

        self.split = split
        self.imgsz = int(imgsz)
        self.augment = bool(augment)
        self.hyp = hyp or {}
        self.mosaic_prob = float(mosaic_prob if augment else 0.0)
        self.nc = int(self.data["nc"])
        self.da_classes = int(self.data.get("da_classes", 3))
        self.ll_classes = int(self.data.get("ll_classes", 3))
        self.names = self.data.get("names", [f"c{i}" for i in range(self.nc)])

        img_dir = self.root / self.data.get(split, f"images/{split}")
        if not img_dir.exists():
            raise FileNotFoundError(f"image directory not found: {img_dir}")
        self.image_paths: List[Path] = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            raise RuntimeError(f"no images found under {img_dir}")
        self.det_dir = self.root / "labels_det" / split
        self.da_dir = self.root / "labels_da" / split
        self.ll_dir = self.root / "labels_ll" / split

        self.transform = build_train_transform(imgsz, self.hyp) if augment else build_val_transform(imgsz)
        self._rng = random.Random(self.hyp.get("seed", 0))

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.image_paths)

    def _raw(self, idx: int) -> Dict:
        ip = self.image_paths[idx]
        img = _load_image(ip)
        h, w = img.shape[:2]
        bboxes, cls = _load_yolo_txt(self.det_dir / f"{ip.stem}.txt", w, h)
        da, has_da = _load_mask(self.da_dir / f"{ip.stem}.png", h, w)
        ll, has_ll = _load_mask(self.ll_dir / f"{ip.stem}.png", h, w)
        return {
            "img": img,
            "bboxes": bboxes,
            "cls": cls,
            "da_mask": da,
            "ll_mask": ll,
            "has_da": has_da,
            "has_ll": has_ll,
            "im_file": str(ip),
            "ori_shape": (h, w),
        }

    # ------------------------------------------------------------------ getitem
    def __getitem__(self, idx: int) -> Dict:
        do_mosaic = self.augment and self.mosaic_prob > 0 and self._rng.random() < self.mosaic_prob
        if do_mosaic:
            indices = [idx] + [self._rng.randint(0, len(self) - 1) for _ in range(3)]
            samples = [self._raw(i) for i in indices]
            sample = mosaic4(samples, self.imgsz, self._rng)
            sample["im_file"] = samples[0]["im_file"]
            sample["ori_shape"] = samples[0]["ori_shape"]
        else:
            sample = self._raw(idx)

        out = apply_transform(self.transform, sample)

        # albumentations ToTensorV2 returns image as (C, H, W) float
        img = out["image"].float() / 255.0
        da_mask = out["da_mask"].long()
        ll_mask = out["ll_mask"].long()
        bboxes = torch.as_tensor(out["bboxes"], dtype=torch.float32).reshape(-1, 4)
        cls = torch.as_tensor(out["cls"], dtype=torch.int64).reshape(-1, 1)

        return {
            "img": img,
            "bboxes": bboxes,
            "cls": cls,
            "da_mask": da_mask,
            "ll_mask": ll_mask,
            "has_da": torch.tensor(bool(sample.get("has_da", False))),
            "has_ll": torch.tensor(bool(sample.get("has_ll", False))),
            "im_file": sample["im_file"],
            "ori_shape": sample["ori_shape"],
        }

    # ----------------------------------------------------------------- collate
    @staticmethod
    def collate_fn(items: List[Dict]) -> Dict:
        imgs = torch.stack([x["img"] for x in items], dim=0)
        da = torch.stack([x["da_mask"] for x in items], dim=0)
        ll = torch.stack([x["ll_mask"] for x in items], dim=0)
        has_da = torch.stack([x["has_da"] for x in items], dim=0)
        has_ll = torch.stack([x["has_ll"] for x in items], dim=0)

        # Build v8-style flat targets: bboxes as (M, 4) in normalized xywh + a
        # parallel batch_idx + cls. v8DetectionLoss internally converts the
        # xywh*scale to xyxy via preprocess() (see utils/loss.py:194).
        bboxes_xywh: List[torch.Tensor] = []
        cls_flat: List[torch.Tensor] = []
        batch_idx: List[torch.Tensor] = []
        H, W = imgs.shape[-2:]
        for i, x in enumerate(items):
            b = x["bboxes"]
            if b.numel() == 0:
                continue
            xywh = torch.zeros_like(b)
            xywh[:, 0] = (b[:, 0] + b[:, 2]) / 2 / W
            xywh[:, 1] = (b[:, 1] + b[:, 3]) / 2 / H
            xywh[:, 2] = (b[:, 2] - b[:, 0]) / W
            xywh[:, 3] = (b[:, 3] - b[:, 1]) / H
            bboxes_xywh.append(xywh)
            cls_flat.append(x["cls"].view(-1, 1).float())
            batch_idx.append(torch.full((b.shape[0], 1), i, dtype=torch.float32))

        if bboxes_xywh:
            bboxes = torch.cat(bboxes_xywh, 0)
            cls_t = torch.cat(cls_flat, 0)
            bidx = torch.cat(batch_idx, 0)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            cls_t = torch.zeros((0, 1), dtype=torch.float32)
            bidx = torch.zeros((0, 1), dtype=torch.float32)

        return {
            "img": imgs,
            "bboxes": bboxes,
            "cls": cls_t,
            "batch_idx": bidx.view(-1),
            "da_mask": da,
            "ll_mask": ll,
            "has_da": has_da,
            "has_ll": has_ll,
            "ori_shape": [x["ori_shape"] for x in items],
            "im_file": [x["im_file"] for x in items],
        }
