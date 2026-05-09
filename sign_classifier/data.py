"""ImageFolder-style dataset with Albumentations augmentations for sign crops."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """A simpler, Albumentations-friendly ImageFolder.

    Layout::
        root/
            class_a/
                xxx.jpg
            class_b/
                yyy.jpg
            unknown/
                zzz.jpg

    The ``classes`` list is sorted alphabetically; ``class_to_idx`` is the map.
    """

    def __init__(self, root: Path | str, transform: Callable | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.classes: List[str] = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: List[Tuple[Path, int]] = []
        for c in classes:
            for p in (self.root / c).iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((p, self.class_to_idx[c]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, int(label)


def build_train_transform(imgsz: int = 96):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.MotionBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_val_transform(imgsz: int = 96):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
