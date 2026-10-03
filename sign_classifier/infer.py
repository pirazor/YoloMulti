"""Batched inference helper: feed detection boxes -> fine-grained sign labels."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from sign_classifier.data import build_val_transform
from sign_classifier.models import build_model


def _square_pad_box(x1: int, y1: int, x2: int, y2: int, pad: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh) * (1.0 + pad)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = side / 2
    return (
        int(max(0, round(cx - half))),
        int(max(0, round(cy - half))),
        int(min(w, round(cx + half))),
        int(min(h, round(cy + half))),
    )


class SignClassifier:
    def __init__(
        self,
        weights: str | Path,
        device: Optional[str] = None,
        imgsz: int = 96,
        padding: float = 0.15,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        self.classes: List[str] = list(ckpt["classes"])
        self.model = build_model(ckpt["model_name"], num_classes=len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device).eval()
        self.imgsz = int(imgsz)
        self.padding = float(padding)
        self.transform = build_val_transform(self.imgsz)

    @torch.no_grad()
    def classify(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: Sequence[Tuple[float, float, float, float]],
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        if not len(boxes_xyxy):
            return []
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crops: List[torch.Tensor] = []
        for x1, y1, x2, y2 in boxes_xyxy:
            ix1, iy1, ix2, iy2 = _square_pad_box(int(x1), int(y1), int(x2), int(y2), self.padding, w, h)
            crop = rgb[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                crop = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            crops.append(self.transform(image=crop)["image"])
        out: List[Tuple[str, float]] = []
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i : i + batch_size]).to(self.device)
            logits = self.model(batch)
            probs = logits.softmax(dim=1)
            conf, idx = probs.max(dim=1)
            for c, k in zip(conf.cpu().tolist(), idx.cpu().tolist()):
                out.append((self.classes[k], float(c)))
        return out
