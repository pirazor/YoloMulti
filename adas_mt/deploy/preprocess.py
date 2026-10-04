"""Deployment preprocessing == validation preprocessing, pixel for pixel.

Training/validation geometry (Ultralytics 8.4): ``BaseDataset.load_image`` resizes the long side to ``max(hw)`` with
``INTER_LINEAR`` and ``LetterBox(new_shape=hw, scaleup=False, center=True)`` pads with 114 (masks: 255); the tensor
is RGB, /255. ``letterbox_bgr`` reproduces exactly that with cv2 only (no ultralytics import), and tests compare it
with the real dataset pipeline for several frame sizes. ``letterbox_torch`` is the GPU variant for the Jetson
(bilinear, no antialiasing == cv2 INTER_LINEAR); it matches to within one 8-bit level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    """Mapping between the original frame and the network input."""

    orig_hw: Tuple[int, int]
    net_hw: Tuple[int, int]
    unpad_wh: Tuple[int, int]  # size of the resized image inside the padded input (w, h)
    top: int
    left: int

    @property
    def scale_xy(self) -> Tuple[float, float]:
        """x/y scale from original pixels to network pixels (the two resizes are composed)."""
        return self.unpad_wh[0] / self.orig_hw[1], self.unpad_wh[1] / self.orig_hw[0]

    def boxes_to_original(self, xyxy: np.ndarray) -> np.ndarray:
        """(N,4) boxes in network pixels -> original frame pixels (clipped to the frame)."""
        sx, sy = self.scale_xy
        b = np.asarray(xyxy, dtype=np.float32).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.left) / sx
        b[:, [1, 3]] = (b[:, [1, 3]] - self.top) / sy
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, self.orig_hw[1])
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.orig_hw[0])
        return b

    def mask_to_original(self, mask: np.ndarray) -> np.ndarray:
        """(H_net, W_net) class map -> (H0, W0): crop the padding away, then nearest-resize."""
        w, h = self.unpad_wh
        crop = np.ascontiguousarray(mask[self.top : self.top + h, self.left : self.left + w]).astype(np.uint8)
        return cv2.resize(crop, (self.orig_hw[1], self.orig_hw[0]), interpolation=cv2.INTER_NEAREST)


def letterbox_params(orig_hw: Sequence[int], net_hw: Sequence[int]) -> Tuple[Tuple[int, int], Tuple[int, int], int, int, Tuple[int, int]]:
    """Return ``(stage1_wh, unpad_wh, top, left, (bottom, right))`` using Ultralytics' exact arithmetic."""
    h0, w0 = int(orig_hw[0]), int(orig_hw[1])
    s = max(net_hw)
    r = s / max(h0, w0)
    if r != 1:  # BaseDataset.load_image
        w1, h1 = min(math.ceil(w0 * r), s), min(math.ceil(h0 * r), s)
    else:
        w1, h1 = w0, h0
    r2 = min(net_hw[0] / h1, net_hw[1] / w1, 1.0)  # LetterBox(scaleup=False)
    unpad = (round(w1 * r2), round(h1 * r2))
    dw, dh = (net_hw[1] - unpad[0]) / 2, (net_hw[0] - unpad[1]) / 2  # center=True
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return (w1, h1), unpad, top, left, (bottom, right)


def letterbox_bgr(img: np.ndarray, net_hw: Sequence[int] = (384, 640)) -> Tuple[np.ndarray, LetterboxInfo]:
    """BGR uint8 frame -> BGR uint8 letterboxed to ``net_hw`` (same pixels the validator feeds the model)."""
    h0, w0 = img.shape[:2]
    (w1, h1), unpad, top, left, (bottom, right) = letterbox_params((h0, w0), net_hw)
    if (w1, h1) != (w0, h0):
        img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_LINEAR)
    if (w1, h1) != unpad:
        img = cv2.resize(img, unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    assert img.shape[:2] == tuple(net_hw), (img.shape, net_hw)
    return img, LetterboxInfo((h0, w0), tuple(net_hw), unpad, top, left)


def to_tensor_array(letterboxed_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> float32 RGB CHW in [0, 1] (the exported model's input)."""
    return np.ascontiguousarray(letterboxed_bgr[..., ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


def preprocess(img_bgr: np.ndarray, net_hw: Sequence[int] = (384, 640)) -> Tuple[np.ndarray, LetterboxInfo]:
    lb, info = letterbox_bgr(img_bgr, net_hw)
    return to_tensor_array(lb), info


def letterbox_torch(frame_bgr, net_hw: Sequence[int] = (384, 640)):
    """GPU variant: ``frame_bgr`` is a uint8 (H, W, 3) torch tensor (any device). Returns ``(1,3,H,W)`` float RGB in
    [0,1] on the same device and the :class:`LetterboxInfo`. Bilinear without antialiasing == cv2 INTER_LINEAR
    (half-pixel centres), so the result is within one 8-bit level of :func:`preprocess`."""
    import torch
    import torch.nn.functional as F

    h0, w0 = frame_bgr.shape[:2]
    (w1, h1), unpad, top, left, (bottom, right) = letterbox_params((h0, w0), net_hw)
    x = frame_bgr.permute(2, 0, 1)[None].float()
    if (w1, h1) != (w0, h0):
        x = F.interpolate(x, size=(h1, w1), mode="bilinear", align_corners=False)
    if (w1, h1) != unpad:
        x = F.interpolate(x, size=(unpad[1], unpad[0]), mode="bilinear", align_corners=False)
    x = torch.floor(x + 0.5).clamp_(0, 255)  # cv2 returns uint8, rounding half up (torch.round is half-to-even)
    x = F.pad(x, (left, right, top, bottom), value=float(PAD_VALUE))
    x = x.flip(1) / 255.0  # BGR -> RGB
    return x.contiguous(), LetterboxInfo((h0, w0), tuple(net_hw), unpad, top, left)
