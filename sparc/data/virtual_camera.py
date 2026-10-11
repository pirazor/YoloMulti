"""The canonical virtual camera and the affine warps that put every dataset into it.

Every image is warped to one pinhole camera with a fixed horizontal field of view at the network resolution
(Omni3D's idea: one focal length for all datasets, so metric depth and object size mean the same thing everywhere,
and scale augmentation is a change of intrinsics the network is told about through its ``K`` input). The camera
orientation is unchanged, so the warp from a source camera ``K_s`` to the virtual ``K_v`` is ``K_v K_s^-1``: an
affine map (both matrices are upper triangular), applied with one ``cv2.warpAffine`` together with the training
augmentation (scale, translation, horizontal flip) composed into the same 3x3 matrix.

Sources without calibration get a nominal ``K`` (:func:`nominal_K`): a field of view guess and the horizon row as
the principal point; the sample's ``calib="nominal"`` tells the losses to trust its metric lift less.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_HW = (384, 640)
DEFAULT_HFOV_DEG = 60.0
PAD_VALUE = 114
MASK_IGNORE = 255


@dataclass(frozen=True)
class VirtualCamera:
    hw: Tuple[int, int] = DEFAULT_HW
    hfov_deg: float = DEFAULT_HFOV_DEG

    @property
    def fx(self) -> float:
        return (self.hw[1] / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def K(self) -> np.ndarray:
        h, w = self.hw
        return np.array([[self.fx, 0.0, (w - 1) / 2.0], [0.0, self.fx, (h - 1) / 2.0], [0.0, 0.0, 1.0]])

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.hw[0] / 2.0) / self.fx))


def nominal_K(hw: Sequence[int], hfov_deg: float = 65.0, horizon_row: Optional[float] = None,
              fx: Optional[float] = None) -> np.ndarray:
    """A plausible pinhole matrix for an uncalibrated dashcam: ``fx`` from the FOV guess, principal point at the
    image centre horizontally and at the horizon row vertically (so a nominal level mount is consistent)."""
    h, w = int(hw[0]), int(hw[1])
    f = float(fx) if fx is not None else (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cy = (h - 1) / 2.0 if horizon_row is None else float(horizon_row)
    return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, cy], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- 3x3 affine algebra
def intrinsics_affine(K_src: np.ndarray, K_dst: np.ndarray) -> np.ndarray:
    """3x3 map from source pixels to destination pixels for the same camera orientation."""
    A = np.asarray(K_dst, dtype=np.float64) @ np.linalg.inv(np.asarray(K_src, dtype=np.float64))
    if not np.allclose(A[2], (0, 0, 1), atol=1e-9):
        raise ValueError("intrinsics must be upper-triangular pinhole matrices")
    A[2] = (0, 0, 1)
    return A


def scale_translate(s: float, tx: float = 0.0, ty: float = 0.0, about: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Scale by ``s`` about ``about`` then translate by (tx, ty)."""
    ax, ay = about
    return np.array([[s, 0.0, ax - s * ax + tx], [0.0, s, ay - s * ay + ty], [0.0, 0.0, 1.0]])


def hflip(w: int) -> np.ndarray:
    return np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def compose(*mats: np.ndarray) -> np.ndarray:
    """``compose(A3, A2, A1)`` applies A1 first."""
    out = np.eye(3)
    for m in mats:
        out = out @ np.asarray(m, dtype=np.float64)
    return out


def apply_to_K(A: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Intrinsics of the warped image (the warp is a change of intrinsics, the orientation is unchanged)."""
    return np.asarray(A, dtype=np.float64) @ np.asarray(K, dtype=np.float64)


# --------------------------------------------------------------------------- warps
def warp_image(img: np.ndarray, A: np.ndarray, hw: Sequence[int], border: int = PAD_VALUE,
               interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    h, w = int(hw[0]), int(hw[1])
    val = (border,) * 3 if img.ndim == 3 else border
    return cv2.warpAffine(img, np.asarray(A, dtype=np.float64)[:2], (w, h), flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=val)


def warp_mask(mask: np.ndarray, A: np.ndarray, hw: Sequence[int], border: int = MASK_IGNORE) -> np.ndarray:
    return warp_image(mask, A, hw, border=border, interpolation=cv2.INTER_NEAREST)


def warp_points(pts: np.ndarray, A: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    A = np.asarray(A, dtype=np.float64)
    return (p @ A[:2, :2].T + A[:2, 2]).astype(np.float32)


def warp_boxes(xyxy: np.ndarray, A: np.ndarray, hw: Sequence[int], min_visible: float = 0.2,
               min_size: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned boxes after the affine map, clipped to the image.

    Returns ``(boxes, keep)``: ``keep`` is False for boxes that lost more than ``1 - min_visible`` of their area to
    the border or ended up smaller than ``min_size`` pixels (callers drop their labels)."""
    b = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    if len(b) == 0:
        return b.astype(np.float32), np.zeros((0,), dtype=bool)
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [0, 3]], b[:, [2, 3]]], 1)  # (N, 4, 2)
    wc = warp_points(corners.reshape(-1, 2), A).reshape(-1, 4, 2)
    out = np.concatenate([wc.min(1), wc.max(1)], 1)
    area0 = (out[:, 2] - out[:, 0]) * (out[:, 3] - out[:, 1])
    h, w = hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w - 1)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h - 1)
    bw, bh = out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]
    keep = (bw >= min_size) & (bh >= min_size) & (bw * bh >= min_visible * np.maximum(area0, 1e-9))
    return out.astype(np.float32), keep
