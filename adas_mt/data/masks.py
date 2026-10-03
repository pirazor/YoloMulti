"""Packing of the drivable-area (DA) and lane (LL) label maps into ONE uint8 mask.

Ultralytics' augmentation stack (Mosaic / RandomPerspective / RandomFlip / LetterBox)
already warps a single ``semantic_mask`` together with the image and boxes, using
nearest-neighbour interpolation and an ignore value of 255 for padded pixels. We
reuse that machinery by packing both task maps into one value per pixel::

    packed = da_code + K * ll_code          K = da_classes + 1
    da_code in [0, da_classes]              da_classes  == "this task is unlabelled"
    ll_code in [0, ll_classes]              ll_classes  == "this task is unlabelled"
    packed == 255                           padding / outside the image: ignore everything

Nearest-neighbour warps never create new values, so the encoding is preserved exactly.
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE = 255


def code_base(da_classes: int) -> int:
    return int(da_classes) + 1


def check_capacity(da_classes: int, ll_classes: int) -> None:
    top = da_classes + code_base(da_classes) * ll_classes
    if top >= IGNORE:
        raise ValueError(f"da_classes={da_classes}, ll_classes={ll_classes} do not fit in a uint8 packed mask")


def pack_masks(
    da: np.ndarray | None,
    ll: np.ndarray | None,
    da_classes: int,
    ll_classes: int,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Pack DA / LL class maps (uint8, 0=background) into one uint8 mask.

    A ``None`` map means the task is unlabelled for this image. Class ids outside the
    valid range (including 255) are treated as unlabelled pixels for that task.
    """
    check_capacity(da_classes, ll_classes)
    ref = da if da is not None else ll
    if ref is None:
        if shape is None:
            raise ValueError("shape is required when both maps are None")
        h, w = shape
    else:
        h, w = ref.shape[:2]

    def codes(m: np.ndarray | None, n: int) -> np.ndarray:
        if m is None:
            return np.full((h, w), n, dtype=np.uint8)
        if m.shape[:2] != (h, w):
            raise ValueError(f"mask shape {m.shape[:2]} != {(h, w)}")
        return np.where(m < n, m, n).astype(np.uint8)  # out-of-range -> unlabelled code

    k = code_base(da_classes)
    return (codes(da, da_classes) + k * codes(ll, ll_classes)).astype(np.uint8)


def unpack_masks(packed: torch.Tensor, da_classes: int, ll_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed (..., H, W) integer mask -> (da, ll) int64 maps where unlabelled/padding = 255."""
    packed = packed.long()
    k = code_base(da_classes)
    pad = packed == IGNORE
    da = packed % k
    ll = packed // k
    da = torch.where(pad | (da == da_classes), torch.full_like(da, IGNORE), da)
    ll = torch.where(pad | (ll == ll_classes), torch.full_like(ll, IGNORE), ll)
    return da, ll
