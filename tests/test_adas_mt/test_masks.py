import numpy as np
import pytest
import torch

from adas_mt.data.masks import IGNORE, pack_masks, unpack_masks


def test_roundtrip_and_unlabelled():
    da = np.array([[0, 1, 2, 2]], np.uint8)
    ll = np.array([[0, 0, 1, 2]], np.uint8)
    p = pack_masks(da, ll, 3, 3)
    d, l = unpack_masks(torch.from_numpy(p), 3, 3)
    assert d.tolist() == [[0, 1, 2, 2]] and l.tolist() == [[0, 0, 1, 2]]

    p = pack_masks(None, ll, 3, 3)  # DA unannotated -> ignored, LL intact
    d, l = unpack_masks(torch.from_numpy(p), 3, 3)
    assert (d == IGNORE).all() and l.tolist() == [[0, 0, 1, 2]]

    p = pack_masks(da, None, 3, 3)
    d, l = unpack_masks(torch.from_numpy(p), 3, 3)
    assert d.tolist() == [[0, 1, 2, 2]] and (l == IGNORE).all()


def test_padding_and_out_of_range():
    p = np.array([[IGNORE, pack_masks(np.array([[1]], np.uint8), np.array([[1]], np.uint8), 3, 3)[0, 0]]])
    d, l = unpack_masks(torch.from_numpy(p), 3, 3)
    assert d.tolist() == [[IGNORE, 1]] and l.tolist() == [[IGNORE, 1]]
    # class ids out of range (e.g. 255 in a source PNG) become unlabelled pixels for that task
    p = pack_masks(np.array([[255]], np.uint8), np.array([[1]], np.uint8), 3, 3)
    d, l = unpack_masks(torch.from_numpy(p), 3, 3)
    assert d.item() == IGNORE and l.item() == 1


def test_capacity_guard():
    with pytest.raises(ValueError):
        pack_masks(np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.uint8), 3, 100)


def test_nearest_warp_preserves_codes():
    import cv2

    rng = np.random.default_rng(0)
    da = rng.integers(0, 3, (50, 80)).astype(np.uint8)
    ll = rng.integers(0, 3, (50, 80)).astype(np.uint8)
    p = pack_masks(da, ll, 3, 3)
    q = cv2.resize(p, (33, 21), interpolation=cv2.INTER_NEAREST)
    assert set(np.unique(q)) <= set(np.unique(p))
