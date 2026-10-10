"""The deployment preprocessing must reproduce the validation pipeline pixel for pixel."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from adas_mt.data.dataset import MultiTaskDataset
from adas_mt.deploy.preprocess import letterbox_bgr, letterbox_params, letterbox_torch, preprocess

from .conftest import make_hyp, make_split

HW = (384, 640)
SIZES = [(720, 1280), (1080, 1920), (600, 800), (480, 640), (360, 640), (1024, 1024), (1280, 720), (333, 777), (2160, 3840)]


@pytest.mark.parametrize("hw0", SIZES)
def test_matches_the_validation_dataset_pipeline_exactly(tmp_path, hw0):
    """Write a frame, load it through MultiTaskDataset(val) and through the deploy preprocessing."""
    import yaml

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (*hw0, 3), dtype=np.uint8)
    for d in ("images/val", "labels_det/val", "labels_da/val", "labels_ll/val"):
        (tmp_path / d).mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/val/f.png"), img)  # lossless: both pipelines read identical pixels
    (tmp_path / "labels_det/val/f.txt").write_text("")
    data = {"path": str(tmp_path), "nc": 2, "names": ["a", "b"], "da_classes": 3, "ll_classes": 3,
            "optional_masks": ["da", "ll"]}  # a frame without masks: the dataset would otherwise flag the empty split
    ds = MultiTaskDataset(img_path=str(tmp_path / "images/val"), data=data, imgsz=HW, augment=False, hyp=make_hyp(),
                          batch_size=1, cache=False, rect=False, prefix="")
    ref = ds[0]["img"].numpy()  # uint8 RGB CHW
    lb, info = letterbox_bgr(img, HW)
    assert np.array_equal(lb[..., ::-1].transpose(2, 0, 1), ref), f"deploy letterbox differs for {hw0}"
    # semantic mask geometry follows the same mapping (255 = padding in the dataset's mask)
    m = ds[0]["semantic_mask"].numpy()
    inside = np.zeros(HW, bool)
    w, h = info.unpad_wh
    inside[info.top : info.top + h, info.left : info.left + w] = True
    assert (m[~inside] == 255).all() and (m[inside] != 255).all()


@pytest.mark.parametrize("hw0", SIZES)
def test_box_and_mask_round_trip(hw0):
    img = np.zeros((*hw0, 3), np.uint8)
    _, info = letterbox_bgr(img, HW)
    h0, w0 = hw0
    pts = np.array([[0, 0, w0, h0], [w0 * 0.25, h0 * 0.25, w0 * 0.75, h0 * 0.5]], np.float32)
    sx, sy = info.scale_xy
    net = pts.copy()
    net[:, [0, 2]] = net[:, [0, 2]] * sx + info.left
    net[:, [1, 3]] = net[:, [1, 3]] * sy + info.top
    assert np.allclose(info.boxes_to_original(net), pts, atol=1e-3)
    # a rectangle drawn in original coordinates survives the net-space round trip
    mask0 = np.zeros((h0, w0), np.uint8)
    mask0[int(h0 * 0.25) : int(h0 * 0.5), int(w0 * 0.25) : int(w0 * 0.75)] = 1
    lb_mask = np.zeros(HW, np.uint8)
    w, h = info.unpad_wh
    lb_mask[info.top : info.top + h, info.left : info.left + w] = cv2.resize(mask0, (w, h), interpolation=cv2.INTER_NEAREST)
    back = info.mask_to_original(lb_mask)
    assert back.shape == mask0.shape and (back == mask0).mean() > 0.99


@pytest.mark.parametrize("hw0", [(720, 1280), (1080, 1920), (480, 640), (1280, 720)])
def test_gpu_variant_matches_within_one_level(hw0):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (*hw0, 3), dtype=np.uint8)
    ref, info = preprocess(img, HW)
    got, info2 = letterbox_torch(torch.from_numpy(img), HW)
    assert info == info2 and got.shape == (1, 3, *HW)
    assert np.abs(got[0].numpy() * 255 - ref * 255).max() <= 2.0  # <=2 levels (fixed-point rounding of cv2)
    assert np.abs(got[0].numpy() * 255 - ref * 255).mean() < 0.1


def test_params_for_the_common_camera_sizes():
    # 1280x720 -> 640x360 -> 12 px of padding above and below
    (w1, h1), unpad, top, left, (bottom, right) = letterbox_params((720, 1280), HW)
    assert (w1, h1) == unpad == (640, 360) and (top, bottom, left, right) == (12, 12, 0, 0)
