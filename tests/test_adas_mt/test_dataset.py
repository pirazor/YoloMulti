"""Dataset / augmentation alignment tests on the colour-coded synthetic set."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adas_mt.data.dataset import MultiTaskDataset
from adas_mt.data.masks import IGNORE, unpack_masks

from .conftest import make_hyp

HW = (384, 640)


def build(root, data, split="train", augment=True, **hyp):
    return MultiTaskDataset(
        img_path=str(root / "images" / split), data=data, imgsz=HW, augment=augment,
        hyp=make_hyp(**hyp), batch_size=4, cache=False, prefix="",
    )


def red_green(sample):
    rgb = sample["img"].permute(1, 2, 0).numpy().astype(int)  # Format flips BGR -> RGB
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 170) & (g < 90) & (b < 90), (g > 170) & (r < 90) & (b < 90)


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


@pytest.mark.parametrize("mosaic", [0.0, 1.0])
@pytest.mark.parametrize("degrees", [0.0, 3.0])
def test_image_boxes_masks_aligned(synth_root, synth_data, mosaic, degrees):
    # with rotation the axis-aligned box of a rotated rectangle legitimately contains some background
    min_fill = 0.9 if degrees == 0 else 0.5
    ds = build(synth_root, synth_data, mosaic=mosaic, degrees=degrees)
    ious_da, ious_ll, n_checked = [], [], 0
    import random

    random.seed(0)
    for rep in range(6):
        for i in range(len(ds)):
            s = ds[i]
            assert s["img"].shape == (3, *HW) and s["semantic_mask"].shape == HW
            red, green = red_green(s)
            da, ll = (t.numpy() for t in unpack_masks(s["semantic_mask"], 3, 3))
            if red.sum() > 500:
                ious_da.append(iou(red, da == 1))
            if green.sum() > 500:
                ious_ll.append(iou(green, ll == 1))
            # every box must be filled with its colour (clipped boxes are still pure)
            for (cx, cy, w, h), c in zip(s["bboxes"].tolist(), s["cls"].view(-1).tolist()):
                x1, x2 = int(round((cx - w / 2) * HW[1])) + 2, int(round((cx + w / 2) * HW[1])) - 2
                y1, y2 = int(round((cy - h / 2) * HW[0])) + 2, int(round((cy + h / 2) * HW[0])) - 2
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                m = (red if int(c) == 0 else green)[y1:y2, x1:x2]
                assert m.mean() > min_fill, f"box does not cover its object (class {c}, fill {m.mean():.2f})"
                n_checked += 1
    assert n_checked > 20
    assert np.mean(ious_da) > 0.9, np.mean(ious_da)
    assert np.mean(ious_ll) > 0.9, np.mean(ious_ll)


def test_mosaic_keeps_native_scale(synth_root, synth_data):
    """Regression for the legacy 2x-downscale bug: object size should stay ~native (long side=640)."""
    ds = build(synth_root, synth_data, mosaic=1.0, degrees=0.0, scale=0.0, translate=0.0, fliplr=0.0)
    widths = []
    for i in range(len(ds)):
        s = ds[i]
        red, _ = red_green(s)
        if red.sum() > 2000:
            xs = np.where(red.any(0))[0]
            widths.append(xs.max() - xs.min() + 1)
    # source rect widths are 150..350 px at 1280 wide -> 75..175 px after the 0.5 long-side resize
    assert widths and 40 < np.median(widths) < 200


def test_val_letterbox_pads_with_ignore(synth_root, synth_data):
    ds = build(synth_root, synth_data, split="val", augment=False)
    s = ds[0]
    assert s["img"].shape == (3, *HW) and s["semantic_mask"].shape == HW
    m = s["semantic_mask"]
    assert (m[:10] == IGNORE).all() and (m[-10:] == IGNORE).all()  # 24 pad rows split top/bottom
    assert (m[20:-20] != IGNORE).any()
    red, green = red_green(s)
    da, ll = unpack_masks(m, 3, 3)
    assert iou(red, da == 1) > 0.9 and iou(green, ll == 1) > 0.9


def test_missing_mask_is_unlabelled_not_background(tmp_path, synth_data):
    from .conftest import make_split

    make_split(tmp_path, "train", 2, with_da=False, with_ll=True)
    data = dict(synth_data, path=str(tmp_path))
    ds = build(tmp_path, data, augment=False)
    da, ll = unpack_masks(ds[0]["semantic_mask"], 3, 3)
    assert (da == IGNORE).all()  # task unannotated -> ignored everywhere
    assert (ll == 1).any() and (ll == 0).any()


def test_collate_and_workers(synth_root, synth_data):
    ds = build(synth_root, synth_data, mosaic=1.0)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2, shuffle=True, collate_fn=ds.collate_fn)
    b = next(iter(dl))
    assert b["img"].shape == (4, 3, *HW) and b["img"].dtype == torch.uint8
    assert b["semantic_mask"].shape == (4, *HW) and b["semantic_mask"].dtype == torch.int32
    assert b["bboxes"].shape[1] == 4 and b["batch_idx"].shape[0] == b["bboxes"].shape[0] == b["cls"].shape[0]
    assert b["bboxes"].min() >= 0 and b["bboxes"].max() <= 1
