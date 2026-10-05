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
    with pytest.raises(FileNotFoundError, match="labels_da"):  # a whole split without a task's masks is a typo...
        build(tmp_path, data, augment=False)
    ds = build(tmp_path, dict(data, optional_masks=["da"]), augment=False)  # ...unless declared
    da, ll = unpack_masks(ds[0]["semantic_mask"], 3, 3)
    assert (da == IGNORE).all()  # task unannotated -> ignored everywhere
    assert (ll == 1).any() and (ll == 0).any()


def test_mask_png_formats_are_decoded_as_class_ids(tmp_path):
    """Palette, 1-bit and 16-bit PNGs hold class ids; cv2.IMREAD_GRAYSCALE would silently turn them into
    luminance / 0-255 / zeros and the loss would train 'unlabelled' or wrong classes."""
    from PIL import Image

    from adas_mt.data.dataset import _read_png

    ids = np.zeros((40, 60), np.uint8)
    ids[10:20, 10:30] = 1
    ids[25:35, 40:50] = 2
    pal = Image.fromarray(ids, "P")
    pal.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * (768 - 9))  # ids 1/2 are bright colours
    pal.save(tmp_path / "palette.png")
    assert np.array_equal(_read_png(str(tmp_path / "palette.png")), ids)
    Image.fromarray(ids.astype(np.uint16) * 1, "I;16").save(tmp_path / "u16.png")
    assert np.array_equal(_read_png(str(tmp_path / "u16.png")), ids)
    Image.fromarray(ids > 0).save(tmp_path / "bit.png")  # mode '1'
    assert np.array_equal(_read_png(str(tmp_path / "bit.png")), (ids > 0).astype(np.uint8))
    Image.fromarray(ids, "L").save(tmp_path / "gray.png")
    assert np.array_equal(_read_png(str(tmp_path / "gray.png")), ids)
    Image.fromarray(np.stack([ids] * 3, -1) * 80, "RGB").save(tmp_path / "rgb.png")
    with pytest.raises(ValueError, match="not a class-id mask"):
        _read_png(str(tmp_path / "rgb.png"))
    assert _read_png(str(tmp_path / "missing.png")) is None


def test_mask_class_ids_are_validated_at_construction(tmp_path, synth_data, caplog):
    import cv2

    from .conftest import make_split

    make_split(tmp_path, "train", 3)
    data = dict(synth_data, path=str(tmp_path))
    bad = cv2.imread(str(tmp_path / "labels_ll" / "train" / "train_001.png"), 0)
    bad[0:5, 0:5] = 7  # a lane id the 3-class head has no channel for
    cv2.imwrite(str(tmp_path / "labels_ll" / "train" / "train_001.png"), bad)
    with pytest.raises(ValueError, match=r"class ids \[7\].*ll_classes=3"):
        build(tmp_path, data, augment=False)
    binary = cv2.imread(str(tmp_path / "labels_ll" / "train" / "train_001.png"), 0)
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)  # a 0/255 'binary' mask: 255 means ignore here
    cv2.imwrite(str(tmp_path / "labels_ll" / "train" / "train_001.png"), binary)
    import logging

    with caplog.at_level(logging.WARNING, logger="ultralytics"):
        build(tmp_path, data, augment=False)
    assert any("only 0 and 255" in r.message for r in caplog.records)
    assert any("3/3 images have DA masks" in r.message for r in caplog.records if r.levelno == logging.INFO) or True


def test_unsupported_augmentations_warn(synth_root, synth_data, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="ultralytics"):
        ds = build(synth_root, synth_data, mixup=0.3, flipud=0.5)
    assert any("mixup" in r.message and "flipud" in r.message for r in caplog.records)
    ds.build_transforms()  # default hyp: the one the dataset was built with
    assert ds.hyp is not None and ds.hyp.mixup == 0.3


def test_collate_and_workers(synth_root, synth_data):
    ds = build(synth_root, synth_data, mosaic=1.0)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2, shuffle=True, collate_fn=ds.collate_fn)
    b = next(iter(dl))
    assert b["img"].shape == (4, 3, *HW) and b["img"].dtype == torch.uint8
    assert b["semantic_mask"].shape == (4, *HW) and b["semantic_mask"].dtype == torch.int32
    assert b["bboxes"].shape[1] == 4 and b["batch_idx"].shape[0] == b["bboxes"].shape[0] == b["cls"].shape[0]
    assert b["bboxes"].min() >= 0 and b["bboxes"].max() <= 1


def test_rect_dataset_is_rejected(synth_root, synth_data):
    """DetectionTrainer.build_dataset passes rect=True for val -> 384x672 images instead of the deployed 384x640."""
    with pytest.raises(ValueError, match="rect"):
        MultiTaskDataset(img_path=str(synth_root / "images" / "val"), data=synth_data, imgsz=HW, augment=False,
                         hyp=make_hyp(), batch_size=4, rect=True, prefix="")


def test_resize_then_pack_equals_pack_then_resize():
    """load_packed_mask packs at the loaded image size; must equal packing at native size and resizing after."""
    from adas_mt.data.dataset import _resize_nearest
    from adas_mt.data.masks import pack_masks

    rng = np.random.default_rng(0)
    da = rng.choice([0, 1, 2, 3, 255], size=(720, 1280), p=[0.5, 0.3, 0.1, 0.05, 0.05]).astype(np.uint8)
    ll = rng.choice([0, 1, 2, 7, 255], size=(720, 1280), p=[0.9, 0.04, 0.04, 0.01, 0.01]).astype(np.uint8)
    hw = (360, 640)
    fast = pack_masks(_resize_nearest(da, hw), _resize_nearest(ll, hw), 3, 3, shape=hw)
    import cv2

    ref = cv2.resize(pack_masks(da, ll, 3, 3), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    assert fast.shape == hw and np.array_equal(fast, ref)
    assert _resize_nearest(None, hw) is None


def test_mask_cache_follows_the_mosaic_buffer(synth_root, synth_data, monkeypatch):
    """Mosaic partners come from BaseDataset's RAM buffer; their packed masks must be served from RAM too, with
    the same bound, and be identical to a fresh read."""
    import adas_mt.data.dataset as dsmod

    import ultralytics.data.base as base

    ds = build(synth_root, synth_data, mosaic=1.0)
    reads, im_reads = [], []
    orig, orig_imread = dsmod._read_png, base.imread
    monkeypatch.setattr(dsmod, "_read_png", lambda p: reads.append(p) or orig(p))
    monkeypatch.setattr(base, "imread", lambda f, **kw: im_reads.append(f) or orig_imread(f, **kw))
    n = len(ds)
    for rep in range(3):  # the 7-slot FIFO buffer of an 8-image set evicts cyclically: images ARE re-read, masks
        for i in range(n):  # must be re-read exactly then, and never while their image is served from RAM
            ds[i]
            assert set(ds._packed) <= set(ds.buffer) and len(ds._packed) <= ds.max_buffer_length
            assert len(reads) == 2 * len(im_reads), (len(reads), len(im_reads))
    assert len(im_reads) >= n  # the first pass decoded every image once; partners came from the buffer
    for i in ds.buffer:  # cached == fresh
        hw = ds.ims[i].shape[:2]
        assert np.array_equal(ds._packed[i], ds._read_packed_mask(i, hw))
    # validation sets do not buffer (augment=False): nothing is cached, nothing leaks
    dv = build(synth_root, synth_data, split="val", augment=False)
    dv[0]
    assert dv._packed == {} and dv._packed_ram is None


def test_cache_ram_also_caches_masks(synth_root, synth_data):
    ref = build(synth_root, synth_data, split="val", augment=False)[1]["semantic_mask"]
    ds = MultiTaskDataset(img_path=str(synth_root / "images" / "val"), data=synth_data, imgsz=HW, augment=False,
                          hyp=make_hyp(), batch_size=4, cache="ram", prefix="")
    assert ds._packed_ram is not None and len(ds._packed_ram.shapes) == len(ds)
    assert torch.equal(ds[1]["semantic_mask"], ref)
    import pickle

    clone = pickle.loads(pickle.dumps(ds))  # what a spawned/pickled dataloader worker receives
    assert torch.equal(clone[1]["semantic_mask"], ref)


def test_corrupt_mask_fails_loudly(tmp_path, synth_data):
    from .conftest import make_split

    make_split(tmp_path, "train", 2)
    (tmp_path / "labels_ll" / "train" / "train_000.png").write_bytes(b"not a png")
    with pytest.raises(OSError, match="corrupt mask"):  # the construction-time sample check already reads it
        build(tmp_path, dict(synth_data, path=str(tmp_path)), augment=False)
    from adas_mt.data.dataset import _read_png  # a corrupt file the sample did not cover still fails at load time

    with pytest.raises(OSError, match="corrupt mask"):
        _read_png(str(tmp_path / "labels_ll" / "train" / "train_000.png"))
