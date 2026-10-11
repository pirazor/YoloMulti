"""Mini CULane dump: list files with existence flags, per-image .lines.txt, optional laneseg_label_w16 PNGs."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.converters.culane import orders_from_x
from sparc.data.schema import LANE_ORDER_UNLABELLED, Sample

H, W = 590, 1640
CLIP = "driver_23_30frame/05151649_0422.MP4"
STEM = CLIP.replace("/", "__").replace(".", "_")
ROWS = list(range(589, 279, -10))  # 31 rows, bottom-up as in the real files


def _lane(x_bottom, x_top, rows=ROWS, top_down=False):
    xs = np.linspace(x_bottom, x_top, len(rows))
    pairs = [f"{x:.4f} {y}" for x, y in zip(xs, rows)]
    return " ".join(pairs[::-1] if top_down else pairs)


def _write_frame(root: Path, rel: str, lines, with_image=True):
    img = root / rel
    img.parent.mkdir(parents=True, exist_ok=True)
    if with_image:
        cv2.imwrite(str(img), np.full((H, W, 3), 90, np.uint8))
    if lines is not None:
        img.with_suffix(".lines.txt").write_text("\n".join(lines) + "\n")


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "culane"
    (root / "list").mkdir(parents=True)
    # train 00000: all four slots; the left-left lane starts outside the image (negative x -> dropped point);
    # the right lane is written top-down to check the sort
    f0 = f"/{CLIP}/00000.jpg"
    _write_frame(root, f0.lstrip("/"), [
        _lane(-20, 700),                                    # left-left
        _lane(600, 600),                                    # left: vertical at x = 600
        _lane(1000, 860, top_down=True),                    # right, written top-down in the file
        _lane(1400, 950),                                   # right-right
    ])
    # train 00030: flags 0 1 1 0, the right lane has a single point (skipped) but still occupies its slot
    f1 = f"/{CLIP}/00030.jpg"
    _write_frame(root, f1.lstrip("/"), [_lane(500, 700), "1000.0 589"])
    f2 = f"/{CLIP}/00060.jpg"  # listed, image missing
    (root / "list" / "train.txt").write_text("\n".join([f0, f1, f2]) + "\n")
    (root / "list" / "train_gt.txt").write_text(
        f"{f0} /laneseg_label_w16/{CLIP}/00000.png 1 1 1 1\n{f1} /laneseg_label_w16/{CLIP}/00030.png 0 1 1 0\n"
        f"{f2} /laneseg_label_w16/{CLIP}/00060.png 1 1 1 1\n")
    # val: the flags say two lanes, the file has three -> mismatch, derived (weak) orders; and a frame without lines.txt
    v0, v1 = f"/{CLIP}/00090.jpg", f"/{CLIP}/00120.jpg"
    _write_frame(root, v0.lstrip("/"), [_lane(300, 700), _lane(700, 800), _lane(1200, 900)])
    _write_frame(root, v1.lstrip("/"), None)
    (root / "list" / "val.txt").write_text(f"{v0}\n{v1}\n")
    (root / "list" / "val_gt.txt").write_text(f"{v0} /laneseg_label_w16/{CLIP}/00090.png 1 1 0 0\n"
                                              f"{v1} /laneseg_label_w16/{CLIP}/00120.png 1 1 0 0\n")
    # test: no gt file -> orders derived from x, weak
    t0 = f"/{CLIP}/00150.jpg"
    _write_frame(root, t0.lstrip("/"), [_lane(1100, 900), _lane(300, 700)])
    (root / "list" / "test.txt").write_text(f"{t0}\n")
    # official seg PNG for 00000 only: a block of slot id 2
    seg = np.zeros((H, W), np.uint8)
    seg[400:420, 100:140] = 2
    p = root / "laneseg_label_w16" / CLIP / "00000.png"
    p.parent.mkdir(parents=True)
    cv2.imwrite(str(p), seg)
    return root


def _mask(out: Path, split: str, s: Sample):
    return cv2.imread(str(out / split / "samples" / s.lane_mask), cv2.IMREAD_UNCHANGED)


def test_convert(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "culane", "--", "--root", str(root), "--dst", str(out),
                 "--splits", "train", "val", "test"]) == 0
    text = capsys.readouterr().out
    assert "4 frames" in text and "lane < 2 points: 1" in text and "lane flags mismatch: 1" in text
    assert "lane point outside image: 1" in text and "image missing: 1" in text and "lines.txt missing: 1" in text
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 2, "val": 1, "test": 1} and m["source"] == "culane" and m["licence"] == "research"

    s = Sample.load(out / "train" / "samples" / f"{STEM}__00000.npz")
    assert s.hw == (H, W) and s.calib == "nominal" and s.clip_id == CLIP and s.frame_idx == 0
    assert s.label_mask.lane_geom and s.label_mask.lane_order and s.label_mask.lane_mask
    assert not (s.label_mask.lane_type or s.label_mask.lane_cont or s.label_mask.lane_3d)
    assert [l.order for l in s.lanes] == [-2, -1, 1, 2] and not any(l.weak for l in s.lanes)
    assert all(l.ltype == -1 and l.continuity == -1 and l.points3d is None for l in s.lanes)
    assert len(s.lanes[0].points) == 30 and s.lanes[0].points[0, 1] == 579  # the x < 0 point at row 589 dropped
    for l in s.lanes:
        assert (np.diff(l.points[:, 1]) < 0).all()  # near (bottom) first, also for the top-down file
    assert s.lanes[2].points[0].tolist() == [1000, 589]
    ll = _mask(out, "train", s)
    assert ll.shape == (H, W) and ll[450, 600] == 1 and ll[100, 600] == 0 and ll[450, 100] == 0 and ll[410, 120] == 0
    assert s.K[0, 0] == pytest.approx((W / 2) / math.tan(math.radians(30))) and s.K[0, 2] == (W - 1) / 2
    assert s.K[1, 2] == pytest.approx(0.45 * (H - 1)) and s.cam2ego[2, 3] == 1.5
    img = out / "train" / "samples" / s.image
    assert img.exists() and img.is_symlink()

    s = Sample.load(out / "train" / "samples" / f"{STEM}__00030.npz")
    assert [l.order for l in s.lanes] == [-1] and not s.lanes[0].weak  # the one-point right lane is skipped

    s = Sample.load(out / "val" / "samples" / f"{STEM}__00090.npz")
    assert [l.order for l in s.lanes] == [-2, -1, 1] and all(l.weak for l in s.lanes)

    s = Sample.load(out / "test" / "samples" / f"{STEM}__00150.npz")
    assert [l.order for l in s.lanes] == [1, -1] and all(l.weak for l in s.lanes)
    with pytest.raises(FileExistsError):
        main(["convert", "culane", "--", "--root", str(root), "--dst", str(out)])


def test_seg_limit_and_options(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "culane", "--", "--root", str(root), "--dst", str(out), "--splits", "train", "val", "test",
                 "--limit", "1", "--seg", str(root / "laneseg_label_w16"), "--copy", "--hfov", "70", "--horizon", "0.4",
                 "--cam-height", "1.2"]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "seg png missing: 2" in text  # val / test frames have no PNG -> rasterised
    s = Sample.load(out / "train" / "samples" / f"{STEM}__00000.npz")
    ll = _mask(out, "train", s)
    assert ll[410, 120] == 1 and ll[450, 600] == 0 and ll.max() == 1  # the official PNG, not the polylines
    assert s.K[0, 0] == pytest.approx((W / 2) / math.tan(math.radians(35)))
    assert s.K[1, 2] == pytest.approx(0.4 * (H - 1))
    assert s.cam2ego[2, 3] == 1.2
    img = out / "train" / "samples" / s.image
    assert img.exists() and not img.is_symlink()
    assert not (out / "train" / "samples" / f"{STEM}__00030.npz").exists()
    s = Sample.load(out / "test" / "samples" / f"{STEM}__00150.npz")
    assert _mask(out, "test", s)[589, 300] == 1  # rasterised (no PNG): the near end of the left lane


def test_orders_from_x():
    assert orders_from_x([300, 700, 1200, 100, 1500], 819.5) == [-2, -1, 1, -3, 2]
    assert orders_from_x([100, 200, 300, 400, 900], 819.5) == [LANE_ORDER_UNLABELLED, -3, -2, -1, 1]
    assert orders_from_x([], 819.5) == []
