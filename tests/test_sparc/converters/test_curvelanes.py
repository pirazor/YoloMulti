"""Mini CurveLanes dump: train/valid with images/ + labels/<stem>.lines.json (string coordinates)."""

import json
import math

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.schema import LANE_ORDER_UNLABELLED, Sample


def _frame(root, split, stem, hw, lines, with_image=True):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    if with_image:
        cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), np.full((*hw, 3), 90, np.uint8))
    (root / split / "labels" / f"{stem}.lines.json").write_text(
        json.dumps({"Lines": [[{"x": str(x), "y": str(y)} for x, y in line] for line in lines]}))


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "curvelanes"
    _frame(root, "train", "a", (720, 1280), [
        [(640.0, 400.0), (640.0, 719.0), (640.0, 560.0)],            # unsorted: near point second
        [(100.5, 700.0)],                                             # one point -> skipped
        [(900.0, 719.0), ("nan", 1.0), (1000.0, 600.0), (1100.0, 500.0)],
    ])
    _frame(root, "train", "b", (720, 1280), [[(1, 700), (2, 600)]], with_image=False)
    _frame(root, "valid", "v1", (1440, 2560), [[(1200, 1439), (1300, 1000)], [(2000, 1439), (1800, 900)]])
    _frame(root, "valid", "v2", (660, 1570), [[(700, 659), (800, 300)]])
    return root


def test_convert(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "curvelanes", "--", "--root", str(root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "lane < 2 points: 1" in text and "lane point not finite: 1" in text
    assert "image missing: 1" in text
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 1, "val": 2} and m["source"] == "curvelanes"

    s = Sample.load(out / "train" / "samples" / "a.npz")
    assert s.hw == (720, 1280) and s.calib == "nominal" and s.licence == "research"
    assert s.label_mask.lane_geom and s.label_mask.lane_mask
    assert not (s.label_mask.lane_order or s.label_mask.lane_type or s.label_mask.lane_cont or s.label_mask.lane_3d)
    assert len(s.lanes) == 2
    assert s.lanes[0].points.tolist() == [[640, 719], [640, 560], [640, 400]]  # near first
    assert s.lanes[1].points.tolist() == [[900, 719], [1000, 600], [1100, 500]]
    assert all(l.order == LANE_ORDER_UNLABELLED and l.ltype == -1 and l.continuity == -1 and not l.weak
               and l.points3d is None for l in s.lanes)
    ll = cv2.imread(str(out / "train" / "samples" / s.lane_mask), cv2.IMREAD_UNCHANGED)
    assert ll.shape == (720, 1280) and ll[500, 640] == 1 and ll[300, 640] == 0 and ll[100, 100] == 0
    assert s.K[0, 0] == pytest.approx(640 / math.tan(math.radians(30))) and s.K[0, 2] == 639.5
    assert s.K[1, 2] == pytest.approx(0.5 * 719) and s.cam2ego[2, 3] == 1.5
    assert (out / "train" / "samples" / s.image).is_symlink()

    s = Sample.load(out / "val" / "samples" / "v1.npz")
    assert s.hw == (1440, 2560) and len(s.lanes) == 2 and s.lanes[1].points[0].tolist() == [2000, 1439]
    ll = cv2.imread(str(out / "val" / "samples" / s.lane_mask), cv2.IMREAD_UNCHANGED)
    assert ll.shape == (1440, 2560) and ll[1439, 1200] == 1 and ll[1439, 1600] == 0
    assert Sample.load(out / "val" / "samples" / "v2.npz").hw == (660, 1570)
    with pytest.raises(FileExistsError):
        main(["convert", "curvelanes", "--", "--root", str(root), "--dst", str(out)])


def test_limit_and_split_names(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "curvelanes", "--", "--root", str(root), "--dst", str(out), "--splits", "valid", "test",
                 "--limit", "1", "--copy", "--hfov", "50", "--cam-height", "1.3", "--horizon", "0.45"]) == 0
    text = capsys.readouterr().out
    assert "1 frames" in text
    s = Sample.load(out / "valid" / "samples" / "v1.npz")
    assert not (out / "valid" / "samples" / "v2.npz").exists()
    assert s.K[0, 0] == pytest.approx(1280 / math.tan(math.radians(25))) and s.K[1, 2] == pytest.approx(0.45 * 1439)
    assert s.cam2ego[2, 3] == 1.3 and not (out / "valid" / "samples" / s.image).is_symlink()
