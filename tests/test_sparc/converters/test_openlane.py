"""Mini OpenLane dump: synthetic ground lanes projected with a known intrinsic under the Waymo-axes camera
convention (and once under a plain OpenCV camera frame) so the convention detector is exercised."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.converters.openlane import convention_errors, detect_convention
from sparc.data.geometry import project_ego
from sparc.data.schema import LANE_CONTINUITY, LANE_ORDER_UNLABELLED, LANE_TYPES, Sample

H, W = 1280, 1920
K = np.array([[2000.0, 0.0, 959.5], [0.0, 2000.0, 639.5], [0.0, 0.0, 1.0]])
CAM_H = 1.6
XS = np.arange(5.0, 41.0)  # forward distances; points beyond 30 m are not visible
SEG = "segment-10017090168044687777_6380_000_6400_000_with_camera_labels"
# (lateral y, attribute, category): left-left yellow-dash, left double-white-solid, right white-solid,
# right-right white-ldash-rsolid (mixed), right curb (no attribute), undefined category 13
LANES = [(-5.25, 1, 7), (-1.75, 2, 4), (1.75, 3, 2), (5.25, 4, 5), (7.0, 0, 20), (-9.0, 0, 13)]


def _waymo_points(y, xs=XS):
    """Ground lane in the camera frame with Waymo axes (x forward, y left, z up)."""
    return np.stack([xs, np.full_like(xs, y), np.full_like(xs, -CAM_H)], 1)


def _project(pw):
    cv = np.stack([-pw[:, 1], -pw[:, 2], pw[:, 0]], 1)  # (X, Y, Z) = (-y, -z, x)
    return np.stack([K[0, 0] * cv[:, 0] / cv[:, 2] + K[0, 2], K[1, 1] * cv[:, 1] / cv[:, 2] + K[1, 2]], 1)


def _lane_json(y, attr, cat, convention, vis=None, track=0):
    pw = _waymo_points(y)
    uv = _project(pw)
    xyz = pw if convention == "waymo" else np.stack([-pw[:, 1], -pw[:, 2], pw[:, 0]], 1)
    if vis is None:
        vis = (XS <= 30).astype(float)
    return {"category": cat, "visibility": vis.tolist(), "uv": uv.T.tolist(), "xyz": xyz.T.tolist(),
            "attribute": attr, "track_id": track}


def _anno(convention, lanes=LANES, extra=()):
    lane_lines = [_lane_json(y, a, c, convention, track=i) for i, (y, a, c) in enumerate(lanes)] + list(extra)
    ext = np.eye(4)
    ext[:3, 3] = (1.5, 0.0, 2.1)
    return {"intrinsic": K.tolist(), "extrinsic": ext.tolist(), "lane_lines": lane_lines}


def _write(root: Path, anno_dir: str, split: str, stem: str, anno: dict, with_image=True):
    anno["file_path"] = f"{split}/{SEG}/{stem}.jpg"
    img = root / "images" / split / SEG / f"{stem}.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    if with_image:
        cv2.imwrite(str(img), np.full((H, W, 3), 90, np.uint8))
    js = root / anno_dir / split / SEG / f"{stem}.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(anno))


@pytest.fixture(params=["waymo", "opencv"])
def convention(request):
    return request.param


@pytest.fixture
def root(tmp_path, convention):
    root = tmp_path / "openlane"
    one_visible = _lane_json(-12.0, 0, 1, convention, vis=np.r_[1.0, np.zeros(len(XS) - 1)])
    _write(root, "lane3d_1000", "training", "150913082291421700", _anno(convention, extra=[one_visible]))
    _write(root, "lane3d_1000", "training", "150913082391421700", _anno(convention, lanes=LANES[:2]))
    _write(root, "lane3d_1000", "training", "150913082491421700", _anno(convention), with_image=False)
    _write(root, "lane3d_1000", "training", "150913082591421700", _anno(convention, lanes=[]))  # no lanes
    _write(root, "lane3d_1000", "validation", "150913090000000000", _anno(convention, lanes=LANES[2:4]))
    return root


def test_convert(root, convention, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "openlane", "--", "--root", str(root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "4 frames" in text and "image missing: 1" in text and "lane < 2 visible points: 1" in text
    assert "lane point not visible: 135" in text  # 10 far points x 10 lanes written + 35 of the one-visible lane
    assert "lane category unknown: 1" in text and "height from extrinsic (frame without lanes): 1" in text
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 3, "val": 1} and m["source"] == "openlane" and m["licence"] == "research"
    assert m["xyz_convention"] == {convention: 3}

    s = Sample.load(out / "train" / "samples" / f"{SEG}__150913082291421700.npz")
    assert s.hw == (H, W) and s.calib == "measured" and np.array_equal(s.K, K)
    assert s.clip_id == SEG and s.frame_idx == 0 and s.timestamp_ns is None
    lm = s.label_mask
    assert lm.lane_geom and lm.lane_3d and lm.lane_order and lm.lane_cont and lm.lane_type and lm.lane_mask
    assert not lm.lane_dir and not lm.agents
    assert len(s.lanes) == 6  # the one-visible-point lane is skipped
    assert [l.order for l in s.lanes] == [-2, -1, 1, 2, LANE_ORDER_UNLABELLED, LANE_ORDER_UNLABELLED]
    assert [LANE_TYPES[l.ltype] for l in s.lanes] == ["single_yellow", "double_white", "single_white", "double_white",
                                                      "curb", "other"]
    assert [l.continuity for l in s.lanes] == [LANE_CONTINUITY.index("dashed"), LANE_CONTINUITY.index("solid"),
                                               LANE_CONTINUITY.index("solid"), LANE_CONTINUITY.index("solid"), -1, -1]
    assert not any(l.weak for l in s.lanes)
    # camera height from the near ground points; ego origin on the ground below the camera
    assert s.cam2ego[2, 3] == pytest.approx(CAM_H)
    assert np.allclose(s.cam2ego[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    for l, (y, _, _) in zip(s.lanes, LANES):
        assert len(l.points) == 26 and len(l.points3d) == 26  # x = 5..30 m visible
        assert np.allclose(l.points3d[:, 0], XS[:26]) and np.allclose(l.points3d[:, 1], y)  # ego x, y round-trip
        assert np.allclose(l.points3d[:, 2], 0.0, atol=1e-6)  # on the ground
        assert (np.diff(l.points[:, 1]) < 0).all() and l.points[0, 1] > l.points[-1, 1]  # near (bottom) first
        uv, z = project_ego(l.points3d, s.K, s.cam2ego)
        assert (z > 0).all() and np.abs(uv - l.points).max() < 1e-3
        assert np.allclose(l.points, _project(_waymo_points(y))[:26], atol=1e-3)
    ll = cv2.imread(str(out / "train" / "samples" / s.lane_mask), cv2.IMREAD_UNCHANGED)
    assert ll.shape == (H, W) and ll[960, 610] == 1 and ll[100, 100] == 0 and ll[960, 960] == 0  # right lane, x = 10 m
    assert (out / "train" / "samples" / s.image).is_symlink()

    s = Sample.load(out / "train" / "samples" / f"{SEG}__150913082391421700.npz")
    assert len(s.lanes) == 2 and s.frame_idx == 1
    s = Sample.load(out / "train" / "samples" / f"{SEG}__150913082591421700.npz")
    assert s.lanes == [] and s.cam2ego[2, 3] == 2.1 and s.label_mask.lane_3d  # height from the extrinsic
    assert cv2.imread(str(out / "train" / "samples" / s.lane_mask), cv2.IMREAD_UNCHANGED).max() == 0
    s = Sample.load(out / "val" / "samples" / f"{SEG}__150913090000000000.npz")
    assert [l.order for l in s.lanes] == [1, 2] and s.frame_idx == 0
    with pytest.raises(FileExistsError):
        main(["convert", "openlane", "--", "--root", str(root), "--dst", str(out)])


def test_limit_lane3d_300_and_copy(tmp_path, capsys):
    root = tmp_path / "openlane"
    _write(root, "lane3d_300", "training", "1", _anno("waymo"))
    _write(root, "lane3d_300", "training", "2", _anno("waymo"))
    out = tmp_path / "out"
    assert main(["convert", "openlane", "--", "--root", str(root), "--dst", str(out), "--limit", "1", "--copy",
                 "--licence", "own"]) == 0
    assert "1 frames" in capsys.readouterr().out
    s = Sample.load(out / "train" / "samples" / f"{SEG}__1.npz")
    assert s.licence == "own" and not (out / "train" / "samples" / s.image).is_symlink()
    assert not (out / "train" / "samples" / f"{SEG}__2.npz").exists()


def test_detect_convention():
    a, b = _anno("waymo")["lane_lines"], _anno("opencv")["lane_lines"]
    assert detect_convention(a, K) == "waymo" and detect_convention(b, K) == "opencv"
    ea, eb = convention_errors(a, K), convention_errors(b, K)
    assert ea["waymo"] < 1e-6 and ea["opencv"] > 100 and eb["opencv"] < 1e-6 and eb["waymo"] > 100
    bad = json.loads(json.dumps(a))
    for ln in bad:
        ln["uv"] = [list(reversed(ln["uv"][0])), ln["uv"][1]]  # pixels no longer match the points
    with pytest.raises(ValueError, match=r"waymo .* px, opencv .* px"):
        detect_convention(bad, K)
    assert convention_errors([], K) == {"waymo": float("inf"), "opencv": float("inf")}


def test_convention_failure_is_counted(tmp_path, capsys):
    root = tmp_path / "openlane"
    anno = _anno("waymo", lanes=LANES[:2])
    for ln in anno["lane_lines"]:
        ln["uv"] = [[u + 500.0 for u in ln["uv"][0]], ln["uv"][1]]
    _write(root, "lane3d_1000", "training", "1", anno)
    _write(root, "lane3d_1000", "training", "2", _anno("waymo", lanes=LANES[:2]))
    out = tmp_path / "out"
    assert main(["convert", "openlane", "--", "--root", str(root), "--dst", str(out), "--splits", "train"]) == 0
    text = capsys.readouterr().out
    assert "1 frames" in text and "convention undetermined: 1" in text
