"""Mini BDD100K dumps in the Scalabel (det_20 / lane / drivable) and the legacy single-file layout."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.schema import EL_LIGHT, EL_SIGN, ELEMENT_ATTRS, LANE_TYPES, Sample

H, W = 720, 1280


def _img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((H, W, 3), 90, np.uint8))


def _det_labels():
    return [
        {"id": "1", "category": "car", "attributes": {"occluded": False, "truncated": False, "trafficLightColor": "NA"},
         "box2d": {"x1": 100, "y1": 300, "x2": 300, "y2": 420}},
        {"id": "2", "category": "traffic light", "attributes": {"trafficLightColor": "G"}, "box2d": {"x1": 600, "y1": 100, "x2": 620, "y2": 150}},
        {"id": "3", "category": "traffic sign", "attributes": {"trafficLightColor": "NA"}, "box2d": {"x1": 900, "y1": 200, "x2": 950, "y2": 250}},
        {"id": "4", "category": "bike", "attributes": {}, "box2d": {"x1": 50, "y1": 400, "x2": 90, "y2": 460}},
        {"id": "5", "category": "other vehicle", "attributes": {}, "box2d": {"x1": 50, "y1": 400, "x2": 90, "y2": 460}},
        {"id": "6", "category": "truck", "attributes": {}, "box2d": {"x1": 50, "y1": 400, "x2": 50, "y2": 460}},
    ]


def _lane_labels(scalabel: bool):
    def lab(cat, direction, style, verts, types="LL"):
        if scalabel:
            return {"category": cat, "attributes": {"laneDirection": direction, "laneStyle": style},
                    "poly2d": [{"vertices": verts, "types": types, "closed": False}]}
        return {"category": "lane", "attributes": {"laneDirection": direction, "laneStyle": style, "laneType": cat},
                "poly2d": [{"vertices": verts, "types": types, "closed": False}]}
    return [
        lab("single white", "parallel", "dashed", [[400, 700], [600, 400]]),
        lab("double yellow", "parallel", "solid", [[800, 700], [820, 600], [840, 500], [700, 400]], types="LCCL"),
        lab("crosswalk", "vertical", "solid", [[100, 650], [1100, 650]]),
        lab("road curb", "vertical", "solid", [[100, 600], [1100, 600]]),
    ]


def _drv_labels(scalabel: bool):
    direct = [[300, 720], [980, 720], [700, 400], [580, 400]]
    alt = [[980, 720], [1280, 720], [1280, 500], [700, 400]]
    if scalabel:
        return [{"category": "direct", "poly2d": [{"vertices": direct, "types": "LLLL", "closed": True}]},
                {"category": "alternative", "poly2d": [{"vertices": alt, "types": "LLLL", "closed": True}]}]
    return [{"category": "drivable area", "attributes": {"areaType": "direct"}, "poly2d": [{"vertices": direct, "types": "LLLL", "closed": True}]},
            {"category": "drivable area", "attributes": {"areaType": "alternative"}, "poly2d": [{"vertices": alt, "types": "LLLL", "closed": True}]}]


@pytest.fixture
def scalabel_root(tmp_path):
    root = tmp_path / "bdd"
    for split, names in (("train", ["a1", "a2"]), ("val", ["v1"])):
        for n in names:
            _img(root / "images" / "100k" / split / f"{n}.jpg")
        det = [{"name": f"{n}.jpg", "attributes": {}, "timestamp": 10000, "labels": _det_labels()} for n in names]
        (root / "labels" / "det_20").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "det_20" / f"det_{split}.json").write_text(json.dumps(det))
        (root / "labels" / "lane" / "polygons").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "lane" / "polygons" / f"lane_{split}.json").write_text(
            json.dumps([{"name": f"{n}.jpg", "labels": _lane_labels(True)} for n in names[:1]]))  # a2 / v1: no lanes
        (root / "labels" / "drivable" / "polygons").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "drivable" / "polygons" / f"drivable_{split}.json").write_text(
            json.dumps([{"name": f"{n}.jpg", "labels": _drv_labels(True)} for n in names]))
    return root


@pytest.fixture
def legacy_root(tmp_path):
    root = tmp_path / "bdd_legacy"
    for split, names in (("train", ["b1"]), ("val", ["b2"])):
        for n in names:
            _img(root / "images" / "100k" / split / f"{n}.jpg")
        frames = [{"name": f"{n}.jpg", "attributes": {}, "timestamp": 1,
                   "labels": _det_labels() + _lane_labels(False) + _drv_labels(False)} for n in names]
        (root / "labels").mkdir(parents=True, exist_ok=True)
        (root / "labels" / f"bdd100k_labels_images_{split}.json").write_text(json.dumps(frames))
    return root


def _check_frame(s: Sample, with_lanes: bool, root: Path, split: str):
    assert s.hw == (H, W) and s.calib == "nominal" and s.source == "bdd100k"
    assert s.boxes2d.tolist() == [[100, 300, 300, 420], [50, 400, 90, 460]] and s.cls.tolist() == [2, 7]  # car, bicycle
    assert s.el_cls.tolist() == [EL_LIGHT, EL_SIGN]
    assert s.el_attr.tolist() == [ELEMENT_ATTRS.index("green"), -1]
    assert s.label_mask.agents and s.label_mask.elements and s.label_mask.el_attr and s.label_mask.da
    assert s.label_mask.cls_vocab == tuple(range(8)) and s.label_mask.el_vocab == (0, 1)
    da = cv2.imread(str(root / split / "samples" / s.da_mask), cv2.IMREAD_UNCHANGED)
    assert da.shape == (H, W) and da[700, 640] == 1 and da[700, 1100] == 2 and da[100, 100] == 0
    assert s.label_mask.lane_geom == with_lanes and s.label_mask.lane_mask == with_lanes
    if with_lanes:
        assert len(s.lanes) == 2  # crosswalk and the vertical curb skipped
        assert s.lanes[0].continuity == 1 and s.lanes[0].ltype == LANE_TYPES.index("single_white")
        assert s.lanes[0].points[0].tolist() == [400, 700]  # near end first
        assert s.lanes[1].ltype == LANE_TYPES.index("double_yellow") and len(s.lanes[1].points) > 4  # bezier flattened
        ll = cv2.imread(str(root / split / "samples" / s.lane_mask), cv2.IMREAD_UNCHANGED)
        assert ll[550, 500] == 1 and ll[650, 100] == 0  # on the dashed line, not on the crosswalk
    else:
        assert s.lanes == [] and s.lane_mask is None
    img = (root / split / "samples" / s.image)
    assert img.exists() and img.is_symlink()


def test_scalabel_layout(scalabel_root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "bdd100k", "--", "--root", str(scalabel_root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "box category other vehicle: 3" in text and "degenerate box: 3" in text
    assert "lane crosswalk: 2" in text and "lane vertical: 2" in text  # a1 and v1 carry lanes, a2 does not
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 2, "val": 1} and m["source"] == "bdd100k"
    _check_frame(Sample.load(out / "train" / "samples" / "a1.npz"), True, out, "train")
    _check_frame(Sample.load(out / "train" / "samples" / "a2.npz"), False, out, "train")
    _check_frame(Sample.load(out / "val" / "samples" / "v1.npz"), True, out, "val")
    with pytest.raises(FileExistsError):
        main(["convert", "bdd100k", "--", "--root", str(scalabel_root), "--dst", str(out)])


def test_legacy_layout_and_limit(legacy_root, tmp_path):
    out = tmp_path / "out"
    assert main(["convert", "bdd100k", "--", "--root", str(legacy_root), "--dst", str(out), "--limit", "1", "--copy",
                 "--horizon", "0.45", "--hfov", "70"]) == 0
    s = Sample.load(out / "train" / "samples" / "b1.npz")
    assert s.K[1, 2] == pytest.approx(0.45 * (H - 1)) and s.K[0, 2] == (W - 1) / 2
    assert s.cam2ego[2, 3] == 1.4
    img = out / "train" / "samples" / s.image
    assert img.exists() and not img.is_symlink()
    # the legacy single file carries all three label types
    assert s.label_mask.lane_geom and s.label_mask.da and len(s.lanes) == 2
    assert (out / "val" / "samples" / "b2.npz").exists()
