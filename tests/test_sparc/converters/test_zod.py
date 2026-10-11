"""Mini ZOD Frames dump: trainval-frames-mini.json + single_frames/<id>/{annotations,camera_front_blur,calibration.json}."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.schema import AGENT_CLASSES, EL_LIGHT, EL_SIGN, Sample

H, W = 54, 96
K_FILE = [[80.0, 0.0, 47.5], [0.0, 80.0, 26.5], [0.0, 0.0, 1.0]]
EXTR = [[0.0, 0.0, 1.0, 2.1], [-1.0, 0.0, 0.0, 0.3], [0.0, -1.0, 0.0, 1.55], [0.0, 0.0, 0.0, 1.0]]


def _rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _feat(coords, **props):
    props.setdefault("annotation_uuid", "u")
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": coords}, "properties": props}


def _objects():
    return [
        _feat(_rect(10, 20, 30, 40), **{"class": "Vehicle", "type": "Car", "unclear": False, "occlusion_ratio": "Light",
                                        "location_3d": {"coordinates": [1, 2, 3]}, "size_3d_length": 4,
                                        "size_3d_width": 2, "size_3d_height": 1.5, "orientation_3d_qw": 1,
                                        "orientation_3d_qx": 0, "orientation_3d_qy": 0, "orientation_3d_qz": 0}),
        _feat(_rect(40, 20, 50, 45), **{"class": "VulnerableVehicle", "type": "Bicycle", "with_rider": "True", "unclear": False}),
        _feat(_rect(60, 20, 65, 45), **{"class": "Pedestrian", "unclear": False, "occlusion_ratio": "None"}),
        _feat(_rect(70, 20, 80, 45), **{"class": "Vehicle", "type": "Other", "unclear": False}),
        _feat(_rect(70, 30, 80, 40), **{"class": "Vehicle", "type": "Bus", "unclear": True}),
        _feat(_rect(1, 1, 5, 5), **{"class": "TrafficSign", "traffic_content_visible": "True", "unclear": False}),
        _feat(_rect(6, 1, 9, 5), **{"class": "TrafficSign", "traffic_content_visible": "False", "unclear": False}),
        _feat(_rect(20, 1, 24, 8), **{"class": "TrafficSignal", "traffic_content_visible": "True", "unclear": False}),
        _feat(_rect(30, 1, 34, 8), **{"class": "TrafficSignal", "traffic_content_visible": "False", "unclear": False}),
        _feat(_rect(85, 40, 95, 50), **{"class": "Animal", "unclear": False}),
        _feat(_rect(85, 10, 95, 20), **{"class": "Inconclusive", "unclear": True}),
        _feat(_rect(85, 20, 85, 30), **{"class": "Pedestrian", "unclear": False}),  # degenerate
    ]


def _signs():
    return [_feat(_rect(1, 1, 5, 5), **{"class": "Priority_Stop", "occlusion_ratio": "None", "is_electronic": False, "unclear": False}),
            _feat(_rect(50, 1, 55, 6), **{"class": "Unclear", "unclear": True}),
            _feat(_rect(56, 1, 60, 6), **{"class": "NotListed", "unclear": False})]


def _ego_road():
    return [_feat(_rect(20, 30, 80, 53), **{"class": "EgoRoad_Road"}),
            _feat([[[0, 50], [10, 50], [10, 53], [0, 53]]], **{"class": "EgoRoad_Debris"}),  # nested ring
            _feat(_rect(0, 0, 5, 5), **{"class": "Barrier"})]


def _lanes():
    return [_feat(_rect(44, 30, 46, 53), **{"class": "solid", "coloured": False}),
            _feat(_rect(0, 0, 3, 3), **{"class": "shaded_area", "ShadedAreaType": "x", "coloured": False}),
            _feat(_rect(5, 5, 8, 8), ContainsArrow=True)]


def _frame(root: Path, fid: str, annos: dict, calib: bool):
    d = root / "single_frames" / fid
    (d / "annotations").mkdir(parents=True)
    for name, feats in annos.items():
        (d / "annotations" / f"{name}.json").write_text(json.dumps(feats))
    (d / "camera_front_blur").mkdir()
    cv2.imwrite(str(d / "camera_front_blur" / f"{fid}_2022-02-14T13:23:33.029609Z.jpg"), np.full((H, W, 3), 60, np.uint8))
    if calib:
        (d / "calibration.json").write_text(json.dumps({"FC": {"intrinsics": K_FILE, "distortion": [0.1, 0.0, 0.0, 0.0],
                                                               "undistortion": [0, 0, 0, 0], "extrinsics": EXTR,
                                                               "image_dimensions": [W, H], "field_of_view": [100, 60]},
                                                        "lidar_extrinsics": EXTR}))


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "zod"
    _frame(root, "000001", {"object_detection": _objects(), "traffic_signs": _signs(), "ego_road": _ego_road(),
                            "lane_markings": _lanes()}, calib=True)
    _frame(root, "000002", {"traffic_signs": _signs()}, calib=False)
    _frame(root, "000003", {"object_detection": _objects()[:3]}, calib=True)
    (root / "single_frames" / "000004").mkdir()  # no image, no labels
    (root / "trainval-frames-mini.json").write_text(json.dumps({
        "train": [{"id": "000001", "calibration_path": "single_frames/000001/calibration.json"}, "4", "000003"],
        "val": ["000002"]}))
    return root


def test_convert(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "zod", "--", "--root", str(root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "image missing: 1" in text and "calibration missing: 1" in text
    for reason in ("3d box ignored: 2", "sign back side: 1", "light back side: 1", "sign box replaced by traffic_signs: 1",
                   "object class Vehicle/Other: 1", "object class Animal: 1", "unclear object: 1", "degenerate box: 1",
                   "ego road class Barrier: 1", "lane shaded area: 1", "road painting: 1"):
        assert reason in text, reason
    s = Sample.load(out / "train" / "samples" / "000001.npz")
    assert s.source == "zod" and s.licence == "bsd" and s.calib == "measured" and s.hw == (H, W)
    assert s.K.tolist() == K_FILE
    assert s.cam2ego[:3, :3].tolist() == [r[:3] for r in EXTR[:3]] and s.cam2ego[:3, 3].tolist() == [0.0, 0.0, 1.55]
    assert s.boxes2d.tolist() == [[10, 20, 30, 40], [40, 20, 50, 45], [60, 20, 65, 45], [70, 30, 80, 40]]
    assert s.cls.tolist() == [AGENT_CLASSES.index(c) for c in ("car", "bicycle", "pedestrian", "bus")]
    # light from object_detection, signs from traffic_signs (named)
    assert s.el_cls.tolist() == [EL_LIGHT, EL_SIGN, EL_SIGN, EL_SIGN]
    assert s.el_boxes.tolist() == [[20, 1, 24, 8], [1, 1, 5, 5], [50, 1, 55, 6], [56, 1, 60, 6]]
    assert s.el_sign.tolist() == ["", "zod:Priority_Stop", "", ""] and s.el_attr.tolist() == [-1, -1, -1, -1]
    lm = s.label_mask
    assert lm.agents and lm.cls_vocab == tuple(AGENT_CLASSES.index(c) for c in ("pedestrian", "car", "truck", "bus", "train", "motorcycle", "bicycle"))
    assert lm.elements and lm.el_vocab == (EL_LIGHT, EL_SIGN) and lm.el_sign and not lm.el_attr
    assert lm.da and lm.lane_mask and not lm.lane_geom and s.lanes == []
    da = cv2.imread(str(out / "train" / "samples" / s.da_mask), cv2.IMREAD_UNCHANGED)
    assert da.shape == (H, W) and da[45, 50] == 1 and da[52, 5] == 1 and da[2, 2] == 0 and da[10, 50] == 0
    ll = cv2.imread(str(out / "train" / "samples" / s.lane_mask), cv2.IMREAD_UNCHANGED)
    assert ll[45, 45] == 1 and ll[1, 1] == 0 and ll[6, 6] == 0 and ll[45, 50] == 0
    # signs-only frame without calibration: nominal camera, el_vocab = (sign,)
    v = Sample.load(out / "val" / "samples" / "000002.npz")
    assert v.calib == "nominal" and v.cam2ego[2, 3] == 1.6 and v.K[0, 2] == (W - 1) / 2
    assert v.label_mask.el_vocab == (EL_SIGN,) and not v.label_mask.agents and v.label_mask.el_sign
    assert v.el_sign.tolist() == ["zod:Priority_Stop", "", ""] and v.da_mask is None and v.lane_mask is None
    # object_detection only: signs come from it, unnamed
    t = Sample.load(out / "train" / "samples" / "000003.npz")
    assert t.label_mask.el_vocab == (EL_LIGHT, EL_SIGN) and not t.label_mask.el_sign and len(t.boxes2d) == 3
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 2, "val": 1} and m["signs"] == ["zod:Priority_Stop"]


def test_limit_full_version_and_dnat(root, tmp_path, capsys):
    full = root / "trainval-frames-full.json"
    full.write_text(json.dumps({"train": ["000003", "000001"], "val": []}))
    d = root / "single_frames" / "000003" / "camera_front_dnat"
    d.mkdir()
    cv2.imwrite(str(d / "000003_x.jpg"), np.full((H, W, 3), 60, np.uint8))
    out = tmp_path / "out"
    assert main(["convert", "zod", "--", "--root", str(root), "--dst", str(out), "--limit", "1", "--anonymization",
                 "dnat", "--copy"]) == 0
    text = capsys.readouterr().out
    assert "1 frames" in text and "{'train': 1}" in text
    s = Sample.load(out / "train" / "samples" / "000003.npz")
    assert not (out / "train" / "samples" / s.image).is_symlink()
    assert main(["convert", "zod", "--", "--root", str(root), "--dst", str(out), "--version", "mini", "--overwrite",
                 "--splits", "val"]) == 0
    assert "{'val': 1}" in capsys.readouterr().out
