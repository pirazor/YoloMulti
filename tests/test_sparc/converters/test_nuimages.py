"""Mini nuImages dump: v1.0-train / v1.0-val / v1.0-test table directories + samples/CAM_FRONT images."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.converters.nuimages import cam2ego, map_category, quat_to_rot
from sparc.data.schema import AGENT_CLASSES, Sample, ground_cam2ego

H, W = 90, 160
K = [[120.0, 0.0, 80.0], [0.0, 120.0, 45.0], [0.0, 0.0, 1.0]]
Q_FRONT = [0.5, -0.5, 0.5, -0.5]  # w, x, y, z: camera (x right, y down, z fwd) -> ego (x fwd, y left, z up)
CATS = {"c_ped": "human.pedestrian.adult", "c_child": "human.pedestrian.child", "c_car": "vehicle.car",
        "c_truck": "vehicle.truck", "c_bus": "vehicle.bus.rigid", "c_police": "vehicle.emergency.police",
        "c_trailer": "vehicle.trailer", "c_moto": "vehicle.motorcycle", "c_bike": "vehicle.bicycle",
        "c_cone": "movable_object.trafficcone", "c_ego": "vehicle.ego"}


def _img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((H, W, 3), 30, np.uint8))


def _ann(token, sd, cat, bbox):
    return {"token": token, "sample_data_token": sd, "category_token": cat, "attribute_tokens": [], "bbox": bbox,
            "mask": None}


def _tables(root: Path, version: str, frames, anns):
    d = root / version
    d.mkdir(parents=True)
    sensors = [{"token": "s_front", "channel": "CAM_FRONT", "modality": "camera"},
               {"token": "s_back", "channel": "CAM_BACK", "modality": "camera"}]
    calibrated = [{"token": "cs_front", "sensor_token": "s_front", "translation": [1.7, 0.02, 1.5], "rotation": Q_FRONT,
                   "camera_intrinsic": K, "camera_distortion": [0, 0, 0, 0, 0]},
                  {"token": "cs_back", "sensor_token": "s_back", "translation": [0.0, 0.0, 1.5], "rotation": [0.5, 0.5, -0.5, -0.5],
                   "camera_intrinsic": K, "camera_distortion": [0, 0, 0, 0, 0]}]
    sample_data = []
    for i, (token, fname, key, cs) in enumerate(frames):
        sample_data.append({"token": token, "sample_token": f"sample_{i}", "ego_pose_token": "ep", "calibrated_sensor_token": cs,
                            "filename": fname, "fileformat": "jpg", "width": W, "height": H, "timestamp": 1533151603512404 + i,
                            "is_key_frame": key, "next": "", "prev": ""})
        if key and "missing" not in fname:
            _img(root / fname)
    categories = [{"token": t, "name": n, "description": ""} for t, n in CATS.items()]
    for name, table in (("sensor", sensors), ("calibrated_sensor", calibrated), ("sample_data", sample_data),
                        ("object_ann", anns), ("category", categories)):
        (d / f"{name}.json").write_text(json.dumps(table))


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "nuimages"
    _tables(root, "v1.0-train", [
        ("sd1", "samples/CAM_FRONT/n008-2018__CAM_FRONT__1533151603512404.jpg", True, "cs_front"),
        ("sd2", "samples/CAM_FRONT/n008-2018__CAM_FRONT__1533151604012404.jpg", True, "cs_front"),
        ("sd3", "sweeps/CAM_FRONT/n008-2018__CAM_FRONT__1533151603612404.jpg", False, "cs_front"),
        ("sd4", "samples/CAM_BACK/n008-2018__CAM_BACK__1533151603512404.jpg", True, "cs_back"),
        ("sd5", "samples/CAM_FRONT/missing.jpg", True, "cs_front"),
    ], [
        _ann("a1", "sd1", "c_ped", [10, 20, 20, 50]), _ann("a2", "sd1", "c_car", [30, 20, 60, 50]),
        _ann("a3", "sd1", "c_bus", [70, 10, 120, 50]), _ann("a4", "sd1", "c_police", [125, 20, 150, 50]),
        _ann("a5", "sd1", "c_cone", [1, 1, 5, 5]), _ann("a6", "sd1", "c_trailer", [1, 60, 40, 80]),
        _ann("a7", "sd1", "c_bike", [50, 60, 60, 80]), _ann("a8", "sd1", "c_moto", [70, 60, 80, 80]),
        _ann("a9", "sd1", "c_child", [90, 60, 95, 80]), _ann("a10", "sd1", "c_ego", [0, 80, 160, 90]),
        _ann("a11", "sd1", "c_car", [100, 60, 100, 80]),
        _ann("a12", "sd4", "c_car", [1, 1, 10, 10]),
    ])
    (root / "v1.0-train" / "sample_data.json").touch()  # keep mtime order irrelevant
    _tables(root, "v1.0-val", [("sdv", "samples/CAM_FRONT/val__CAM_FRONT__1.jpg", True, "cs_front")],
            [_ann("av", "sdv", "c_truck", [5, 5, 50, 40])])
    _tables(root, "v1.0-test", [("sdt", "samples/CAM_FRONT/test__CAM_FRONT__1.jpg", True, "cs_front")], [])
    return root


def test_helpers():
    assert np.allclose(quat_to_rot(Q_FRONT), ground_cam2ego(1.5)[:3, :3])
    T = cam2ego({"rotation": Q_FRONT, "translation": [1.7, 0.02, 1.5]})
    assert np.allclose(T, ground_cam2ego(1.5))
    assert map_category("human.pedestrian.police_officer") == "pedestrian" and map_category("vehicle.bus.bendy") == "bus"
    assert map_category("vehicle.emergency.ambulance") == "truck" and map_category("vehicle.construction") == "truck"
    assert map_category("animal") is None and map_category("static_object.bicycle_rack") is None


def test_convert(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "nuimages", "--", "--root", str(root), "--dst", str(out), "--splits", "train", "val", "test"]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "image missing: 1" in text and "v1.0-test unannotated: 1" in text
    assert "category movable_object.trafficcone: 1" in text and "category vehicle.ego: 1" in text and "degenerate box: 1" in text
    s = Sample.load(out / "train" / "samples" / "n008-2018__CAM_FRONT__1533151603512404.npz")
    assert s.source == "nuimages" and s.licence == "research" and s.hw == (H, W) and s.calib == "measured"
    assert s.K.tolist() == K and np.allclose(s.cam2ego, ground_cam2ego(1.5))
    assert s.timestamp_ns == 1533151603512404 * 1000
    assert s.boxes2d.tolist() == [[10, 20, 20, 50], [30, 20, 60, 50], [70, 10, 120, 50], [125, 20, 150, 50], [1, 60, 40, 80],
                                  [50, 60, 60, 80], [70, 60, 80, 80], [90, 60, 95, 80]]
    assert s.cls.tolist() == [AGENT_CLASSES.index(c) for c in ("pedestrian", "car", "bus", "truck", "truck", "bicycle",
                                                               "motorcycle", "pedestrian")]
    lm = s.label_mask
    assert lm.agents and not lm.elements and lm.el_vocab == ()
    assert lm.cls_vocab == tuple(AGENT_CLASSES.index(c) for c in ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"))
    assert len(s.el_boxes) == 0
    e = Sample.load(out / "train" / "samples" / "n008-2018__CAM_FRONT__1533151604012404.npz")
    assert len(e.boxes2d) == 0 and e.label_mask.agents  # a key frame without objects is still labelled
    assert not list((out / "train" / "samples").glob("*CAM_BACK*"))
    v = Sample.load(out / "val" / "samples" / "val__CAM_FRONT__1.npz")
    assert v.cls.tolist() == [AGENT_CLASSES.index("truck")]
    assert not (out / "test").exists()
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 2, "val": 1}


def test_limit_copy_and_version_names(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "nuimages", "--", "--root", str(root), "--dst", str(out), "--limit", "1", "--copy",
                 "--splits", "v1.0-train"]) == 0
    text = capsys.readouterr().out
    assert "1 frames" in text and "{'train': 1}" in text
    s = Sample.load(out / "train" / "samples" / "n008-2018__CAM_FRONT__1533151603512404.npz")
    assert not (out / "train" / "samples" / s.image).is_symlink()
    assert main(["convert", "nuimages", "--", "--root", str(root), "--dst", str(out), "--overwrite", "--splits", "mini"]) == 0
    assert "0 frames" in capsys.readouterr().out
