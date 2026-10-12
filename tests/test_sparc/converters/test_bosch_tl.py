"""Mini Bosch Small Traffic Lights dump: train.yaml / test.yaml + rgb/train/<bag>/*.png, rgb/test/*.png."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from sparc.cli import main
from sparc.data.schema import EL_LIGHT, ELEMENT_ATTRS, Sample

H, W = 72, 128


def _img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((H, W, 3), 50, np.uint8))


def _box(label, x_min, y_min, x_max, y_max, occluded=False):
    return {"label": label, "occluded": occluded, "x_max": float(x_max), "x_min": float(x_min), "y_max": float(y_max),
            "y_min": float(y_min)}


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "bosch"
    _img(root / "rgb" / "train" / "2015-10-05-10-52-01_bag" / "24088.png")
    _img(root / "rgb" / "train" / "2015-10-05-10-52-01_bag" / "24089.png")
    _img(root / "rgb" / "test" / "24068.png")
    train = [{"boxes": [_box("Green", 10, 10, 14, 20), _box("RedLeft", 20, 10, 24, 20, True), _box("off", 30, 10, 34, 20),
                        _box("Purple", 40, 10, 44, 20), _box("Yellow", 50, 10, 50, 20)],
              "path": "./rgb/train/2015-10-05-10-52-01_bag/24088.png"},
             {"boxes": [], "path": "./rgb/train/2015-10-05-10-52-01_bag/24089.png"},
             {"boxes": [], "path": "./rgb/train/2015-10-05-10-52-01_bag/99999.png"}]
    test = [{"boxes": [_box("Red", 1, 1, 5, 9)],
             "path": "/net/pal-soc1.us.bosch.com/ifs/data/Shared_Exports/deep_learning_data/traffic_lights/university_run1/24068.png"}]
    (root / "train.yaml").write_text(yaml.safe_dump(train))
    (root / "test.yaml").write_text(yaml.safe_dump(test))
    return root


def test_convert(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "bosch_tl", "--", "--root", str(root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "image missing: 1" in text
    assert "light arrow meaning dropped: 1" in text and "light label Purple: 1" in text and "degenerate box: 1" in text
    s = Sample.load(out / "train" / "samples" / "2015-10-05-10-52-01_bag_24088.npz")
    assert s.source == "bosch_tl" and s.hw == (H, W) and s.calib == "nominal" and s.cam2ego[2, 3] == 1.4
    assert s.el_boxes.tolist() == [[10, 10, 14, 20], [20, 10, 24, 20], [30, 10, 34, 20]]
    assert s.el_cls.tolist() == [EL_LIGHT] * 3
    assert s.el_attr.tolist() == [ELEMENT_ATTRS.index("green"), ELEMENT_ATTRS.index("red"), 0]
    lm = s.label_mask
    assert lm.elements and lm.el_attr and lm.el_vocab == (EL_LIGHT,) and not lm.agents
    assert s.K[0, 0] == pytest.approx((W / 2) / np.tan(np.radians(27.5)))
    e = Sample.load(out / "train" / "samples" / "2015-10-05-10-52-01_bag_24089.npz")
    assert len(e.el_boxes) == 0 and e.label_mask.elements
    v = Sample.load(out / "val" / "samples" / "test_24068.npz")  # absolute path resolved under rgb/test
    assert v.el_attr.tolist() == [ELEMENT_ATTRS.index("red")]
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 2, "val": 1}


def test_limit_and_explicit_yaml(root, tmp_path, capsys):
    alt = root / "label_files"
    alt.mkdir()
    (alt / "my_test.yaml").write_text(yaml.safe_dump([{"boxes": [_box("GreenStraight", 1, 1, 3, 5)], "path": "./rgb/test/24068.png"}]))
    out = tmp_path / "out"
    assert main(["convert", "bosch_tl", "--", "--root", str(root), "--dst", str(out), "--limit", "1", "--copy",
                 "--val-yaml", str(alt / "my_test.yaml")]) == 0
    text = capsys.readouterr().out
    assert "2 frames" in text and "{'train': 1, 'val': 1}" in text and "light arrow meaning dropped: 2" in text
    v = Sample.load(out / "val" / "samples" / "test_24068.npz")
    assert v.el_attr.tolist() == [ELEMENT_ATTRS.index("green")]
    assert not (out / "val" / "samples" / v.image).is_symlink()
