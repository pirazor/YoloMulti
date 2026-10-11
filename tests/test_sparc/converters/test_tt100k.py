"""Mini TT100K dump: annotations.json (2016) / annotations_all.json (2021) + train/, test/, other/ images."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.schema import EL_SIGN, Sample

H, W = 64, 64


def _img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((H, W, 3), 70, np.uint8))


def _obj(cat, xmin, ymin, xmax, ymax):
    return {"category": cat, "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "ellipse_org": [[xmin, ymin], [xmax, ymax]], "polygon": [[xmin, ymin], [xmax, ymin], [xmax, ymax]]}


def _annotations():
    return {"types": ["pn", "pne", "i5"], "imgs": {
        "1": {"id": 1, "path": "train/1.jpg", "objects": [_obj("pn", 10, 10, 20, 20), _obj("i5", 30, 30, 30, 40)]},
        "2": {"id": 2, "path": "test/2.jpg", "objects": [_obj("pne", 5, -4, 15, 10)]},
        "3": {"id": 3, "path": "other/3.jpg", "objects": []},
        "4": {"id": 4, "path": "train/4.jpg", "objects": []},
        "5": {"id": 5, "path": "train/5.jpg", "objects": [_obj("pn", 1, 1, 9, 9)]},
    }}


def _make(root: Path, name: str):
    for p in ("train/1.jpg", "test/2.jpg", "other/3.jpg", "train/4.jpg"):  # 5.jpg is missing on purpose
        _img(root / p)
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(_annotations()))


def test_2016_layout(tmp_path, capsys):
    root = tmp_path / "tt100k" / "data"
    _make(root, "annotations.json")
    out = tmp_path / "out"
    assert main(["convert", "tt100k", "--", "--root", str(root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "other split: 1" in text and "image missing: 1" in text
    assert "box clipped: 1" in text and "degenerate box: 1" in text
    s = Sample.load(out / "train" / "samples" / "1.npz")
    assert s.source == "tt100k" and s.hw == (H, W) and s.calib == "nominal" and s.cam2ego[2, 3] == 1.4
    assert s.el_boxes.tolist() == [[10, 10, 20, 20]] and s.el_cls.tolist() == [EL_SIGN]
    assert s.el_sign.tolist() == ["tt100k:pn"] and s.el_attr.tolist() == [-1]
    lm = s.label_mask
    assert lm.elements and lm.el_sign and lm.el_vocab == (EL_SIGN,) and not lm.agents and not lm.el_attr
    assert s.K[0, 0] == pytest.approx((W / 2) / np.tan(np.radians(30)))
    e = Sample.load(out / "train" / "samples" / "4.npz")
    assert len(e.el_boxes) == 0 and e.label_mask.elements
    v = Sample.load(out / "val" / "samples" / "2.npz")
    assert v.el_boxes.tolist() == [[5, 0, 15, 10]] and v.el_sign.tolist() == ["tt100k:pne"]
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 2, "val": 1} and m["types"] == ["pn", "pne", "i5"]
    assert m["signs"] == ["tt100k:pn", "tt100k:pne"]


def test_2021_layout_root_discovery_and_limit(tmp_path, capsys):
    root = tmp_path / "tt100k_2021"
    _make(root / "data", "annotations_all.json")
    out = tmp_path / "out"
    assert main(["convert", "tt100k", "--", "--root", str(root), "--dst", str(out), "--limit", "1", "--copy",
                 "--splits", "train"]) == 0
    text = capsys.readouterr().out
    assert "1 frames" in text and "{'train': 1}" in text and "test split: 1" in text
    s = Sample.load(out / "train" / "samples" / "1.npz")
    img = out / "train" / "samples" / s.image
    assert img.exists() and not img.is_symlink()
    assert not (out / "val").exists()
    with pytest.raises(FileNotFoundError):
        main(["convert", "tt100k", "--", "--root", str(tmp_path / "nowhere"), "--dst", str(tmp_path / "out2")])
