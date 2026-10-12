"""Mini MTSD dump: images/<key>.jpg, annotations/<key>.json, splits/<split>.txt."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.schema import EL_SIGN, Sample

H, W = 60, 100


def _img(path: Path, hw=(H, W)):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((hw[0], hw[1], 3), 80, np.uint8))


def _anno(path: Path, hw, objects, ispano=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"width": hw[1], "height": hw[0], "ispano": ispano, "objects": objects}))


def _obj(label, xmin, ymin, xmax, ymax, **props):
    base = {"occluded": False, "out-of-frame": False, "exterior": False, "ambiguous": False, "included": False,
            "direction-or-information": False, "highway": False, "dummy": False}
    base.update(props)
    return {"key": f"{label}-{xmin}", "label": label, "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "properties": base}


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "mtsd"
    _img(root / "images" / "k1.jpg")
    _anno(root / "annotations" / "k1.json", (H, W), [
        _obj("regulatory--stop--g1", 10, 10, 30, 30),
        _obj("other-sign", 40, 10, 50, 25),
        _obj("warning--curve-left--g1", 60, 10, 70, 20, dummy=True),
        _obj("regulatory--yield--g1", 80, 10, 80, 20),  # degenerate
    ])
    _img(root / "images" / "k2.jpg")
    _anno(root / "annotations" / "k2.json", (H, W), [_obj("information--parking--g1", 5, 5, 15, 15)])
    _img(root / "images" / "pano.jpg", (H, 300))
    _anno(root / "annotations" / "pano.json", (H, 300), [_obj("regulatory--stop--g1", 1, 1, 9, 9)], ispano=True)
    _img(root / "images" / "wide.jpg", (H, 200))
    _anno(root / "annotations" / "wide.json", (H, 200), [])
    _img(root / "images" / "k3.jpg")
    _anno(root / "annotations" / "k3.json", (H, W), [])
    (root / "splits").mkdir()
    (root / "splits" / "train.txt").write_text("k1\npano\nwide\nnolabel\nk3\n")
    (root / "splits" / "val.txt").write_text("k2\n")
    (root / "splits" / "test.txt").write_text("k9\n")
    return root


def test_convert(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "mtsd", "--", "--root", str(root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "panorama: 2" in text and "annotation missing: 1" in text
    assert "dummy sign: 1" in text and "degenerate box: 1" in text
    s = Sample.load(out / "train" / "samples" / "k1.npz")
    assert s.source == "mtsd" and s.licence == "research" and s.hw == (H, W) and s.calib == "nominal"
    assert s.el_boxes.tolist() == [[10, 10, 30, 30], [40, 10, 50, 25]]
    assert s.el_cls.tolist() == [EL_SIGN, EL_SIGN] and s.el_attr.tolist() == [-1, -1]
    assert s.el_sign.tolist() == ["mtsd:regulatory--stop--g1", ""]
    assert len(s.boxes2d) == 0
    lm = s.label_mask
    assert lm.elements and lm.el_sign and not lm.el_attr and not lm.agents and lm.el_vocab == (EL_SIGN,)
    assert s.K[0, 2] == (W - 1) / 2 and s.K[1, 2] == pytest.approx(0.5 * (H - 1)) and s.cam2ego[2, 3] == 1.5
    assert (out / "train" / "samples" / s.image).is_symlink()
    assert (out / "train" / "samples" / "k3.npz").exists()  # labelled frame without signs
    assert not (out / "train" / "samples" / "wide.npz").exists() and not (out / "train" / "samples" / "pano.npz").exists()
    v = Sample.load(out / "val" / "samples" / "k2.npz")
    assert v.el_sign.tolist() == ["mtsd:information--parking--g1"]
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 2, "val": 1}
    assert m["signs"] == ["mtsd:information--parking--g1", "mtsd:regulatory--stop--g1"]
    with pytest.raises(FileExistsError):
        main(["convert", "mtsd", "--", "--root", str(root), "--dst", str(out)])


def test_limit_and_options(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "mtsd", "--", "--root", str(root), "--dst", str(out), "--limit", "2", "--copy",
                 "--max-aspect", "4.0", "--cam-height", "2.0", "--splits", "train"]) == 0
    text = capsys.readouterr().out
    assert "2 frames" in text and "{'train': 2}" in text and "panorama: 1" in text  # ispano only; wide kept
    assert (out / "train" / "samples" / "wide.npz").exists() and not (out / "train" / "samples" / "k3.npz").exists()
    s = Sample.load(out / "train" / "samples" / "k1.npz")
    assert s.cam2ego[2, 3] == 2.0
    img = out / "train" / "samples" / s.image
    assert img.exists() and not img.is_symlink()
    assert not (out / "val").exists()
