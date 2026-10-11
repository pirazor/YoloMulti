"""Mini LISA traffic light dump in the Kaggle layout (Annotations/Annotations/<group>/..., <group>/<group>/.../frames)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.schema import EL_LIGHT, ELEMENT_ATTRS, Sample

H, W = 48, 64
HEADER = ("Filename;Annotation tag;Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;"
          "Origin file;Origin frame number;Origin track;Origin track frame number")


def _img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((H, W, 3), 40, np.uint8))


def _row(name, tag, x1, y1, x2, y2, n=0):
    return f"{name};{tag};{x1};{y1};{x2};{y2};dayTraining/{Path(name).parent.name}.avi;{n};{n};{n}"


def _csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([HEADER] + rows) + "\n")


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "lisa"
    for i in range(3):
        _img(root / "dayTrain" / "dayTrain" / "dayClip1" / "frames" / f"dayClip1--{i:05d}.jpg")
    _csv(root / "Annotations" / "Annotations" / "dayTrain" / "dayClip1" / "frameAnnotationsBOX.csv", [
        _row("dayTraining/dayClip1--00000.jpg", "go", 10, 5, 14, 15),
        _row("dayTraining/dayClip1--00000.jpg", "stopLeft", 20, 5, 24, 15),
        _row("dayTraining/dayClip1--00001.jpg", "warning", 30, 5, 34, 15, 1),
        _row("dayTraining/dayClip1--00001.jpg", "weird", 40, 5, 44, 15, 1),
        _row("dayTraining/dayClip1--00001.jpg", "go", 50, 5, 50, 15, 1),
        _row("dayTraining/dayClip1--00009.jpg", "go", 1, 1, 5, 5, 9),  # image missing
    ])
    _csv(root / "Annotations" / "Annotations" / "dayTrain" / "dayClip1" / "frameAnnotationsBULB.csv", [])  # ignored
    _img(root / "nightTrain" / "nightTrain" / "nightClip2" / "frames" / "nightClip2--00000.jpg")
    _csv(root / "Annotations" / "Annotations" / "nightTrain" / "nightClip2" / "frameAnnotationsBOX.csv", [
        _row("nightTraining/nightClip2--00000.jpg", "stop", 1, 1, 5, 9)])
    _img(root / "daySequence1" / "daySequence1" / "frames" / "daySequence1--00000.jpg")
    _csv(root / "Annotations" / "Annotations" / "daySequence1" / "frameAnnotationsBOX.csv", [
        _row("daySequence1/daySequence1--00000.jpg", "goForward", 2, 2, 6, 8)])
    return root


def test_convert(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "lisa_tl", "--", "--root", str(root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "4 frames" in text and "image missing: 1" in text
    assert "light arrow meaning dropped: 2" in text and "light tag weird: 1" in text and "degenerate box: 1" in text
    s = Sample.load(out / "train" / "samples" / "dayClip1--00000.npz")
    assert s.source == "lisa_tl" and s.hw == (H, W) and s.calib == "nominal" and s.cam2ego[2, 3] == 1.4
    assert s.el_boxes.tolist() == [[10, 5, 14, 15], [20, 5, 24, 15]] and s.el_cls.tolist() == [EL_LIGHT, EL_LIGHT]
    assert s.el_attr.tolist() == [ELEMENT_ATTRS.index("green"), ELEMENT_ATTRS.index("red")]
    assert s.clip_id == "dayTrain/dayClip1"
    lm = s.label_mask
    assert lm.elements and lm.el_attr and lm.el_vocab == (EL_LIGHT,) and not lm.agents
    t = Sample.load(out / "train" / "samples" / "dayClip1--00001.npz")
    assert t.el_attr.tolist() == [ELEMENT_ATTRS.index("yellow")]
    assert not (out / "train" / "samples" / "dayClip1--00002.npz").exists()  # no csv row -> not written by default
    n = Sample.load(out / "train" / "samples" / "nightClip2--00000.npz")
    assert n.el_attr.tolist() == [ELEMENT_ATTRS.index("red")] and n.clip_id == "nightTrain/nightClip2"
    v = Sample.load(out / "val" / "samples" / "daySequence1--00000.npz")
    assert v.el_attr.tolist() == [ELEMENT_ATTRS.index("green")] and v.clip_id == "daySequence1/daySequence1"
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 3, "val": 1}


def test_include_empty_limit_and_clip_override(root, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["convert", "lisa_tl", "--", "--root", str(root), "--dst", str(out), "--include-empty", "--splits",
                 "train", "--train-clips", "dayTrain", "--copy"]) == 0
    text = capsys.readouterr().out
    assert "3 frames" in text and "{'train': 3}" in text
    e = Sample.load(out / "train" / "samples" / "dayClip1--00002.npz")
    assert len(e.el_boxes) == 0 and e.label_mask.elements
    assert not (out / "train" / "samples" / e.image).is_symlink()
    assert main(["convert", "lisa_tl", "--", "--root", str(root), "--dst", str(out), "--overwrite", "--limit", "1",
                 "--val-clips", "nightTrain"]) == 0
    text = capsys.readouterr().out
    assert "2 frames" in text and "{'train': 1, 'val': 1}" in text
    assert (out / "val" / "samples" / "nightClip2--00000.npz").exists()
