"""Mini DTLD dump: 16-bit Bayer-GB TIFFs with a known colour, v2.0 json labels, OpenCV-style intrinsic yml."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sparc.cli import main
from sparc.data.schema import EL_LIGHT, ELEMENT_ATTRS, Sample

H, W = 32, 64
BGR = (40, 120, 200)
INTRINSIC_YML = """%YAML:1.0
intrisicData: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 2290.5, 0., 1066.9, 0., 2290.5, 477.1, 0., 0.,
       1. ]
"""


def _bayer_tiff(path: Path, bgr=BGR):
    """12-bit Bayer mosaic in a 16-bit TIFF, laid out as OpenCV's COLOR_BAYER_GB expects: [[G, R], [B, G]]."""
    b, g, r = bgr
    cell = [[g, r], [b, g]]
    m = np.zeros((H, W), np.uint16)
    for y in range(H):
        for x in range(W):
            m[y, x] = cell[y % 2][x % 2] << 4
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), m)


def _label(x, y, w, h, state, direction="front", pictogram="circle"):
    return {"x": x, "y": y, "w": w, "h": h, "unique_id": 1, "track_id": "t",
            "attributes": {"direction": direction, "relevance": "relevant", "occlusion": "not_occluded",
                           "orientation": "vertical", "aspects": "three_aspects", "state": state,
                           "pictogram": pictogram, "reflection": "not_reflected"}}


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "DTLD"
    _bayer_tiff(root / "Berlin" / "Berlin1" / "2015-04-17_10-50-05" / "DE_BBBR667_2015-04-17_10-50-13-633939_k0.tiff")
    _bayer_tiff(root / "Bochum" / "Bochum1" / "2015-04-13_11-51-05" / "DE_BBBR667_2015-04-13_11-51-44-125133_k0.tiff")
    labels = root / "DTLD_labels_v2.0" / "v2.0"
    labels.mkdir(parents=True)
    train = {"images": [{"image_path": "Berlin/Berlin1/2015-04-17_10-50-05/DE_BBBR667_2015-04-17_10-50-13-633939_k0.tiff",
                         "disparity_image_path": "Berlin/Berlin1/2015-04-17_10-50-05/DE_BBBR667_2015-04-17_10-50-13-633939_nativeV2.tiff",
                         "time_stamp": 1.0,
                         "labels": [_label(10, 2, 4, 10, "red"), _label(20, 2, 4, 10, "green", pictogram="arrow_left"),
                                    _label(30, 2, 4, 10, "red_yellow", direction="back"), _label(40, 2, 4, 10, "off"),
                                    _label(50, 2, 4, 10, "unknown", pictogram="unknown"),
                                    _label(55, 2, 4, 10, "yellow", pictogram="pedestrian"),
                                    _label(60, 2, 0, 10, "red")]},
                        {"image_path": "/old/absolute/DTLD/Nowhere/x_k0.tiff", "disparity_image_path": "", "time_stamp": 2.0,
                         "labels": []}]}
    test = {"images": [{"image_path": "/data/DTLD/Bochum/Bochum1/2015-04-13_11-51-05/DE_BBBR667_2015-04-13_11-51-44-125133_k0.tiff",
                        "disparity_image_path": "", "time_stamp": 3.0, "labels": [_label(1, 1, 3, 6, "red_yellow")]}]}
    (labels / "DTLD_train.json").write_text(json.dumps(train))
    (labels / "DTLD_test.json").write_text(json.dumps(test))
    return root


def test_convert_with_calibration(root, tmp_path, capsys):
    (root / "calibration").mkdir()
    (root / "calibration" / "intrinsic_left.yml").write_text(INTRINSIC_YML)
    out = tmp_path / "out"
    assert main(["convert", "dtld", "--", "--root", str(root), "--dst", str(out)]) == 0
    text = capsys.readouterr().out
    assert "2 frames" in text and "image missing: 1" in text
    assert "light facing back: 1" in text and "light arrow meaning dropped: 1" in text
    assert "light pictogram pedestrian dropped: 1" in text and "degenerate box: 1" in text
    s = Sample.load(out / "train" / "samples" / "DE_BBBR667_2015-04-17_10-50-13-633939_k0.npz")
    assert s.source == "dtld" and s.hw == (H, W) and s.calib == "measured"
    assert s.K[0, 0] == 2290.5 and s.K[0, 2] == 1066.9 and s.K[1, 2] == 477.1 and s.cam2ego[2, 3] == 1.4
    img = out / "train" / "samples" / s.image
    assert img.suffix == ".jpg" and img.exists() and not img.is_symlink() and img.parent.name == "images"
    decoded = cv2.imread(str(img))
    assert decoded.shape == (H, W, 3) and decoded.dtype == np.uint8
    assert np.abs(decoded.reshape(-1, 3).mean(0) - np.array(BGR)).max() < 4
    assert s.el_boxes.tolist() == [[10, 2, 14, 12], [20, 2, 24, 12], [40, 2, 44, 12], [50, 2, 54, 12], [55, 2, 59, 12]]
    assert s.el_cls.tolist() == [EL_LIGHT] * 5
    assert s.el_attr.tolist() == [ELEMENT_ATTRS.index("red"), ELEMENT_ATTRS.index("green"), 0, -1, ELEMENT_ATTRS.index("yellow")]
    lm = s.label_mask
    assert lm.elements and lm.el_attr and lm.el_vocab == (EL_LIGHT,) and not lm.agents and not lm.el_sign
    v = Sample.load(out / "val" / "samples" / "DE_BBBR667_2015-04-13_11-51-44-125133_k0.npz")
    assert v.el_attr.tolist() == [ELEMENT_ATTRS.index("red")] and v.el_boxes.tolist() == [[1, 1, 4, 7]]
    assert not (out / ".decode").exists() and not any(p.suffix == ".tiff" for p in out.rglob("*"))
    m = json.loads((out / "manifest.json").read_text())
    assert m["splits"] == {"train": 1, "val": 1}


def test_nominal_fallback_limit_and_explicit_labels(root, tmp_path, capsys):
    out = tmp_path / "out"
    labels = str(root / "DTLD_labels_v2.0" / "v2.0" / "DTLD_{split}.json")
    assert main(["convert", "dtld", "--", "--root", str(root), "--dst", str(out), "--labels", labels, "--limit", "1",
                 "--splits", "train", "--hfov", "40", "--jpeg-quality", "80"]) == 0
    text = capsys.readouterr().out
    assert "1 frames" in text and "{'train': 1}" in text
    s = Sample.load(out / "train" / "samples" / "DE_BBBR667_2015-04-17_10-50-13-633939_k0.npz")
    assert s.calib == "nominal" and s.K[0, 2] == (W - 1) / 2
    assert s.K[0, 0] == pytest.approx((W / 2) / np.tan(np.radians(20)))
    assert not (out / "val").exists()
    # v1 labels (no attributes dict) are refused with a clear message
    v1 = root / "DTLD_labels_v1.0"
    v1.mkdir()
    (v1 / "DTLD_train.yml").write_text("images:\n- image_path: Berlin/x_k0.tiff\n  objects: []\n")
    with pytest.raises(ValueError, match="v1"):
        main(["convert", "dtld", "--", "--root", str(root), "--dst", str(out), "--labels", str(v1 / "DTLD_{split}.yml"),
              "--overwrite", "--splits", "train"])
