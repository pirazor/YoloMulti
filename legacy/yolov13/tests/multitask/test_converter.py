"""End-to-end converter test: synthesize a tiny Supervisely dataset and convert.

We assert:
  - exactly one image is processed
  - one YOLO det line is written and decodes to roughly the right xywh
  - the DA mask has the expected count of '1' pixels in the polygon area
  - the LL mask has the expected non-zero pixels along the dashed line
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from yolov13_multitask.data.convert_supervisely import convert


def test_convert_tiny(tmp_path, tiny_supervisely):
    src = tiny_supervisely
    dst = tmp_path / "yolo_out"
    stats = convert(src=src, dst=dst, lane_grouping="style", val_fraction=0.0)

    assert stats.images == 1
    assert stats.boxes == 1
    assert stats.da_polys == 1
    assert stats.ll_polys == 1

    # data.yaml -----------------------------------------------------
    data = yaml.safe_load((dst / "data.yaml").read_text())
    assert data["nc"] == 9
    assert data["da_classes"] == 3
    assert data["ll_classes"] == 3
    assert data["names"][0] == "car"
    assert data["ll_names"] == ["background", "solid", "dashed"]

    # Detection label content --------------------------------------
    txt = (dst / "labels_det" / "train" / "frame.txt").read_text().strip()
    parts = txt.split()
    assert int(parts[0]) == 0  # car
    cx, cy, bw, bh = (float(x) for x in parts[1:5])
    # box was [200,100]-[400,200] in 1280x720 -> cx=300/1280, cy=150/720
    assert abs(cx - 300 / 1280) < 1e-4
    assert abs(cy - 150 / 720) < 1e-4
    assert abs(bw - 200 / 1280) < 1e-4
    assert abs(bh - 100 / 720) < 1e-4

    # DA mask: polygon was 500x200 px of value 1 -----------------
    da = cv2.imread(str(dst / "labels_da" / "train" / "frame.png"), cv2.IMREAD_GRAYSCALE)
    assert da is not None
    assert da.shape == (720, 1280)
    direct_area = int((da == 1).sum())
    # polygon was [100,500]-[600,500]-[600,700]-[100,700] => 500*200 = 100000
    assert 95_000 < direct_area < 105_000
    assert int((da == 2).sum()) == 0

    # LL mask: dashed line at class id 2, drawn with thickness ~ 4px
    ll = cv2.imread(str(dst / "labels_ll" / "train" / "frame.png"), cv2.IMREAD_GRAYSCALE)
    assert ll is not None
    assert int((ll == 2).sum()) > 100  # something drawn
    assert int((ll == 1).sum()) == 0
