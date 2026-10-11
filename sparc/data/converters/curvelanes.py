"""CurveLanes -> SPARC samples: lane polylines (no attributes) and a rasterised lane mask.

Expected layout under ``--root`` (the official download; confirmed against github.com/SoulmateB/CurveLanes):

* ``train/images/<stem>.jpg`` + ``train/labels/<stem>.lines.json``, likewise ``valid/``; ``test/images`` has no labels;
* each JSON: ``{"Lines": [[{"x": "2079.41", "y": "1439.0"}, ...], ...]}``: one list of key points per lane, numbers
  stored as strings, at least two points per lane; mixed resolutions (2560 x 1440, 1570 x 660, 1280 x 720, ...).

Assumptions:

* ``--splits train val`` map to the ``train`` / ``valid`` directories (``valid`` is accepted as well). The frames of
  a split are the label files (sorted); the image is ``images/<stem>.jpg`` (or ``.png``), missing ones are counted.
* Point order in the file is not trusted: points are sorted bottom (near) to top (far). Non-finite points are
  dropped; a lane left with < 2 points is skipped and counted.
* CurveLanes labels geometry only: labels ``lane_geom`` + ``lane_mask``, no order / direction / type.
* No calibration: a nominal camera (``--hfov``, ``--cam-height``, ``--horizon``), ``calib="nominal"``.
* Every labelled frame is fully annotated, so a frame without lanes is a true negative (written with an empty mask).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, image_size, nominal_camera,
                                        nominal_camera_args)
from sparc.data.raster import draw_polylines, line_thickness
from sparc.data.schema import LabelMask, LaneInstance

LOG = logging.getLogger("sparc.convert.curvelanes")

SPLIT_DIRS = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="CurveLanes root (train/, valid/ with images/ and labels/)")
    p.add_argument("--lane-px", type=float, default=8.0, help="lane line thickness at 1280 px width")
    add_common_args(p)
    nominal_camera_args(p, hfov=60.0, height=1.5, horizon=0.5)


def read_label(path: Path, stats: ConvertStats) -> List[np.ndarray]:
    """``<stem>.lines.json`` -> one (M, 2) float array per lane (near first), lanes with < 2 points dropped."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lanes = []
    for line in data.get("Lines", []):
        pts = []
        for pt in line:
            try:
                pts.append((float(pt["x"]), float(pt["y"])))
            except (KeyError, TypeError, ValueError):
                stats.skip("lane point unparsable")
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        finite = np.isfinite(p).all(1)
        if not finite.all():
            stats.skip("lane point not finite", int((~finite).sum()))
            p = p[finite]
        if len(p) < 2:
            stats.skip("lane < 2 points")
            continue
        lanes.append(p[np.argsort(-p[:, 1], kind="stable")])
    return lanes


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    w = Writer(args.dst, "curvelanes", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    lm = LabelMask(lane_geom=True, lane_mask=True)
    for split in args.splits:
        sub = root / SPLIT_DIRS.get(split, split)
        labels = sorted((sub / "labels").glob("*.lines.json")) if (sub / "labels").is_dir() else []
        if not labels:
            LOG.warning("%s: no labels in %s/labels (the test split has none)", split, sub)
            continue
        LOG.info("%s: %d label files in %s", split, len(labels), sub)
        count = 0
        for lab in labels:
            if args.limit is not None and count >= args.limit:
                break
            stem = lab.name[: -len(".lines.json")]
            candidates = (sub / "images" / f"{stem}{ext}" for ext in (".jpg", ".png", ".jpeg"))
            img = next((p for p in candidates if p.exists()), None)
            hw = image_size(img) if img is not None else None
            if hw is None:
                w.stats.skip_frame("image missing")
                continue
            try:
                lanes = [LaneInstance(p) for p in read_label(lab, w.stats)]
            except (ValueError, json.JSONDecodeError):
                w.stats.skip_frame("label unreadable")
                continue
            mask = np.zeros(hw, np.uint8)
            draw_polylines(mask, [l.points for l in lanes], 1, line_thickness(hw, args.lane_px))
            K, T, calib = nominal_camera(args, hw)
            if w.add(split, stem, img, lm, lane_mask=mask, hw=hw, K=K, cam2ego=T, calib=calib, lanes=lanes) is not None:
                count += 1
    return w.finish()
