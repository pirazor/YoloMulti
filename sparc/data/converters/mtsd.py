"""MTSD (Mapillary Traffic Sign Dataset) -> SPARC samples: traffic-sign boxes with the fine sign class.

Expected layout under ``--root`` (the official download, unpacked; the fully annotated set)::

    images/<key>.jpg
    annotations/<key>.json
    splits/train.txt  splits/val.txt  splits/test.txt      one image key per line

One annotation file per image::

    {"width": W, "height": H, "ispano": false,
     "objects": [{"key": "...", "label": "regulatory--stop--g1" | "other-sign",
                  "bbox": {"xmin", "ymin", "xmax", "ymax"},
                  "properties": {"occluded", "out-of-frame", "exterior", "ambiguous", "included",
                                 "direction-or-information", "highway", "dummy", ...}}]}

Format assumptions (the official download is behind a registration wall; the keys ``width`` / ``height`` / ``objects``
/ ``label`` / ``bbox`` and the panorama flag ``ispano`` are those described by a mirror of the v2 fully-annotated
README and by the paper, arXiv:1909.04422; ``ismetric``, named in the task brief, was not found anywhere and is
ignored):

* signs go to the element table with ``el_cls = traffic_sign`` and ``el_sign = "mtsd:<label>"``; ``other-sign``
  (a sign outside the taxonomy) keeps its box with ``el_sign = ""`` (unlabelled fine class);
* ``properties.dummy`` (sign look-alikes: stickers, reflections) are skipped and counted; every other property
  (occluded, ambiguous, out-of-frame, ...) is kept as a normal box;
* panoramas (``ispano`` true, or width > ``--max-aspect`` x height) are skipped and counted: their boxes may cross
  the image boundary and a pinhole ``K`` is meaningless for them;
* the published ``test`` split has no annotation files: its keys are counted as ``annotation missing``;
* the annotated ``width`` / ``height`` must match the image file, otherwise the frame is counted and skipped.

No calibration is shipped: a nominal camera (``--hfov``, ``--cam-height``, ``--horizon``) is used and the sample is
marked ``calib="nominal"``. Label mask: ``elements`` + ``el_sign`` over ``el_vocab = (traffic_sign,)``; there are no
road-user or light labels. Licence: MTSD is released for non-commercial research use (``research``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, image_size, nominal_camera,
                                        nominal_camera_args)
from sparc.data.schema import EL_SIGN, LabelMask

LOG = logging.getLogger("sparc.convert.mtsd")

UNLABELLED = {"other-sign", ""}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="MTSD root (images/, annotations/, splits/)")
    p.add_argument("--max-aspect", type=float, default=2.5, help="frames wider than this x height are panoramas")
    add_common_args(p)
    nominal_camera_args(p, hfov=70.0, height=1.5, horizon=0.5)


def _keys(root: Path, split: str) -> Optional[List[str]]:
    p = root / "splits" / f"{split}.txt"
    if not p.exists():
        return None
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _image(root: Path, key: str) -> Optional[Path]:
    for suf in IMAGE_SUFFIXES:
        p = root / "images" / (key + suf)
        if p.exists():
            return p
    return None


def _frame(anno: dict, hw, args, stats: ConvertStats) -> Optional[dict]:
    w_anno, h_anno = anno.get("width"), anno.get("height")
    if w_anno is not None and h_anno is not None and (int(h_anno), int(w_anno)) != tuple(hw):
        stats.skip_frame("annotation size mismatch")
        return None
    if anno.get("ispano") or hw[1] > args.max_aspect * hw[0]:
        stats.skip_frame("panorama")
        return None
    boxes, names = [], []
    for obj in anno.get("objects", []):
        props = obj.get("properties", {}) or {}
        if props.get("dummy"):
            stats.skip("dummy sign")
            continue
        b = obj.get("bbox")
        if b is None:
            stats.skip("object without bbox")
            continue
        xyxy = [float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            stats.skip("degenerate box")
            continue
        label = str(obj.get("label", "")).strip()
        boxes.append(xyxy)
        names.append("" if label in UNLABELLED else f"mtsd:{label}")
    lm = LabelMask(elements=True, el_vocab=(EL_SIGN,), el_sign=True)
    return dict(label_mask=lm, el_boxes=np.array(boxes, np.float32).reshape(-1, 4), el_cls=[EL_SIGN] * len(boxes),
                el_attr=[-1] * len(boxes), el_sign=names)


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    w = Writer(args.dst, "mtsd", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    for split in args.splits:
        keys = _keys(root, split)
        if keys is None:
            LOG.warning("%s: no split file splits/%s.txt", root, split)
            continue
        count = 0
        for key in keys:
            if args.limit is not None and count >= args.limit:
                break
            anno_path = root / "annotations" / f"{key}.json"
            if not anno_path.exists():
                w.stats.skip_frame("annotation missing")
                continue
            img = _image(root, key)
            hw = image_size(img) if img is not None else None
            if hw is None:
                w.stats.skip_frame("image missing")
                continue
            with open(anno_path, "r", encoding="utf-8") as f:
                anno = json.load(f)
            fields = _frame(anno, hw, args, w.stats)
            if fields is None:
                continue
            K, T, calib = nominal_camera(args, hw)
            fields.update(K=K, cam2ego=T, calib=calib)
            if w.add(split, key, img, hw=hw, **fields) is not None:
                count += 1
    return w.finish()
