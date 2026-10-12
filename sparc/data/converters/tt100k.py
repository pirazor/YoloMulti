"""TT100K (Tsinghua-Tencent 100K) -> SPARC samples: traffic-sign boxes with the fine sign class.

Expected layout under ``--root`` (the ``data`` directory of the download)::

    annotations.json          2016 release        (or)   annotations_all.json   2021 release (``tt100k_2021``)
    train/<id>.jpg  test/<id>.jpg  other/<id>.jpg           2048 x 2048 images

The annotation file (``--annotations`` overrides discovery) is one JSON object::

    {"types": ["pn", "pne", ...],
     "imgs": {"<id>": {"id": <id>, "path": "train/<id>.jpg",
                       "objects": [{"category": "pn", "bbox": {"xmin", "ymin", "xmax", "ymax"},
                                    "ellipse_org": [...], "ellipse": [...], "polygon": [...]}]}}}

Format facts and assumptions:

* ``imgs`` / ``path`` / ``objects`` / ``category`` / ``bbox{xmin,ymin,xmax,ymax}`` and the optional ``ellipse_org`` /
  ``polygon`` shape keys are the ones read by the official tutorial code and by third-party converters of both the
  2016 and the 2021 release (e.g. the "TT100K ver. 2021 to COCO" gist); the 2021 release renames the file to
  ``annotations_all.json`` and keeps the per-image structure (more images, 232 categories). Both names are tried.
* The split is the first component of ``path``: ``train`` -> the first of ``--splits``, ``test`` -> the second
  (default ``train val``); ``other`` (images without annotated signs of the 45 benchmark classes) and any unknown
  prefix are skipped and counted. The 2021 release's own ``train`` / ``test`` lists are not used.
* Boxes may fall slightly outside the image (``ymin: -4`` occurs in the 2021 file): they are clipped and counted;
  boxes empty after clipping are counted as degenerate.
* ``el_sign = "tt100k:<category>"`` for every object (TT100K has no "unknown sign" category); the image size is
  read from the file, not assumed.

No calibration: nominal camera (``--hfov``, ``--cam-height``, ``--horizon``), ``calib="nominal"``. Label mask:
``elements`` + ``el_sign`` over ``el_vocab = (traffic_sign,)``. Licence: research use (``research``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, image_size, nominal_camera,
                                        nominal_camera_args)
from sparc.data.schema import EL_SIGN, LabelMask

LOG = logging.getLogger("sparc.convert.tt100k")

ANNOTATION_FILES = ("annotations_all.json", "annotations.json")
PREFIX_ORDER = ("train", "test")  # path prefix -> index into --splits


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="TT100K data root (annotations[_all].json, train/, test/, other/)")
    p.add_argument("--annotations", default=None, help="annotation json (overrides discovery)")
    add_common_args(p)
    nominal_camera_args(p, hfov=60.0, height=1.4, horizon=0.5)


def _find_annotations(root: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for base in (root, root / "data"):
        for name in ANNOTATION_FILES:
            if (base / name).exists():
                return base / name
    return None


def _frame(rec: dict, hw, stats: ConvertStats) -> dict:
    h, w = hw
    boxes, names = [], []
    for obj in rec.get("objects", []):
        b = obj.get("bbox")
        if b is None:
            stats.skip("object without bbox")
            continue
        raw = [float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])]
        xyxy = [min(max(raw[0], 0.0), w), min(max(raw[1], 0.0), h), min(max(raw[2], 0.0), w), min(max(raw[3], 0.0), h)]
        if xyxy != raw:
            stats.skip("box clipped")
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            stats.skip("degenerate box")
            continue
        cat = str(obj.get("category", "")).strip()
        if not cat:
            stats.skip("object without category")
            continue
        boxes.append(xyxy)
        names.append(f"tt100k:{cat}")
    lm = LabelMask(elements=True, el_vocab=(EL_SIGN,), el_sign=True)
    return dict(label_mask=lm, el_boxes=np.array(boxes, np.float32).reshape(-1, 4), el_cls=[EL_SIGN] * len(boxes),
                el_attr=[-1] * len(boxes), el_sign=names)


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    anno_path = _find_annotations(root, args.annotations)
    if anno_path is None:
        raise FileNotFoundError(f"no {' / '.join(ANNOTATION_FILES)} under {root}")
    LOG.info("annotations: %s", anno_path)
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    imgs: Dict[str, dict] = data.get("imgs", {})
    prefix_to_split = {pre: args.splits[i] for i, pre in enumerate(PREFIX_ORDER) if i < len(args.splits)}
    base = anno_path.parent
    w = Writer(args.dst, "tt100k", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    counts: Dict[str, int] = {}
    for img_id in sorted(imgs, key=lambda k: (len(k), k)):
        rec = imgs[img_id]
        rel = str(rec.get("path", ""))
        prefix = rel.split("/", 1)[0] if "/" in rel else ""
        split = prefix_to_split.get(prefix)
        if split is None:
            w.stats.skip_frame(f"{prefix or 'unknown'} split")
            continue
        if args.limit is not None and counts.get(split, 0) >= args.limit:
            continue
        img = base / rel
        hw = image_size(img)
        if hw is None:
            w.stats.skip_frame("image missing")
            continue
        fields = _frame(rec, hw, w.stats)
        K, T, calib = nominal_camera(args, hw)
        fields.update(K=K, cam2ego=T, calib=calib)
        if w.add(split, str(rec.get("id", img_id)), img, hw=hw, **fields) is not None:
            counts[split] = counts.get(split, 0) + 1
    return w.finish({"types": data.get("types", [])})
