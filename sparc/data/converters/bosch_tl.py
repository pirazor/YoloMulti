"""Bosch Small Traffic Lights Dataset -> SPARC samples: traffic-light boxes with the colour.

Expected layout under ``--root`` (the download, unpacked; the yaml files next to ``rgb/``)::

    train.yaml   test.yaml  (``--train-yaml`` / ``--val-yaml`` override; the updated files from
                             github.com/bosch-ros-pkg/bstld/label_files work the same way)
    rgb/train/<bag>/<frame>.png      1280 x 720
    rgb/test/<frame>.png

Each yaml is a list of frames (confirmed from ``bstld/read_label_file.py`` and ``bosch_to_pascal.py``)::

    - boxes:
      - {label: Green, occluded: false, x_max: 1052.0, x_min: 1042.0, y_max: 371.0, y_min: 348.0}
      path: ./rgb/train/2015-10-05-10-52-01_bag/24088.png

Paths are relative to the yaml's directory (``./`` prefix); the original ``test.yaml`` holds absolute paths of the
authors' file server, so a path that does not exist is retried relative to the yaml directory, then as
``rgb/<split>/<basename>``, then by basename anywhere under ``rgb/`` (counted as ``image missing`` when all fail).
A frame with an empty ``boxes`` list is a labelled frame without lights.

Labels (assumed from the published label set, not a spec): ``Red``, ``Yellow``, ``Green``, ``off`` and the arrow
variants ``RedLeft``, ``RedRight``, ``RedStraight``, ``RedStraightLeft``, ``GreenLeft``, ``GreenRight``,
``GreenStraight``, ``GreenStraightLeft``, ``GreenStraightRight``, ... The colour is the label's prefix (case
insensitive) -> ``el_attr``; ``off`` -> unknown (0); the arrow suffix is dropped and counted
(``light arrow meaning dropped``); any other label is skipped and counted. ``occluded`` is kept.

``--splits train val`` -> ``train.yaml`` / ``test.yaml``. No calibration: nominal camera (``--hfov``,
``--cam-height``, ``--horizon``), ``calib="nominal"``. Label mask: ``elements`` + ``el_attr`` over
``el_vocab = (traffic_light,)``. Licence: the dataset ships a non-commercial licence (``research``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, image_size, nominal_camera,
                                        nominal_camera_args)
from sparc.data.schema import EL_LIGHT, ELEMENT_ATTRS, LabelMask

LOG = logging.getLogger("sparc.convert.bosch_tl")

SPLIT_YAML = {"train": "train.yaml", "val": "test.yaml", "test": "test.yaml"}
SPLIT_DIR = {"train": "train", "val": "test", "test": "test"}
COLOURS = (("red", "red"), ("green", "green"), ("yellow", "yellow"), ("off", "unknown"))


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="dataset root (train.yaml, test.yaml, rgb/)")
    p.add_argument("--train-yaml", default=None)
    p.add_argument("--val-yaml", default=None)
    add_common_args(p)
    nominal_camera_args(p, hfov=55.0, height=1.4, horizon=0.5)


class _Index:
    """Lazy basename -> path index of every png under <root>/rgb (for the absolute paths of test.yaml)."""

    def __init__(self, root: Path):
        self.root = root
        self._map: Optional[Dict[str, List[Path]]] = None

    def find(self, name: str) -> Optional[Path]:
        if self._map is None:
            self._map = {}
            base = self.root / "rgb" if (self.root / "rgb").exists() else self.root
            for p in base.rglob("*.png"):
                self._map.setdefault(p.name, []).append(p)
        hits = self._map.get(name, [])
        return sorted(hits)[0] if hits else None


def _resolve(raw: str, yaml_dir: Path, split: str, index: _Index) -> Optional[Path]:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for cand in (yaml_dir / raw, yaml_dir / "rgb" / SPLIT_DIR.get(split, split) / p.name):
        if cand.exists():
            return cand
    return index.find(p.name)


def _colour(label: str, stats: ConvertStats) -> Optional[int]:
    key = str(label).strip().lower()
    for prefix, attr in COLOURS:
        if key.startswith(prefix):
            if key != prefix:
                stats.skip("light arrow meaning dropped")
            return ELEMENT_ATTRS.index(attr)
    stats.skip(f"light label {label}")
    return None


def _frame(rec: dict, stats: ConvertStats):
    boxes, attrs = [], []
    for b in rec.get("boxes", []) or []:
        xyxy = [float(b["x_min"]), float(b["y_min"]), float(b["x_max"]), float(b["y_max"])]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            stats.skip("degenerate box")
            continue
        attr = _colour(b.get("label", ""), stats)
        if attr is None:
            continue
        boxes.append(xyxy)
        attrs.append(attr)
    return np.array(boxes, np.float32).reshape(-1, 4), attrs


def convert(args: argparse.Namespace) -> ConvertStats:
    import yaml

    root = Path(args.root)
    w = Writer(args.dst, "bosch_tl", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    index = _Index(root)
    lm = LabelMask(elements=True, el_vocab=(EL_LIGHT,), el_attr=True)
    for split in args.splits:
        explicit = {"train": args.train_yaml, "val": args.val_yaml}.get(split)
        yml = Path(explicit) if explicit else root / SPLIT_YAML.get(split, f"{split}.yaml")
        if not yml.exists():
            LOG.warning("%s: no yaml for split %s", root, split)
            continue
        with open(yml, "r", encoding="utf-8") as f:
            records = yaml.safe_load(f) or []
        count = 0
        for rec in records:
            if args.limit is not None and count >= args.limit:
                break
            img = _resolve(str(rec.get("path", "")), yml.parent, split, index)
            hw = image_size(img) if img is not None else None
            if hw is None:
                w.stats.skip_frame("image missing")
                continue
            el_boxes, el_attr = _frame(rec, w.stats)
            K, T, calib = nominal_camera(args, hw)
            stem = f"{img.parent.name}_{img.stem}"
            if w.add(split, stem, img, hw=hw, label_mask=lm, K=K, cam2ego=T, calib=calib, el_boxes=el_boxes,
                     el_cls=[EL_LIGHT] * len(el_boxes), el_attr=el_attr) is not None:
                count += 1
    return w.finish()
