"""LISA Traffic Light Dataset -> SPARC samples: traffic-light boxes with the colour.

Expected layout under ``--root`` (the Kaggle / original archive, unpacked)::

    Annotations/Annotations/<group>/<clip>/frameAnnotationsBOX.csv      e.g. dayTrain/dayClip1, nightTrain/nightClip3
    Annotations/Annotations/<group>/frameAnnotationsBOX.csv             e.g. daySequence1, nightSequence2
    <group>/<group>/<clip>/frames/<clip>--<nnnnn>.jpg                   1280 x 960
    <group>/<group>/frames/<group>--<nnnnn>.jpg                         sequences

(the single-level variants ``Annotations/<group>/...`` and ``<group>/<clip>/frames`` are found too: the csv files
are discovered with ``Annotations/**/<group>/**/frameAnnotationsBOX.csv`` and each clip's ``frames/`` directory with
``<group>/**/<clip>/frames``.)

CSV format (confirmed header): ``;`` separated, columns ``Filename;Annotation tag;Upper left corner X;Upper left
corner Y;Lower right corner X;Lower right corner Y;Origin file;Origin frame number;Origin track;Origin track frame
number``; ``Filename`` holds paths like ``dayTraining/dayClip1--00000.jpg`` whose directory does not exist on disk,
so the image is resolved by basename inside the clip's ``frames/`` directory. One row per light: rows are grouped
per image. Only ``frameAnnotationsBOX.csv`` (the whole light housing) is used, not ``frameAnnotationsBULB.csv``.

Tags (assumed from the published tag set): ``go`` -> green, ``stop`` -> red, ``warning`` -> yellow; ``goLeft``,
``goForward``, ``stopLeft``, ``warningLeft`` -> the colour, the arrow is dropped and counted
(``light arrow meaning dropped``); other tags are skipped and counted. Frames of a clip that have no csv row are
not written unless ``--include-empty`` is given (then they are labelled frames without lights; the csv lists only
frames with lights).

Splits: ``--splits train val`` -> ``--train-clips`` (default ``dayTrain nightTrain``) / ``--val-clips`` (default
``daySequence1 nightSequence1``; ``daySequence2`` / ``nightSequence2`` are left for testing). No calibration: nominal
camera (``--hfov``, ``--cam-height``, ``--horizon``), ``calib="nominal"``. Label mask: ``elements`` + ``el_attr``
over ``el_vocab = (traffic_light,)``. Licence: CC BY-NC-SA 4.0 (``research``).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, image_size, nominal_camera,
                                        nominal_camera_args)
from sparc.data.schema import EL_LIGHT, ELEMENT_ATTRS, LabelMask

LOG = logging.getLogger("sparc.convert.lisa_tl")

COLOURS = (("go", "green"), ("stop", "red"), ("warning", "yellow"))
COL_FILE, COL_TAG = "Filename", "Annotation tag"
COL_BOX = ("Upper left corner X", "Upper left corner Y", "Lower right corner X", "Lower right corner Y")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="LISA root (Annotations/, dayTrain/, nightTrain/, daySequence1/ ...)")
    p.add_argument("--train-clips", nargs="+", default=["dayTrain", "nightTrain"])
    p.add_argument("--val-clips", nargs="+", default=["daySequence1", "nightSequence1"])
    p.add_argument("--include-empty", action="store_true", help="also write frames without any csv row")
    add_common_args(p)
    nominal_camera_args(p, hfov=55.0, height=1.4, horizon=0.5)


def _csvs(root: Path, group: str) -> List[Path]:
    return sorted(set(root.glob(f"Annotations/**/{group}/**/frameAnnotationsBOX.csv")))


def _frames_dir(root: Path, group: str, clip: str) -> Optional[Path]:
    cands = sorted(p for p in root.glob(f"{group}/**/{clip}/frames") if p.is_dir())
    if not cands and clip == group:
        cands = sorted(p for p in root.glob(f"{group}/**/frames") if p.is_dir())
    return cands[0] if cands else None


def _colour(tag: str, stats: ConvertStats) -> Optional[int]:
    key = str(tag).strip().lower()
    for prefix, attr in COLOURS:
        if key.startswith(prefix):
            if key != prefix:
                stats.skip("light arrow meaning dropped")
            return ELEMENT_ATTRS.index(attr)
    stats.skip(f"light tag {tag}")
    return None


def _rows(path: Path) -> Dict[str, List[dict]]:
    """csv rows grouped by the image basename."""
    out: Dict[str, List[dict]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = Path(str(row.get(COL_FILE, "")).strip()).name
            if name:
                out.setdefault(name, []).append(row)
    return out


def _frame(rows: List[dict], stats: ConvertStats):
    boxes, attrs = [], []
    for row in rows:
        try:
            xyxy = [float(row[c]) for c in COL_BOX]
        except (KeyError, TypeError, ValueError):
            stats.skip("row without box")
            continue
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            stats.skip("degenerate box")
            continue
        attr = _colour(row.get(COL_TAG, ""), stats)
        if attr is None:
            continue
        boxes.append(xyxy)
        attrs.append(attr)
    return np.array(boxes, np.float32).reshape(-1, 4), attrs


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    w = Writer(args.dst, "lisa_tl", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    lm = LabelMask(elements=True, el_vocab=(EL_LIGHT,), el_attr=True)
    groups = {"train": args.train_clips, "val": args.val_clips}
    for split in args.splits:
        count = 0
        for group in groups.get(split, []):
            csvs = _csvs(root, group)
            if not csvs:
                LOG.warning("%s: no frameAnnotationsBOX.csv for %s", root, group)
                continue
            for csv_path in csvs:
                clip = csv_path.parent.name
                frames = _frames_dir(root, group, clip)
                if frames is None:
                    w.stats.skip_frame(f"frames dir missing for {group}/{clip}")
                    continue
                rows = _rows(csv_path)
                names = sorted(rows)
                if args.include_empty:
                    names = sorted(set(names) | {p.name for p in frames.iterdir() if p.suffix.lower() in (".jpg", ".png")})
                for name in names:
                    if args.limit is not None and count >= args.limit:
                        break
                    img = frames / name
                    hw = image_size(img)
                    if hw is None:
                        w.stats.skip_frame("image missing")
                        continue
                    el_boxes, el_attr = _frame(rows.get(name, []), w.stats)
                    K, T, calib = nominal_camera(args, hw)
                    if w.add(split, Path(name).stem, img, hw=hw, label_mask=lm, K=K, cam2ego=T, calib=calib,
                             el_boxes=el_boxes, el_cls=[EL_LIGHT] * len(el_boxes), el_attr=el_attr,
                             clip_id=f"{group}/{clip}") is not None:
                        count += 1
    return w.finish()
