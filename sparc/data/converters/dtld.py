"""DTLD (DriveU Traffic Light Dataset) -> SPARC samples: traffic-light boxes with the colour state, decoded from the
16-bit Bayer TIFFs into 8-bit JPEGs.

Expected layout under ``--root`` (the download, see github.com/julimueller/dtld_parsing)::

    <city>/<route>/<sequence>/DE_*_k0.tiff          left camera, 2048 x 1024, 12-bit Bayer in a 16-bit TIFF
    <city>/<route>/<sequence>/DE_*_nativeV2.tiff    disparity (unused)
    DTLD_labels_v2.0/**/DTLD_<train|test>.json      v2.0 labels (also per-city files; ``--labels`` overrides)
    calibration/intrinsic_left.yml                  optional (``--calib-dir``; also calibration/<city>/)

Confirmed from ``dtld_parsing/python/dtld_parsing/driveu_dataset.py`` (v2.0 JSON)::

    {"images": [{"image_path": "<city>/.../DE_..._k0.tiff", "disparity_image_path": "...", "time_stamp": ...,
                 "labels": [{"x", "y", "w", "h", "unique_id", "track_id",
                             "attributes": {"direction": front|back|left|right,
                                            "relevance": relevant|not_relevant,
                                            "occlusion": ..., "orientation": ..., "aspects": ...,
                                            "state": red|green|yellow|red_yellow|off|unknown,
                                            "pictogram": circle|arrow_left|arrow_right|arrow_straight|tram|
                                                         pedestrian|bicycle|unknown,
                                            "reflection": ...}}]}]}

and the image decoding ``cv2.imread(path, cv2.IMREAD_UNCHANGED)`` -> ``cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)`` ->
``np.right_shift(img, 4)`` (12-bit -> 8-bit). The parser rejects the v1 ``.yml`` files ("Yaml support is deprecated"):
a ``.yml`` label file is accepted here only when it carries the same v2 keys (``images`` / ``labels`` /
``attributes``); the v1 6-digit ``class_id`` encoding is not decoded. Label file names: the v2.0 archive was not
inspected, so ``DTLD_<split>.json`` and ``*_<split>.json`` are discovered recursively (``val`` maps to DTLD's
``test``); ``image_path`` is resolved against ``--root`` (the leading path components are dropped one by one until
the file is found, so an archive unpacked one level deeper still works).

Output: the decoded JPEG is written straight into ``<dst>/<split>/images/<stem>.jpg`` (``--copy`` is implied), so no
sample references the TIFFs. Lights not facing the camera (``direction`` != front) are skipped and counted; the
attribute is the colour only (``red_yellow`` -> red, ``off`` -> unknown (0), ``unknown`` -> -1 unlabelled);
pictograms (arrows, tram, pedestrian, bicycle) are dropped and counted; ``relevance`` and ``occlusion`` are kept.

Calibration: ``calibration/intrinsic_left.yml`` is an OpenCV YAML matrix (``rows`` / ``cols`` / ``data``, the key is
spelt ``intrisicData`` in dtld_parsing) and is used as ``K`` (``calib="measured"``, valid for the 2048 x 1024
rectified left images); without it a nominal camera (``--hfov``, ``--cam-height``, ``--horizon``) is used. The mount
(``cam2ego``) is nominal either way (``--cam-height``): the shipped extrinsics are not interpreted. Label mask:
``elements`` + ``el_attr`` over ``el_vocab = (traffic_light,)``. Licence: non-commercial research (``research``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, nominal_camera,
                                        nominal_camera_args)
from sparc.data.schema import EL_LIGHT, ELEMENT_ATTRS, LabelMask, ground_cam2ego

LOG = logging.getLogger("sparc.convert.dtld")

SPLIT_NAME = {"train": "train", "val": "test", "test": "test"}
STATE_ATTR = {"red": "red", "green": "green", "yellow": "yellow", "red_yellow": "red", "off": "unknown"}
PLAIN_PICTOGRAMS = {"circle", "unknown", ""}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="DTLD root (city folders with the tiffs)")
    p.add_argument("--labels", default=None, help="label json, {split} placeholder (overrides discovery)")
    p.add_argument("--calib-dir", default=None, help="directory holding intrinsic_left.yml (default: discovery)")
    p.add_argument("--jpeg-quality", type=int, default=95)
    add_common_args(p)
    nominal_camera_args(p, hfov=45.0, height=1.4, horizon=0.5)


def _label_files(root: Path, split: str, explicit: Optional[str]) -> List[Path]:
    name = SPLIT_NAME.get(split, split)
    if explicit:
        p = Path(explicit.format(split=name))
        return [p] if p.exists() else []
    found = sorted(root.rglob(f"DTLD_{name}.json")) or sorted(root.rglob(f"DTLD_{name}.yml"))
    if not found:
        found = sorted(p for p in root.rglob(f"*_{name}.json")) or sorted(root.rglob(f"*_{name}.yml"))
    return found


def _load_labels(path: Path) -> List[dict]:
    if path.suffix.lower() in (".yml", ".yaml"):
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    images = data.get("images") if isinstance(data, dict) else data
    if not isinstance(images, list):
        raise ValueError(f"{path}: expected {{'images': [...]}} (DTLD v2.0 labels)")
    if images and "labels" not in images[0]:
        raise ValueError(f"{path}: entries carry no 'labels' key; the v1 class_id format is not supported, "
                         "use DTLD_labels_v2.0")
    return images


def _resolve(root: Path, image_path: str) -> Optional[Path]:
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return p
    parts = p.parts[1:] if p.is_absolute() else p.parts
    for i in range(len(parts)):
        cand = root.joinpath(*parts[i:])
        if cand.exists():
            return cand
    return None


def _read_opencv_matrix(path: Path) -> Optional[np.ndarray]:
    """OpenCV YAML matrix (``rows`` / ``cols`` / ``data``) without caring about the node name or the header."""
    text = path.read_text(encoding="utf-8")
    rows = re.search(r"rows:\s*(\d+)", text)
    cols = re.search(r"cols:\s*(\d+)", text)
    data = re.search(r"data:\s*\[([^\]]*)\]", text, re.S)
    if not (rows and cols and data):
        return None
    vals = [float(v) for v in re.split(r"[,\s]+", data.group(1).strip()) if v]
    r, c = int(rows.group(1)), int(cols.group(1))
    if len(vals) != r * c:
        return None
    return np.asarray(vals, dtype=np.float64).reshape(r, c)


def _intrinsics(root: Path, explicit: Optional[str], label_file: Path, city: str) -> Optional[np.ndarray]:
    cands = []
    if explicit:
        cands.append(Path(explicit) / "intrinsic_left.yml")
    else:
        cands += [root / "calibration" / "intrinsic_left.yml", root / "calibration" / city / "intrinsic_left.yml",
                  label_file.parent / "calibration" / "intrinsic_left.yml",
                  label_file.parent.parent / "calibration" / "intrinsic_left.yml"]
    for c in cands:
        if c.exists():
            m = _read_opencv_matrix(c)
            if m is not None and m.shape == (3, 3) and m[0, 0] > 0 and m[1, 1] > 0:
                K = m.copy()
                K[2] = (0.0, 0.0, 1.0)
                return K
            LOG.warning("%s: not a 3x3 intrinsic matrix", c)
    return None


def decode_tiff(path: Path, stats: ConvertStats) -> Optional[np.ndarray]:
    """16-bit Bayer-GB TIFF (12 bit used) -> 8-bit BGR. Already-decoded 8-bit colour images pass through."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
        if img.dtype == np.uint16:
            img = np.right_shift(img, 4).astype(np.uint8)
    elif img.dtype == np.uint16:
        img = np.right_shift(img, 4).astype(np.uint8)
        stats.skip("image already demosaiced")
    else:
        stats.skip("image already 8-bit")
    return img


def _lights(labels: List[dict], stats: ConvertStats) -> Tuple[np.ndarray, List[int]]:
    boxes, attrs = [], []
    for lab in labels:
        a = lab.get("attributes", {}) or {}
        direction = str(a.get("direction", "front")).lower()
        if direction != "front":
            stats.skip(f"light facing {direction}")
            continue
        x, y, bw, bh = (float(lab[k]) for k in ("x", "y", "w", "h"))
        xyxy = [x, y, x + bw, y + bh]
        if bw <= 0 or bh <= 0:
            stats.skip("degenerate box")
            continue
        state = str(a.get("state", "unknown")).lower()
        if state in STATE_ATTR:
            attr = ELEMENT_ATTRS.index(STATE_ATTR[state])
        elif state == "unknown":
            attr = -1
        else:
            stats.skip(f"light state {state}")
            attr = -1
        pict = str(a.get("pictogram", "")).lower()
        if pict not in PLAIN_PICTOGRAMS:
            stats.skip("light arrow meaning dropped" if pict.startswith("arrow") else f"light pictogram {pict} dropped")
        boxes.append(xyxy)
        attrs.append(attr)
    return np.array(boxes, np.float32).reshape(-1, 4), attrs


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    w = Writer(args.dst, "dtld", args.licence, copy_images=True, overwrite=args.overwrite, limit=args.limit)
    scratch = w.dst / ".decode"
    scratch.mkdir(parents=True, exist_ok=True)
    lm = LabelMask(elements=True, el_vocab=(EL_LIGHT,), el_attr=True)
    try:
        for split in args.splits:
            files = _label_files(root, split, args.labels)
            if not files:
                LOG.warning("%s: no label file for split %s", root, split)
                continue
            count = 0
            for lf in files:
                LOG.info("%s: %s", split, lf)
                for rec in _load_labels(lf):
                    if args.limit is not None and count >= args.limit:
                        break
                    src = _resolve(root, str(rec.get("image_path", "")))
                    if src is None:
                        w.stats.skip_frame("image missing")
                        continue
                    img = decode_tiff(src, w.stats)
                    if img is None:
                        w.stats.skip_frame("image unreadable")
                        continue
                    hw = img.shape[:2]
                    stem = src.stem
                    tmp = scratch / f"{stem}.jpg"
                    cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])
                    city = Path(str(rec.get("image_path", ""))).parts[0] if rec.get("image_path") else ""
                    K = _intrinsics(root, args.calib_dir, lf, city)
                    if K is not None:
                        T, calib = ground_cam2ego(args.cam_height), "measured"
                    else:
                        K, T, calib = nominal_camera(args, hw)
                    el_boxes, el_attr = _lights(rec.get("labels", []) or [], w.stats)
                    s = w.add(split, stem, tmp, hw=hw, label_mask=lm, K=K, cam2ego=T, calib=calib, el_boxes=el_boxes,
                              el_cls=[EL_LIGHT] * len(el_boxes), el_attr=el_attr)
                    tmp.unlink(missing_ok=True)
                    if s is not None:
                        count += 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return w.finish()
