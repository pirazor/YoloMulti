"""nuImages -> SPARC samples: road-user boxes of the CAM_FRONT key frames with measured intrinsics and mount.

Expected layout under ``--root`` (the official download; the JSON tables are read directly, no devkit)::

    v1.0-train/{sample_data,calibrated_sensor,sensor,object_ann,category}.json     (v1.0-val, v1.0-mini, v1.0-test)
    samples/CAM_FRONT/<name>.jpg     1600 x 900 key frames   (sweeps/ holds the unannotated non-key frames)

Table fields used (confirmed from nuscenes-devkit ``docs/schema_nuimages.md``): ``sample_data`` (``token``,
``filename`` relative to the root, ``is_key_frame``, ``calibrated_sensor_token``, ``sample_token``, ``width``,
``height``, ``timestamp``), ``calibrated_sensor`` (``sensor_token``, ``translation`` [x, y, z] m, ``rotation``
quaternion [w, x, y, z], ``camera_intrinsic`` 3x3), ``sensor`` (``channel``), ``object_ann`` (``sample_data_token``,
``category_token``, ``attribute_tokens``, ``bbox`` [xmin, ymin, xmax, ymax]), ``category`` (``name``). Only
``is_key_frame`` records of the ``CAM_FRONT`` channel are converted (object_ann exists for key frames only); a
key frame with no object_ann row is a labelled frame without road users. A version whose object_ann table is empty
(v1.0-test) is skipped with a warning.

Category mapping: ``human.pedestrian.*`` -> pedestrian, ``vehicle.car`` -> car, ``vehicle.truck`` -> truck,
``vehicle.bus.*`` -> bus, ``vehicle.motorcycle`` -> motorcycle, ``vehicle.bicycle`` -> bicycle,
``vehicle.trailer`` / ``vehicle.construction`` / ``vehicle.emergency.*`` -> truck (large special vehicles; the
closest of the 8 classes), everything else (animal, movable_object.*, static_object.*, vehicle.ego, flat.*) is
skipped and counted. nuImages boxes a bicycle / motorcycle together with its rider (attribute ``cycle.with_rider``),
so no ``rider`` box exists and ``cls_vocab`` = (pedestrian, car, truck, bus, motorcycle, bicycle).

Calibration: ``K = camera_intrinsic`` (``calib="measured"``) and ``cam2ego`` from the calibrated sensor's rotation
/ translation. The nuScenes ego frame is x forward / y left / z up with the origin on the ground under the rear
axle and the camera frame is x right / y down / z forward, i.e. the SPARC conventions; the camera's x / y
translation is subtracted so the sample's ego origin lies on the ground below the camera (z kept).

Splits: ``--splits train val`` -> ``v1.0-train`` / ``v1.0-val`` (``test`` -> ``v1.0-test``, ``mini`` -> ``v1.0-mini``;
a full ``v1.0-*`` name is used as given). Agents only (no light / sign labels in nuImages). Licence: CC BY-NC-SA 4.0,
non-commercial (``research``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sparc.data.converters.base import ConvertStats, Writer, add_common_args, image_size
from sparc.data.schema import AGENT_CLASSES, LabelMask

LOG = logging.getLogger("sparc.convert.nuimages")

SPLIT_VERSION = {"train": "v1.0-train", "val": "v1.0-val", "test": "v1.0-test", "mini": "v1.0-mini"}
EXACT_MAP = {"vehicle.car": "car", "vehicle.truck": "truck", "vehicle.motorcycle": "motorcycle",
             "vehicle.bicycle": "bicycle", "vehicle.trailer": "truck", "vehicle.construction": "truck"}
PREFIX_MAP = (("human.pedestrian", "pedestrian"), ("vehicle.bus", "bus"), ("vehicle.emergency", "truck"))
CLS_VOCAB = tuple(AGENT_CLASSES.index(c) for c in ("pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"))
TABLES = ("sample_data", "calibrated_sensor", "sensor", "object_ann", "category")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="nuImages root (v1.0-*/ tables, samples/)")
    p.add_argument("--channel", default="CAM_FRONT")
    add_common_args(p)


def map_category(name: str) -> Optional[str]:
    if name in EXACT_MAP:
        return EXACT_MAP[name]
    for prefix, cls in PREFIX_MAP:
        if name == prefix or name.startswith(prefix + "."):
            return cls
    return None


def quat_to_rot(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion given as (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def cam2ego(calibrated: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(calibrated["rotation"])
    T[2, 3] = float(calibrated["translation"][2])  # x, y dropped: ego origin on the ground below the camera
    return T


def _tables(table_dir: Path) -> Dict[str, list]:
    out = {}
    for name in TABLES:
        p = table_dir / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing")
        with open(p, "r", encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def _boxes(anns: List[dict], cat_name: Dict[str, str], stats: ConvertStats):
    boxes, cls = [], []
    for a in anns:
        name = cat_name.get(a.get("category_token"), "")
        mapped = map_category(name)
        if mapped is None:
            stats.skip(f"category {name or 'unknown'}")
            continue
        b = a.get("bbox")
        if b is None or len(b) != 4:
            stats.skip("object without bbox")
            continue
        xyxy = [float(v) for v in b]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            stats.skip("degenerate box")
            continue
        boxes.append(xyxy)
        cls.append(AGENT_CLASSES.index(mapped))
    return np.array(boxes, np.float32).reshape(-1, 4), cls


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    w = Writer(args.dst, "nuimages", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    lm = LabelMask(agents=True, cls_vocab=CLS_VOCAB)
    for split in args.splits:
        version = split if split.startswith("v1.0-") else SPLIT_VERSION.get(split, f"v1.0-{split}")
        split_name = version[len("v1.0-"):] if split.startswith("v1.0-") else split
        table_dir = root / version
        if not table_dir.exists():
            LOG.warning("%s: no table directory for split %s", root, split)
            continue
        t = _tables(table_dir)
        if not t["object_ann"]:
            LOG.warning("%s: object_ann is empty (unannotated split), skipping", version)
            w.stats.skip_frame(f"{version} unannotated")
            continue
        channel_sensors = {s["token"] for s in t["sensor"] if s.get("channel") == args.channel}
        calibrated = {c["token"]: c for c in t["calibrated_sensor"] if c.get("sensor_token") in channel_sensors}
        cat_name = {c["token"]: c["name"] for c in t["category"]}
        anns: Dict[str, List[dict]] = {}
        for a in t["object_ann"]:
            anns.setdefault(a["sample_data_token"], []).append(a)
        count = 0
        for sd in sorted(t["sample_data"], key=lambda r: (int(r.get("timestamp", 0)), r["token"])):
            if args.limit is not None and count >= args.limit:
                break
            if not sd.get("is_key_frame") or sd.get("calibrated_sensor_token") not in calibrated:
                continue
            img = root / sd["filename"]
            hw = image_size(img)
            if hw is None:
                w.stats.skip_frame("image missing")
                continue
            cal = calibrated[sd["calibrated_sensor_token"]]
            K = np.asarray(cal["camera_intrinsic"], dtype=np.float64).reshape(3, 3)
            boxes, cls = _boxes(anns.get(sd["token"], []), cat_name, w.stats)
            ts = sd.get("timestamp")
            if w.add(split_name, Path(sd["filename"]).stem, img, hw=hw, label_mask=lm, K=K, cam2ego=cam2ego(cal),
                     calib="measured", boxes2d=boxes, cls=cls,
                     timestamp_ns=None if ts is None else int(ts) * 1000) is not None:
                count += 1
    return w.finish()
