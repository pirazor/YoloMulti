"""ZOD Frames (Zenseact Open Dataset) -> SPARC samples: road users, signs (fine class), lights, ego road, lane
markings, measured intrinsics.

Expected layout under ``--root`` (the official download, ``zod download --subset=frames``)::

    trainval-frames-mini.json | trainval-frames-full.json      {"train": [<frame>, ...], "val": [...]}
    single_frames/<id>/annotations/object_detection.json
    single_frames/<id>/annotations/traffic_signs.json
    single_frames/<id>/annotations/ego_road.json
    single_frames/<id>/annotations/lane_markings.json
    single_frames/<id>/camera_front_blur/<id>_<timestamp>.jpg     (``--anonymization dnat``: camera_front_dnat/)
    single_frames/<id>/calibration.json

Confirmed from the devkit (github.com/zenseact/zod, ``zod/constants.py``, ``zod/anno/*.py``,
``zod/data_classes/calibration.py``, ``zod/_zod_dataset.py``):

* the trainval file is ``trainval-frames-<mini|full>.json`` (hyphens; the underscore spelling is also tried), a dict
  keyed ``train`` / ``val`` whose entries are the frame ``Information`` dicts (``{"id": "000000", ...}``); only
  ``id`` is used here, the paths above are rebuilt from the documented folder layout (an entry that is a bare id
  string is accepted too);
* every annotation file but road_condition is a JSON **list** of GeoJSON-like features
  ``{"geometry": {"coordinates": [[x, y], ...]}, "properties": {...}}``; the devkit never reads ``geometry.type`` and
  treats ``coordinates`` as a flat list of ``[x, y]`` points (a nested ring ``[[[x, y], ...]]`` is unwrapped here);
* object_detection properties: ``class`` (Vehicle, VulnerableVehicle, Pedestrian, Animal, PoleObject,
  TrafficBeacon, TrafficSign, TrafficSignal, TrafficGuide, DynamicBarrier, Inconclusive), ``type`` (Vehicle: Car, Van,
  Truck, Bus, Trailer, TramTrain, HeavyEquip, Emergency, Other; VulnerableVehicle: Bicycle, Motorcycle, Stroller,
  Wheelchair, PersonalTransporter, NoRider, Other), ``unclear``, ``occlusion_ratio``, ``with_rider`` and
  ``traffic_content_visible`` (the strings "True" / "False"), ``emergency``, ``artificial``, and ``location_3d`` +
  ``size_3d_*`` / ``orientation_3d_q*`` when a 3D box exists; the 2D box is the min / max of the polygon points;
* traffic_signs properties: ``class`` (the sign name, e.g. ``Priority_Stop``; ``Unclear`` / ``NotListed`` when the
  sign is unreadable or outside the taxonomy), ``occlusion_ratio``, ``is_electronic``, ``unclear``;
* ego_road properties: ``class`` ``EgoRoad_Road`` / ``EgoRoad_Debris`` (the devkit keeps every feature whose class
  contains ``EgoRoad`` and discards the rest, e.g. barriers / road edges);
* lane_markings: features with a ``class`` are lane markings (``solid``, ``dashed``, ...; ``shaded_area`` is a
  hatched area), features without ``class`` are road paintings (arrows, text, crosswalks);
* calibration.json: ``{"FC": {"intrinsics": 3x3, "distortion": [4], "undistortion": [4], "extrinsics": 4x4 camera
  pose in the ego frame, "image_dimensions": [w, h], "field_of_view": [h, v]}}``. The camera is a Kannala-Brandt
  fisheye: only the pinhole part of ``intrinsics`` (fx, fy, cx, cy) is used, the 4 distortion coefficients are ignored
  (``calib="measured"``; boxes near the border are therefore not exactly pinhole-consistent).

Assumptions (not confirmed): the ego frame is x forward / y left / z up at the rear axle on the ground and the camera
frame is OpenCV-style (x right, y down, z forward), so ``extrinsics`` is used as ``cam2ego`` after moving the origin
below the camera (its x / y translation is zeroed). ZOD lights carry no colour label (``el_attr = -1``).

Mapping:

* Vehicle: Car / Van -> car, Truck / Trailer / HeavyEquip / Emergency -> truck, Bus -> bus, TramTrain -> train, Other
  skipped; VulnerableVehicle: Bicycle -> bicycle, Motorcycle -> motorcycle (the box encloses rider and vehicle; no
  separate ``rider`` box exists, so ``rider`` is not in ``cls_vocab``), other types skipped; Pedestrian -> pedestrian;
  Animal, PoleObject, TrafficBeacon, TrafficGuide, DynamicBarrier, Inconclusive / Unclear skipped and counted.
* TrafficSignal facing the camera -> light (``el_attr = -1``); TrafficSign facing the camera -> sign. When the
  ``traffic_signs`` file exists its boxes replace the object_detection sign boxes (counted) and carry
  ``el_sign = "zod:<class>"`` ("" for Unclear / NotListed); backs of signs / lights are skipped and counted.
* ego road polygons -> ``da_mask`` value 1 (direct), lane-marking polygons -> ``lane_mask`` value 1 (filled; no
  ``lane_geom`` because polygons are not centrelines); road paintings and shaded areas are counted and dropped.
* 3D boxes are ignored for now and counted.

Label mask per frame: ``agents`` + ``cls_vocab`` when object_detection exists, ``elements`` with ``el_vocab = (0, 1)``
(object_detection) or ``(traffic_sign,)`` (traffic_signs only), ``el_sign`` when traffic_signs exists, ``da`` /
``lane_mask`` when the respective file exists. Licence: CC BY-SA 4.0 (``bsd``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, image_size, nominal_camera,
                                        nominal_camera_args)
from sparc.data.raster import fill_polygons
from sparc.data.schema import AGENT_CLASSES, EL_LIGHT, EL_SIGN, LabelMask

LOG = logging.getLogger("sparc.convert.zod")

TRAINVAL = {"mini": ("trainval-frames-mini.json", "trainval_frames_mini.json"),
            "full": ("trainval-frames-full.json", "trainval_frames_full.json")}
VEHICLE_MAP = {"Car": "car", "Van": "car", "Truck": "truck", "Trailer": "truck", "HeavyEquip": "truck",
               "Emergency": "truck", "Bus": "bus", "TramTrain": "train"}
VULNERABLE_MAP = {"Bicycle": "bicycle", "Motorcycle": "motorcycle"}
CLS_VOCAB = tuple(AGENT_CLASSES.index(c) for c in ("pedestrian", "car", "truck", "bus", "train", "motorcycle", "bicycle"))
UNNAMED_SIGNS = {"unclear", "notlisted", "inconclusive", ""}
ANNOTATION_FILES = ("object_detection", "traffic_signs", "ego_road", "lane_markings")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="ZOD root (single_frames/, trainval-frames-*.json)")
    p.add_argument("--version", default="auto", choices=("auto", "mini", "full"),
                   help="which trainval file to use (auto: full if present, else mini)")
    p.add_argument("--trainval", default=None, help="explicit trainval json (overrides --version)")
    p.add_argument("--anonymization", default="blur", choices=("blur", "dnat"), help="camera_front_<x> folder")
    add_common_args(p, default_licence="bsd")
    nominal_camera_args(p, hfov=110.0, height=1.6, horizon=0.5)


def _trainval(root: Path, args) -> Path:
    if args.trainval:
        return Path(args.trainval)
    versions = ("full", "mini") if args.version == "auto" else (args.version,)
    for v in versions:
        for name in TRAINVAL[v]:
            if (root / name).exists():
                return root / name
    raise FileNotFoundError(f"no trainval-frames-{'/'.join(versions)}.json under {root}")


def _frame_id(entry) -> str:
    fid = entry.get("id") if isinstance(entry, dict) else entry
    fid = str(fid)
    return fid.zfill(6) if fid.isdigit() else fid


def _points(coords) -> np.ndarray:
    """GeoJSON-like coordinates -> (M, 2); a nested ring (Polygon / MultiPolygon) is unwrapped to its outer ring."""
    c = coords
    while isinstance(c, (list, tuple)) and len(c) and isinstance(c[0], (list, tuple)) and len(c[0]) and \
            isinstance(c[0][0], (list, tuple)):
        c = c[0]
    return np.asarray(c, dtype=np.float64).reshape(-1, 2)


def _bool_prop(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def _box(coords, stats: ConvertStats) -> Optional[List[float]]:
    pts = _points(coords)
    if len(pts) == 0 or not np.isfinite(pts).all():
        stats.skip("object without geometry")
        return None
    xyxy = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
    if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
        stats.skip("degenerate box")
        return None
    return xyxy


def _load(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("features", [])


def _objects(features: list, signs_from_tsr: bool, stats: ConvertStats):
    boxes, cls, el_boxes, el_cls, el_attr, el_sign = [], [], [], [], [], []
    for feat in features:
        props = feat.get("properties", {}) or {}
        klass = str(props.get("class", ""))
        otype = str(props.get("type", "") or "")
        if "location_3d" in props:
            stats.skip("3d box ignored")
        if "Inconclusive" in klass or klass == "Unclear":
            stats.skip("unclear object")
            continue
        if "Inconclusive" in otype:
            stats.skip(f"{klass} type inconclusive")
            continue
        if klass in ("TrafficSign", "TrafficSignal"):
            if _bool_prop(props.get("traffic_content_visible")) is False:
                stats.skip("sign back side" if klass == "TrafficSign" else "light back side")
                continue
            if klass == "TrafficSign" and signs_from_tsr:
                stats.skip("sign box replaced by traffic_signs")
                continue
            xyxy = _box(feat.get("geometry", {}).get("coordinates", []), stats)
            if xyxy is None:
                continue
            el_boxes.append(xyxy)
            el_cls.append(EL_SIGN if klass == "TrafficSign" else EL_LIGHT)
            el_attr.append(-1)
            el_sign.append("")
            continue
        if klass == "Pedestrian":
            name = "pedestrian"
        elif klass == "Vehicle":
            name = VEHICLE_MAP.get(otype)
        elif klass == "VulnerableVehicle":
            name = VULNERABLE_MAP.get(otype)
        else:
            name = None
        if name is None:
            stats.skip(f"object class {klass}{'/' + otype if otype else ''}")
            continue
        xyxy = _box(feat.get("geometry", {}).get("coordinates", []), stats)
        if xyxy is None:
            continue
        boxes.append(xyxy)
        cls.append(AGENT_CLASSES.index(name))
    return boxes, cls, el_boxes, el_cls, el_attr, el_sign


def _signs(features: list, stats: ConvertStats):
    el_boxes, names = [], []
    for feat in features:
        props = feat.get("properties", {}) or {}
        xyxy = _box(feat.get("geometry", {}).get("coordinates", []), stats)
        if xyxy is None:
            continue
        klass = str(props.get("class", "")).strip()
        el_boxes.append(xyxy)
        names.append("" if klass.lower() in UNNAMED_SIGNS else f"zod:{klass}")
    return el_boxes, names


def _ego_road(features: list, hw, stats: ConvertStats) -> np.ndarray:
    mask = np.zeros(hw, np.uint8)
    polys = []
    for feat in features:
        klass = str((feat.get("properties", {}) or {}).get("class", ""))
        if "EgoRoad" not in klass:
            stats.skip(f"ego road class {klass or 'none'}")
            continue
        pts = _points(feat.get("geometry", {}).get("coordinates", []))
        if len(pts) < 3:
            stats.skip("ego road polygon < 3 points")
            continue
        polys.append(pts)
    fill_polygons(mask, polys, 1)
    return mask


def _lane_markings(features: list, hw, stats: ConvertStats) -> np.ndarray:
    mask = np.zeros(hw, np.uint8)
    polys = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        if "class" not in props:
            stats.skip("road painting")
            continue
        if str(props["class"]) == "shaded_area":
            stats.skip("lane shaded area")
            continue
        pts = _points(feat.get("geometry", {}).get("coordinates", []))
        if len(pts) < 3:
            stats.skip("lane polygon < 3 points")
            continue
        polys.append(pts)
    fill_polygons(mask, polys, 1)
    return mask


def _camera(frame_dir: Path, args, hw, stats: ConvertStats) -> Tuple[np.ndarray, np.ndarray, str]:
    calib = frame_dir / "calibration.json"
    if calib.exists():
        with open(calib, "r", encoding="utf-8") as f:
            data = json.load(f)
        fc = data.get("FC", data)
        K_in = fc.get("intrinsics")
        if K_in is not None:
            K_in = np.asarray(K_in, dtype=np.float64).reshape(3, 3)
            K = np.array([[K_in[0, 0], 0.0, K_in[0, 2]], [0.0, K_in[1, 1], K_in[1, 2]], [0.0, 0.0, 1.0]])
            T = np.asarray(fc.get("extrinsics", np.eye(4)), dtype=np.float64).reshape(4, 4).copy()
            T[0, 3] = T[1, 3] = 0.0  # ego origin on the ground below the camera
            T[3] = (0.0, 0.0, 0.0, 1.0)
            return K, T, "measured"
    stats.skip("calibration missing")
    return nominal_camera(args, hw)


def _image(frame_dir: Path, args, stats: ConvertStats) -> Optional[Path]:
    cam = frame_dir / f"camera_front_{args.anonymization}"
    files = sorted(p for p in cam.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if cam.exists() else []
    if not files:
        return None
    if len(files) > 1:
        stats.skip("multiple camera frames")
    return files[0]


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    trainval = _trainval(root, args)
    LOG.info("trainval: %s", trainval)
    with open(trainval, "r", encoding="utf-8") as f:
        lists = json.load(f)
    w = Writer(args.dst, "zod", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    for split in args.splits:
        entries = lists.get(split)
        if entries is None:
            LOG.warning("%s: no split %r (has %s)", trainval, split, sorted(lists))
            continue
        count = 0
        for entry in entries:
            if args.limit is not None and count >= args.limit:
                break
            fid = _frame_id(entry)
            frame_dir = root / "single_frames" / fid
            img = _image(frame_dir, args, w.stats)
            hw = image_size(img) if img is not None else None
            if hw is None:
                w.stats.skip_frame("image missing")
                continue
            files: Dict[str, Optional[Path]] = {n: frame_dir / "annotations" / f"{n}.json" for n in ANNOTATION_FILES}
            files = {n: (p if p.exists() else None) for n, p in files.items()}
            if not any(files.values()):
                w.stats.skip_frame("no labels")
                continue
            fields = _frame(files, hw, w.stats)
            K, T, calib = _camera(frame_dir, args, hw, w.stats)
            fields.update(K=K, cam2ego=T, calib=calib)
            if w.add(split, fid, img, hw=hw, **fields) is not None:
                count += 1
    return w.finish()


def _frame(files: Dict[str, Optional[Path]], hw, stats: ConvertStats) -> dict:
    has_od, has_tsr = files["object_detection"] is not None, files["traffic_signs"] is not None
    boxes, cls, el_boxes, el_cls, el_attr, el_sign = [], [], [], [], [], []
    if has_od:
        boxes, cls, el_boxes, el_cls, el_attr, el_sign = _objects(_load(files["object_detection"]), has_tsr, stats)
    if has_tsr:
        sb, names = _signs(_load(files["traffic_signs"]), stats)
        el_boxes += sb
        el_cls += [EL_SIGN] * len(sb)
        el_attr += [-1] * len(sb)
        el_sign += names
    da_mask = _ego_road(_load(files["ego_road"]), hw, stats) if files["ego_road"] is not None else None
    lane_mask = _lane_markings(_load(files["lane_markings"]), hw, stats) if files["lane_markings"] is not None else None
    el_vocab = (EL_LIGHT, EL_SIGN) if has_od else ((EL_SIGN,) if has_tsr else ())
    lm = LabelMask(agents=has_od, cls_vocab=CLS_VOCAB if has_od else (), elements=bool(el_vocab), el_vocab=el_vocab,
                   el_sign=has_tsr, da=da_mask is not None, lane_mask=lane_mask is not None)
    return dict(label_mask=lm, boxes2d=np.array(boxes, np.float32).reshape(-1, 4), cls=cls,
                el_boxes=np.array(el_boxes, np.float32).reshape(-1, 4), el_cls=el_cls, el_attr=el_attr,
                el_sign=el_sign, da_mask=da_mask, lane_mask=lane_mask)
