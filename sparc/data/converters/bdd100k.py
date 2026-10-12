"""BDD100K -> SPARC samples: detection (10 classes + traffic-light colour), drivable area, lane markings.

Supported label files (both the legacy and the Scalabel layout, auto-discovered under ``--root``):

* detection: ``labels/det_20/det_<split>.json`` or ``labels/bdd100k_labels_images_<split>.json``
  (a list of frames: ``{"name", "labels": [{"category", "attributes": {..., "trafficLightColor"}, "box2d"}]}``);
* drivable area: ``labels/drivable/polygons/drivable_<split>.json`` (``category`` = direct / alternative) or the
  legacy ``category: "drivable area"`` + ``attributes.areaType`` entries of the same frame;
* lanes: ``labels/lane/polygons/lane_<split>.json`` (``category`` = single white / double yellow / road curb / ...,
  ``attributes.laneDirection`` parallel / vertical, ``laneStyle`` solid / dashed) or the legacy ``category: "lane"``
  + ``attributes.laneType``. ``poly2d`` vertices may be bezier control points (``types`` with ``C``).
* images: ``images/100k/<split>/<name>.jpg``.

Road users go to ``boxes2d`` (8 classes), lights and signs to the element table (colour -> ``el_attr``). Lanes marked
``vertical`` (crossing the road) and crosswalks are skipped: they are not markings along the travel direction.
BDD100K has no calibration: a nominal camera is used (``--hfov``, ``--cam-height``, ``--horizon``) and the sample is
marked ``calib="nominal"``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, image_size, nominal_camera,
                                        nominal_camera_args)
from sparc.data.raster import draw_polylines, fill_polygons, flatten_poly2d, line_thickness
from sparc.data.schema import (AGENT_CLASSES, EL_LIGHT, EL_SIGN, ELEMENT_ATTRS, LANE_CONTINUITY, LANE_TYPES,
                               LabelMask, LaneInstance)

LOG = logging.getLogger("sparc.convert.bdd100k")

AGENT_MAP = {"pedestrian": "pedestrian", "person": "pedestrian", "rider": "rider", "car": "car", "truck": "truck",
             "bus": "bus", "train": "train", "motorcycle": "motorcycle", "motor": "motorcycle", "bicycle": "bicycle",
             "bike": "bicycle"}
ELEMENT_MAP = {"traffic light": EL_LIGHT, "traffic sign": EL_SIGN}
COLOR_MAP = {"red": "red", "r": "red", "green": "green", "g": "green", "yellow": "yellow", "y": "yellow",
             "none": "unknown", "n": "unknown"}
DA_MAP = {"direct": 1, "alternative": 2}
LANE_TYPE_MAP = {"single white": "single_white", "single yellow": "single_yellow", "double white": "double_white",
                 "double yellow": "double_yellow", "road curb": "curb", "single other": "other", "double other": "other"}
SKIP_LANE_CATEGORIES = {"crosswalk"}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="BDD100K root (images/100k, labels/...)")
    p.add_argument("--det", default=None, help="detection json (overrides discovery; one per split with {split})")
    p.add_argument("--lane", default=None, help="lane polygons json ({split} placeholder)")
    p.add_argument("--drivable", default=None, help="drivable polygons json ({split} placeholder)")
    p.add_argument("--lane-px", type=float, default=8.0, help="lane line thickness at 1280 px width")
    add_common_args(p)
    nominal_camera_args(p, hfov=65.0, height=1.4, horizon=0.5)


def _find(root: Path, split: str, explicit: Optional[str], candidates: List[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit.format(split=split))
        return p if p.exists() else None
    for c in candidates:
        p = root / c.format(split=split)
        if p.exists():
            return p
    return None


def _load_frames(path: Optional[Path]) -> Dict[str, dict]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "frames" in data:
        data = data["frames"]
    return {fr["name"]: fr for fr in data}


def _color(attrs: dict) -> int:
    v = attrs.get("trafficLightColor")
    if v is None:
        return -1
    key = str(v).strip().lower()
    if key in ("na", ""):
        return -1
    name = COLOR_MAP.get(key)
    return ELEMENT_ATTRS.index(name) if name else -1


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    w = Writer(args.dst, "bdd100k", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    for split in args.splits:
        det = _find(root, split, args.det, ["labels/det_20/det_{split}.json", "labels/bdd100k_labels_images_{split}.json",
                                             "labels/det_{split}.json"])
        lane = _find(root, split, args.lane, ["labels/lane/polygons/lane_{split}.json", "labels/lane_{split}.json"])
        drv = _find(root, split, args.drivable, ["labels/drivable/polygons/drivable_{split}.json",
                                                 "labels/drivable_{split}.json"])
        if det is None and lane is None and drv is None:
            LOG.warning("%s: no label file found for split %s", root, split)
            continue
        LOG.info("%s: det=%s lane=%s drivable=%s", split, det, lane, drv)
        det_frames, lane_frames, drv_frames = _load_frames(det), _load_frames(lane), _load_frames(drv)
        names = sorted(set(det_frames) | set(lane_frames) | set(drv_frames))
        img_dir = root / "images" / "100k" / split
        count = 0
        for name in names:
            if args.limit is not None and count >= args.limit:
                break
            img = img_dir / name
            if not img.exists():
                img = root / "images" / split / name
            hw = image_size(img)
            if hw is None:
                w.stats.skip_frame("image missing")
                continue
            K, T, calib = nominal_camera(args, hw)
            fields = _frame(det_frames.get(name), lane_frames.get(name), drv_frames.get(name), hw, args, w.stats,
                            legacy=det_frames.get(name) if lane is None and drv is None else None)
            if fields is None:
                continue
            fields.update(K=K, cam2ego=T, calib=calib)
            if w.add(split, Path(name).stem, img, **fields) is not None:
                count += 1
    return w.finish()


def _frame(det: Optional[dict], lane: Optional[dict], drv: Optional[dict], hw, args, stats: ConvertStats,
           legacy: Optional[dict] = None) -> Optional[dict]:
    """Labels of one frame from up to three files. In the legacy layout the detection file also carries the
    drivable / lane polygons (``legacy`` is then the same frame)."""
    boxes, cls, el_boxes, el_cls, el_attr = [], [], [], [], []
    has_det = det is not None
    if has_det:
        for lab in det.get("labels", []):
            cat = str(lab.get("category", "")).lower()
            b = lab.get("box2d")
            if b is None:
                continue
            xyxy = [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                stats.skip("degenerate box")
                continue
            if cat in AGENT_MAP:
                boxes.append(xyxy)
                cls.append(AGENT_CLASSES.index(AGENT_MAP[cat]))
            elif cat in ELEMENT_MAP:
                el_boxes.append(xyxy)
                el_cls.append(ELEMENT_MAP[cat])
                el_attr.append(_color(lab.get("attributes", {}) or {}) if ELEMENT_MAP[cat] == EL_LIGHT else -1)
            else:
                stats.skip(f"box category {cat}")
    # drivable area
    da_mask = None
    da_labels = []
    for src in (drv, legacy):
        if src is None:
            continue
        for lab in src.get("labels", []):
            cat = str(lab.get("category", "")).lower()
            attrs = lab.get("attributes", {}) or {}
            if cat in DA_MAP:
                da_labels.append((DA_MAP[cat], lab))
            elif cat == "drivable area":
                t = str(attrs.get("areaType", "")).lower()
                if t in DA_MAP:
                    da_labels.append((DA_MAP[t], lab))
                else:
                    stats.skip(f"drivable areaType {t}")
    has_da = drv is not None or (legacy is not None and any(str(l.get("category", "")).lower() == "drivable area"
                                                            for l in legacy.get("labels", [])))
    if has_da:
        da_mask = np.zeros(hw, np.uint8)
        for value in (2, 1):  # draw alternative first, direct on top
            polys = [flatten_poly2d(p["vertices"], p.get("types", ""), p.get("closed", True))
                     for v, lab in da_labels if v == value for p in lab.get("poly2d", [])]
            fill_polygons(da_mask, polys, value)
    # lanes
    lanes: List[LaneInstance] = []
    lane_mask = None
    lane_labels = []
    for src in (lane, legacy):
        if src is None:
            continue
        for lab in src.get("labels", []):
            cat = str(lab.get("category", "")).lower()
            attrs = lab.get("attributes", {}) or {}
            if cat == "lane":
                ltype = str(attrs.get("laneType", "")).lower()
            elif cat in LANE_TYPE_MAP or cat in SKIP_LANE_CATEGORIES:
                ltype = cat
            else:
                continue
            lane_labels.append((ltype, attrs, lab))
    has_lanes = lane is not None or (legacy is not None and any(str(l.get("category", "")).lower() == "lane"
                                                               for l in legacy.get("labels", [])))
    if has_lanes:
        lane_mask = np.zeros(hw, np.uint8)
        thick = line_thickness(hw, args.lane_px)
        for ltype, attrs, lab in lane_labels:
            if ltype in SKIP_LANE_CATEGORIES:
                stats.skip("lane crosswalk")
                continue
            if str(attrs.get("laneDirection", "parallel")).lower() == "vertical":
                stats.skip("lane vertical")
                continue
            if ltype not in LANE_TYPE_MAP:
                stats.skip(f"lane type {ltype}")
                continue
            style = str(attrs.get("laneStyle", "")).lower()
            cont = LANE_CONTINUITY.index(style) if style in LANE_CONTINUITY else -1
            for poly in lab.get("poly2d", []):
                pts = flatten_poly2d(poly["vertices"], poly.get("types", ""), poly.get("closed", False))
                if len(pts) < 2:
                    stats.skip("lane < 2 points")
                    continue
                order = np.argsort(-pts[:, 1], kind="stable")  # bottom (near) first
                lanes.append(LaneInstance(pts[order], continuity=cont, ltype=LANE_TYPES.index(LANE_TYPE_MAP[ltype])))
                draw_polylines(lane_mask, [pts], 1, thick)
    if not (has_det or has_da or has_lanes):
        stats.skip_frame("no labels")
        return None
    lm = LabelMask(agents=has_det, cls_vocab=range(len(AGENT_CLASSES)) if has_det else (),
                   elements=has_det, el_vocab=(EL_LIGHT, EL_SIGN) if has_det else (), el_attr=has_det,
                   da=has_da, lane_geom=has_lanes, lane_cont=has_lanes, lane_type=has_lanes, lane_mask=has_lanes)
    return dict(label_mask=lm, boxes2d=np.array(boxes, np.float32).reshape(-1, 4), cls=cls,
                el_boxes=np.array(el_boxes, np.float32).reshape(-1, 4), el_cls=el_cls, el_attr=el_attr, lanes=lanes,
                da_mask=da_mask, lane_mask=lane_mask)
