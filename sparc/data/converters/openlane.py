"""OpenLane (Waymo-based 3D lanes) -> SPARC samples: 2D + 3D lane polylines with order, type and continuity.

Expected layout under ``--root`` (the official download):

* ``images/<split>/<segment>/<frame>.jpg`` with ``<split>`` = ``training`` / ``validation`` (``--splits train val``);
* ``lane3d_1000/<split>/<segment>/<frame>.json`` (``--anno``; ``lane3d_300``, the 300-sequence subset, is accepted and
  found automatically when ``lane3d_1000`` is absent).

Each JSON holds ``intrinsic`` (3x3), ``extrinsic`` (4x4, camera -> Waymo vehicle frame), ``file_path`` (image path
relative to ``images/``) and ``lane_lines``: ``{"category": int, "visibility": [n], "uv": [2, n], "xyz": [3, n],
"attribute": int, "track_id": int}``.

Confirmed against the repository documentation (fetched 2026-10):

* ``anno_criterion/Lane/README.md`` (github.com/OpenDriveLab/OpenLane): the keys and shapes above; ``uv`` are "2d lane
  points under image coordinate", ``xyz`` "3d lane points under camera coordinate", visibility "mainly relates to
  points in the far distance"; ``attribute``: "the left-left lane is 1, the left lane is 2, the right lane is 3, and
  the right-right lane is 4" (0 = none of the four); categories 0 unknown, 1 white-dash, 2 white-solid,
  3 double-white-dash, 4 double-white-solid, 5 white-ldash-rsolid, 6 white-lsolid-rdash, 7 yellow-dash,
  8 yellow-solid, 9 double-yellow-dash, 10 double-yellow-solid, 11 yellow-ldash-rsolid, 12 yellow-lsolid-rdash,
  20 left-curbside, 21 right-curbside (13..19 undefined).
* ``data/README.md``: ``images/``, ``lane3d_1000/``, ``lane3d_300/`` (300 sequences: 240 training, 60 validation),
  ``<dir>/training/segment-xxx/`` and ``<dir>/validation/...``, one ``xxx.json`` per frame.
* The README does not state the axes of the camera frame. The official evaluator
  (``eval/LANE_evaluation/lane3d/eval_3D_lane.py``) and the authors' PersFormer loader (``data/Load_Data.py``)
  transform the ground truth as ``extrinsic' @ inv(M) @ xyz`` with ``M = [[0,0,1],[-1,0,0],[0,-1,0]]`` and
  ``extrinsic'[:3,:3] = R_vg^-1 R R_vg R_gc`` where ``R_vg R_gc == M``; the net effect is ``R_vg^-1 (R xyz + t)``, so
  the raw ``xyz`` is in the frame the Waymo extrinsic acts on: the camera frame with Waymo axes (x forward, y left,
  z up, camera at the origin). OpenCV camera coordinates are then ``(X, Y, Z) = (-y, -z, x)`` and
  ``uv = intrinsic @ (X/Z, Y/Z, 1)``. This is **not trusted blindly**: :func:`detect_convention` projects every frame's
  ``xyz`` under this convention (``"waymo"``) and under a plain OpenCV camera frame (``"opencv"``) and keeps the one
  that reproduces ``uv`` (median error <= 3 px), else the frame is skipped and counted (``convention undetermined``).
  The manifest records the conventions seen.

Ego frame of the sample: the Waymo-axes camera frame shifted down to the ground, ``ego = (x, y, z + h)`` with the
camera height ``h = -median(z)`` of the visible lane points nearer than 15 m (lanes lie on the ground; all points
when none is that near; ``extrinsic[2, 3]`` for a frame without lanes, counted). The ego frame is therefore
camera-aligned (the camera pitch / roll of the extrinsic is not removed); ``cam2ego`` = OpenCV camera -> ego axes
with translation ``(0, 0, h)``, so ``project_ego(points3d, K, cam2ego)`` reproduces ``uv``. ``K = intrinsic``,
``calib = "measured"``.

Lanes: only points with ``visibility`` 1 are kept (dropped ones counted), a lane with < 2 visible points is skipped;
points are ordered near to far by ego x; ``points`` = ``uv``, ``points3d`` = ego metres. ``attribute`` -> order
-2 / -1 / +1 / +2 (0 and anything else -> unlabelled). ``category`` -> ``ltype`` single / double x white / yellow,
curb for 20 / 21, other for 0 and undefined numbers (counted); ``continuity`` dashed for ``*-dash``, solid for
``*-solid`` and the mixed ``ldash-rsolid`` / ``lsolid-rdash``, unlabelled for unknown and curbs.
Labels: ``lane_geom``, ``lane_3d``, ``lane_order``, ``lane_cont``, ``lane_type``, ``lane_mask`` (rasterised ``uv``).
Stems are ``<segment>__<frame>``; ``clip_id`` is the segment, ``frame_idx`` the index within it (file order).
The file names look like timestamps but their unit is not documented, so ``timestamp_ns`` is left unset.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparc.data.converters.base import ConvertStats, Writer, add_common_args, image_size
from sparc.data.raster import draw_polylines, line_thickness
from sparc.data.schema import (LANE_CONTINUITY, LANE_ORDER_UNLABELLED, LANE_TYPES, LabelMask, LaneInstance,
                               ground_cam2ego)

LOG = logging.getLogger("sparc.convert.openlane")

SPLIT_DIRS = {"train": "training", "val": "validation", "training": "training", "validation": "validation"}
ANNO_DIRS = ("lane3d_1000", "lane3d_300")
ATTR_ORDER = {1: -2, 2: -1, 3: 1, 4: 2}
# category -> (ltype name, continuity name or None)
CATEGORY_MAP = {
    0: ("other", None),
    1: ("single_white", "dashed"), 2: ("single_white", "solid"),
    3: ("double_white", "dashed"), 4: ("double_white", "solid"),
    5: ("double_white", "solid"), 6: ("double_white", "solid"),      # mixed dash / solid pairs
    7: ("single_yellow", "dashed"), 8: ("single_yellow", "solid"),
    9: ("double_yellow", "dashed"), 10: ("double_yellow", "solid"),
    11: ("double_yellow", "solid"), 12: ("double_yellow", "solid"),
    20: ("curb", None), 21: ("curb", None),
}
CONVENTIONS = ("waymo", "opencv")
CONVENTION_TOL_PX = 3.0
NEAR_M = 15.0
# OpenCV camera axes (x right, y down, z forward) -> Waymo-axes camera / ego axes (x forward, y left, z up)
M_CV2EGO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="OpenLane root (images/, lane3d_1000/ or lane3d_300/)")
    p.add_argument("--anno", default=None, help="annotation directory (default: <root>/lane3d_1000, else lane3d_300)")
    p.add_argument("--lane-px", type=float, default=8.0, help="lane line thickness at 1280 px width")
    add_common_args(p)


def to_opencv(xyz: np.ndarray, convention: str) -> np.ndarray:
    """Raw ``xyz`` (N, 3) -> OpenCV camera coordinates under the given convention."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if convention == "waymo":
        return xyz @ M_CV2EGO  # (-y, -z, x)
    if convention == "opencv":
        return xyz
    raise ValueError(f"unknown convention {convention!r}")


def project_opencv(p_cv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 3) OpenCV camera points -> (N, 2) pixels (NaN where the depth is not positive)."""
    K = np.asarray(K, dtype=np.float64)
    z = p_cv[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p_cv[:, 0] / z + K[0, 1] * p_cv[:, 1] / z + K[0, 2]
        v = K[1, 1] * p_cv[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], 1)
    uv[z <= 1e-6] = np.nan
    return uv


def _parse_lane(ln: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One ``lane_lines`` entry -> (uv (n, 2), xyz (n, 3), visible (n,) bool); ValueError when malformed."""
    uv, xyz = np.asarray(ln["uv"], dtype=np.float64), np.asarray(ln["xyz"], dtype=np.float64)
    if uv.ndim != 2 or uv.shape[0] != 2 or xyz.ndim != 2 or xyz.shape[0] != 3 or uv.shape[1] != xyz.shape[1]:
        raise ValueError(f"uv {uv.shape} / xyz {xyz.shape} are not [2, n] / [3, n]")
    n = uv.shape[1]
    vis = np.asarray(ln.get("visibility", np.ones(n)), dtype=np.float64).reshape(-1)
    if len(vis) != n:
        raise ValueError(f"visibility has {len(vis)} entries for {n} points")
    return uv.T, xyz.T, vis > 0.5


def convention_errors(lane_lines: Sequence[dict], intrinsic: np.ndarray) -> Dict[str, float]:
    """Median pixel error between ``uv`` and the projection of ``xyz`` under each candidate convention (over the
    visible points, or all points when fewer than 2 are visible; inf without usable points)."""
    uvs, xyzs, viss = [], [], []
    for ln in lane_lines:
        try:
            uv, xyz, vis = _parse_lane(ln)
        except (KeyError, TypeError, ValueError):
            continue
        ok = np.isfinite(uv).all(1) & np.isfinite(xyz).all(1)
        uvs.append(uv[ok]), xyzs.append(xyz[ok]), viss.append(vis[ok])
    if not uvs:
        return {c: float("inf") for c in CONVENTIONS}
    uv, xyz, vis = np.concatenate(uvs), np.concatenate(xyzs), np.concatenate(viss)
    if vis.sum() >= 2:
        uv, xyz = uv[vis], xyz[vis]
    if len(uv) == 0:
        return {c: float("inf") for c in CONVENTIONS}
    out = {}
    for c in CONVENTIONS:
        proj = project_opencv(to_opencv(xyz, c), intrinsic)
        err = np.hypot(*(proj - uv).T)
        err[~np.isfinite(err)] = np.inf
        out[c] = float(np.median(err))
    return out


def detect_convention(lane_lines: Sequence[dict], intrinsic: np.ndarray, tol_px: float = CONVENTION_TOL_PX) -> str:
    """Which camera frame ``xyz`` is in: ``"waymo"`` (x forward, y left, z up: OpenCV ``(X, Y, Z) = (-y, -z, x)``) or
    ``"opencv"`` (x right, y down, z forward), by projecting through ``intrinsic`` and comparing with ``uv``.

    Raises ValueError with both median errors when neither reproduces ``uv`` within ``tol_px``."""
    errs = convention_errors(lane_lines, np.asarray(intrinsic, dtype=np.float64).reshape(3, 3))
    best = min(errs, key=errs.get)
    if not errs[best] <= tol_px:
        raise ValueError("xyz matches neither camera convention (median |proj(xyz) - uv|): "
                         + ", ".join(f"{c} {e:.2f} px" for c, e in errs.items()) + f"; tolerance {tol_px} px")
    return best


def lane_attrs(category: int, attribute: int, stats: ConvertStats) -> Tuple[int, int, int]:
    """(order, continuity, ltype) ids from the OpenLane ``attribute`` and ``category``."""
    order = ATTR_ORDER.get(int(attribute), LANE_ORDER_UNLABELLED)
    if int(category) not in CATEGORY_MAP:
        stats.skip("lane category unknown")
    ltype, cont = CATEGORY_MAP.get(int(category), ("other", None))
    return order, -1 if cont is None else LANE_CONTINUITY.index(cont), LANE_TYPES.index(ltype)


def _frame(anno: dict, hw: Tuple[int, int], lane_px: float, stats: ConvertStats, conventions: Counter
           ) -> Optional[dict]:
    K = np.asarray(anno["intrinsic"], dtype=np.float64).reshape(3, 3)
    raw: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    usable = []  # the lane_lines entries that parse, for the convention check
    for ln in anno.get("lane_lines", []):
        try:
            uv, xyz, vis = _parse_lane(ln)
        except (KeyError, TypeError, ValueError):
            stats.skip("lane uv/xyz malformed")
            continue
        usable.append(ln)
        finite = np.isfinite(uv).all(1) & np.isfinite(xyz).all(1)
        if not finite.all():
            stats.skip("lane point not finite", int((~finite).sum()))
        stats.skip("lane point not visible", int((~vis & finite).sum()))
        keep = vis & finite
        if keep.sum() < 2:
            stats.skip("lane < 2 visible points")
            continue
        raw.append((uv[keep], xyz[keep], int(ln.get("category", 0)), int(ln.get("attribute", 0))))
    if raw:
        convention = detect_convention(usable, K)
        conventions[convention] += 1
        p_ego = [to_opencv(xyz, convention) @ M_CV2EGO.T for _, xyz, _, _ in raw]  # camera-aligned, camera at origin
        allp = np.concatenate(p_ego)
        near = allp[allp[:, 0] < NEAR_M]
        if len(near) == 0:
            stats.skip("height from all points (none nearer than 15 m)")
            near = allp
        h = -float(np.median(near[:, 2]))
    else:
        ext = anno.get("extrinsic")
        h = float(np.asarray(ext, dtype=np.float64).reshape(4, 4)[2, 3]) if ext is not None else float("nan")
        if not np.isfinite(h):
            stats.skip_frame("no lanes and no extrinsic height")
            return None
        stats.skip("height from extrinsic (frame without lanes)")
        p_ego = []
    if not h > 0:
        stats.skip_frame("camera height not positive")
        return None
    lanes = []
    for (uv, _, cat, attr), pe in zip(raw, p_ego):
        ego = pe + (0.0, 0.0, h)
        o = np.argsort(ego[:, 0], kind="stable")  # near first
        order, cont, ltype = lane_attrs(cat, attr, stats)
        lanes.append(LaneInstance(uv[o], ego[o], order=order, continuity=cont, ltype=ltype))
    mask = np.zeros(hw, np.uint8)
    draw_polylines(mask, [l.points for l in lanes], 1, line_thickness(hw, lane_px))
    lm = LabelMask(lane_geom=True, lane_3d=True, lane_order=True, lane_cont=True, lane_type=True, lane_mask=True)
    return dict(label_mask=lm, lane_mask=mask, lanes=lanes, K=K, cam2ego=ground_cam2ego(h), calib="measured")


def _anno_root(args) -> Path:
    root = Path(args.root)
    if args.anno:
        p = Path(args.anno)
        return p if p.is_absolute() or p.exists() else root / p
    for d in ANNO_DIRS:
        if (root / d).is_dir():
            return root / d
    raise FileNotFoundError(f"no {' / '.join(ANNO_DIRS)} under {root} (use --anno)")


def convert(args: argparse.Namespace) -> ConvertStats:
    root, anno_root = Path(args.root), _anno_root(args)
    w = Writer(args.dst, "openlane", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    conventions: Counter = Counter()
    for split in args.splits:
        sub = anno_root / SPLIT_DIRS.get(split, split)
        jsons = sorted(sub.rglob("*.json")) if sub.is_dir() else []
        if not jsons:
            LOG.warning("%s: no annotation files under %s", split, sub)
            continue
        LOG.info("%s: %d annotation files under %s", split, len(jsons), sub)
        count, frame_in_clip = 0, Counter()
        for js in jsons:
            if args.limit is not None and count >= args.limit:
                break
            try:
                with open(js, "r", encoding="utf-8") as f:
                    anno = json.load(f)
                np.asarray(anno["intrinsic"], dtype=np.float64).reshape(3, 3)  # must exist and be 3x3
            except (OSError, ValueError, KeyError, TypeError):
                w.stats.skip_frame("annotation unreadable")
                continue
            segment = js.parent.name
            img = root / "images" / str(anno.get("file_path") or "")
            if not anno.get("file_path") or not img.exists():
                img = root / "images" / SPLIT_DIRS.get(split, split) / segment / (js.stem + ".jpg")
            hw = image_size(img)
            if hw is None:
                w.stats.skip_frame("image missing")
                continue
            try:
                fields = _frame(anno, hw, args.lane_px, w.stats, conventions)
            except ValueError as e:
                LOG.warning("%s: %s", js, e)
                w.stats.skip_frame("convention undetermined")
                continue
            if fields is None:
                continue
            idx = frame_in_clip[segment]
            frame_in_clip[segment] += 1
            if w.add(split, f"{segment}__{js.stem}", img, hw=hw, clip_id=segment, frame_idx=idx, **fields) is not None:
                count += 1
    return w.finish({"xyz_convention": dict(conventions)})
