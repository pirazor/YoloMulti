"""CULane -> SPARC samples: lane polylines with their ego-relative slot order and a rasterised lane mask.

Expected layout under ``--root`` (the official download, extracted in place):

* ``list/train.txt``, ``list/val.txt``, ``list/test.txt``: one image per line as
  ``/driver_23_30frame/05151649_0422.MP4/00000.jpg`` (leading slash, relative to the root);
* ``list/train_gt.txt``, ``list/val_gt.txt``: ``<image> <seg label png> <l2> <l1> <r1> <r2>``, four 0/1 flags saying
  whether the left-left, left, right and right-right lane slot exists (the test split has no gt file);
* ``<image stem>.lines.txt`` next to every image: one lane per line, ``x1 y1 x2 y2 ...`` in pixels of the 1640 x 590
  image, sampled at fixed rows (bottom-up, every 10 px);
* ``laneseg_label_w16/<same path>.png`` (optional, ``--seg``): the official per-pixel labels, 0 background, 1..4 slot.

Assumptions (CULane ships no format document beyond the project page; these follow the reference loaders):

* The lines file lists only the existing lanes, in slot order, so the i-th line is the i-th set flag of the gt file:
  ``order`` = -2 / -1 / +1 / +2 with ``weak=False``. When the counts disagree (``lane flags mismatch``) or the split
  has no gt file (test), the order is derived from each lane's x at its lowest point relative to the image centre
  (nearest left = -1, next = -2, right likewise) and the lane is marked ``weak``.
* Points with x or y outside the image (CULane extrapolates the spline past the border, giving negative x) are
  dropped (``lane point outside image``); a lane left with < 2 points is skipped (``lane < 2 points``).
* Point order in the file is not trusted: points are sorted bottom (near) to top (far).
* ``lane_mask`` is the rasterised polylines (``--lane-px`` thickness at 1280 px width); with ``--seg`` the official
  PNG is used instead (any value > 0 -> 1), falling back to rasterising when a PNG is missing.
* CULane has no calibration: a nominal camera (``--hfov``, ``--cam-height``, ``--horizon``) and ``calib="nominal"``.
* Every listed frame is fully annotated, so a frame without lanes is a true negative (written with an empty mask).

Labels: ``lane_geom``, ``lane_order``, ``lane_mask``. Output stems are the image path with ``/`` -> ``__`` and
``.`` -> ``_`` (``driver_23_30frame__05151649_0422_MP4__00000``); ``clip_id`` is the clip directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparc.data.converters.base import (ConvertStats, Writer, add_common_args, image_size, nominal_camera,
                                        nominal_camera_args)
from sparc.data.raster import draw_polylines, line_thickness
from sparc.data.schema import LANE_ORDER_RANGE, LANE_ORDER_UNLABELLED, LabelMask, LaneInstance

LOG = logging.getLogger("sparc.convert.culane")

SLOT_ORDER = (-2, -1, 1, 2)  # the four CULane slots: left-left, left, right, right-right


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="CULane root (list/, driver_*_*frame/, laneseg_label_w16/)")
    p.add_argument("--seg", default=None,
                   help="directory of the official per-pixel labels (laneseg_label_w16) to use as the lane mask; "
                        "default: rasterise the polylines")
    p.add_argument("--lane-px", type=float, default=8.0, help="lane line thickness at 1280 px width")
    add_common_args(p)
    nominal_camera_args(p, hfov=60.0, height=1.5, horizon=0.45)


def read_list(path: Path) -> List[str]:
    """Image paths of a ``list/<split>.txt`` (first token of every non-empty line)."""
    return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.split()]


def read_gt(path: Path, stats: Optional[ConvertStats] = None) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """``list/<split>_gt.txt`` -> image path -> (seg label path, four existence flags)."""
    out: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 2 + len(SLOT_ORDER) or any(p not in ("0", "1") for p in parts[2:2 + len(SLOT_ORDER)]):
            if stats is not None:
                stats.skip("gt line malformed")
            continue
        out[parts[0]] = (parts[1], tuple(int(p) for p in parts[2:2 + len(SLOT_ORDER)]))
    return out


def read_lines(path: Path, stats: ConvertStats) -> List[np.ndarray]:
    """``<stem>.lines.txt`` -> one (M, 2) float array per line, in file order (M may be < 2)."""
    lanes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            stats.skip("lane line unparsable")
            continue
        if len(vals) % 2:
            stats.skip("lane odd coordinate count")
            vals = vals[:-1]
        lanes.append(np.asarray(vals, dtype=np.float64).reshape(-1, 2))
    return lanes


def orders_from_x(x_bottom: Sequence[float], cx: float) -> List[int]:
    """Slot order from the x of each lane's lowest point: nearest left of ``cx`` = -1, next = -2, right likewise;
    beyond ``LANE_ORDER_RANGE`` -> unlabelled."""
    orders = [LANE_ORDER_UNLABELLED] * len(x_bottom)
    left = sorted((i for i, x in enumerate(x_bottom) if x < cx), key=lambda i: -x_bottom[i])
    right = sorted((i for i, x in enumerate(x_bottom) if x >= cx), key=lambda i: x_bottom[i])
    for side, sign in ((left, -1), (right, 1)):
        for k, i in enumerate(side):
            orders[i] = sign * (k + 1) if k < LANE_ORDER_RANGE else LANE_ORDER_UNLABELLED
    return orders


def _lanes(raw: List[np.ndarray], flags: Optional[Tuple[int, ...]], hw: Tuple[int, int], stats: ConvertStats
           ) -> List[LaneInstance]:
    h, w = hw
    # drop points outside the image, then lanes with < 2 points (by raw index, so the slot mapping holds)
    pts: List[Optional[np.ndarray]] = []
    for p in raw:
        inside = np.isfinite(p).all(1) & (p[:, 0] >= -0.5) & (p[:, 0] <= w - 0.5) & (p[:, 1] >= -0.5) & \
            (p[:, 1] <= h - 0.5)
        if not inside.all():
            stats.skip("lane point outside image", int((~inside).sum()))
        p = p[inside]
        if len(p) < 2:
            stats.skip("lane < 2 points")
            pts.append(None)
        else:
            pts.append(p[np.argsort(-p[:, 1], kind="stable")])  # bottom (near) first
    weak = flags is None
    if flags is not None:
        slots = [o for o, f in zip(SLOT_ORDER, flags) if f]
        if len(slots) == len(raw):
            orders = slots
        else:
            stats.skip("lane flags mismatch")
            weak = True
    if weak:
        orders = orders_from_x([p[0, 0] if p is not None else np.nan for p in pts], (w - 1) / 2.0)
    return [LaneInstance(p, order=o, weak=weak) for p, o in zip(pts, orders) if p is not None]


def _seg_mask(seg_dir: Path, rel: Path, hw: Tuple[int, int], stats: ConvertStats) -> Optional[np.ndarray]:
    png = seg_dir / rel.with_suffix(".png")
    if not png.exists():
        stats.skip("seg png missing")
        return None
    seg = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if seg is None:
        stats.skip("seg png unreadable")
        return None
    if seg.ndim == 3:
        seg = seg.max(axis=2)
    if seg.shape[:2] != tuple(hw):
        stats.skip("seg png size mismatch")
        return None
    return (seg > 0).astype(np.uint8)


def convert(args: argparse.Namespace) -> ConvertStats:
    root = Path(args.root)
    seg_dir = Path(args.seg) if args.seg else None
    w = Writer(args.dst, "culane", args.licence, copy_images=args.copy, overwrite=args.overwrite, limit=args.limit)
    lm = LabelMask(lane_geom=True, lane_order=True, lane_mask=True)
    for split in args.splits:
        list_path, gt_path = root / "list" / f"{split}.txt", root / "list" / f"{split}_gt.txt"
        gt = read_gt(gt_path, w.stats) if gt_path.exists() else {}
        names = read_list(list_path) if list_path.exists() else list(gt)
        if not names:
            LOG.warning("%s: no list/%s.txt or list/%s_gt.txt (or empty)", root, split, split)
            continue
        LOG.info("%s: %d images, existence flags %s", split, len(names), "yes" if gt else "no (orders derived, weak)")
        count = 0
        for name in names:
            if args.limit is not None and count >= args.limit:
                break
            rel = Path(name.lstrip("/"))
            img = root / rel
            hw = image_size(img)
            if hw is None:
                w.stats.skip_frame("image missing")
                continue
            lines = img.with_suffix(".lines.txt")
            if not lines.exists():
                w.stats.skip_frame("lines.txt missing")
                continue
            flags = gt[name][1] if name in gt else None
            lanes = _lanes(read_lines(lines, w.stats), flags, hw, w.stats)
            mask = _seg_mask(seg_dir, rel, hw, w.stats) if seg_dir is not None else None
            if mask is None:
                mask = np.zeros(hw, np.uint8)
                draw_polylines(mask, [l.points for l in lanes], 1, line_thickness(hw, args.lane_px))
            K, T, calib = nominal_camera(args, hw)
            stem = rel.with_suffix("").as_posix().replace("/", "__").replace(".", "_")  # no dots: .npz suffix
            frame_idx = int(rel.stem) if rel.stem.isdigit() else 0
            if w.add(split, stem, img, lm, lane_mask=mask, hw=hw, K=K, cam2ego=T, calib=calib, lanes=lanes,
                     clip_id=rel.parent.as_posix(), frame_idx=frame_idx) is not None:
                count += 1
    return w.finish()
