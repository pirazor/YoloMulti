"""Synthetic road scenes written directly as SPARC samples: exact geometry for tests and the learning test.

A flat ground, 2-5 parallel lane markings with a slight curvature (white / yellow, solid / dashed; a double yellow
marks opposing traffic), the drivable area (ego lane = direct, same-direction neighbours = alternative), cars /
trucks / buses standing in lanes (3D boxes on the ground), pedestrians at the road side, traffic lights with a
colour and signs with a fine class drawn as distinct shapes. Everything is rendered by projecting ego-frame
geometry through a nominal camera, so every label can be checked against the drawing. With ``partial=True`` some
frames lack the drivable-area, lane or element labels (the label mask says so), as public sources do.

    python -m sparc synth --dst /tmp/syn --n-train 200 --n-val 40
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sparc.data.converters.base import Writer
from sparc.data.geometry import project_ego
from sparc.data.raster import draw_polylines, fill_polygons, line_thickness
from sparc.data.schema import (AGENT_CLASSES, EL_LIGHT, EL_SIGN, ELEMENT_ATTRS, LANE_CONTINUITY, LANE_DIRECTIONS,
                               LANE_TYPES, LabelMask, LaneInstance, ground_cam2ego)
from sparc.data.virtual_camera import nominal_K

LOG = logging.getLogger("sparc.synthetic")
SIGN_NAMES = ("syn:stop", "syn:yield", "syn:info", "syn:arrow_left", "syn:arrow_right")
SIGN_COLORS = {"syn:stop": (220, 30, 30), "syn:yield": (240, 210, 40), "syn:info": (40, 160, 70),
               "syn:arrow_left": (240, 240, 240), "syn:arrow_right": (240, 240, 240)}
LIGHT_COLORS = {"red": (235, 40, 40), "yellow": (240, 200, 30), "green": (40, 220, 80)}
AGENT_COLORS = {"car": (40, 70, 200), "truck": (150, 30, 30), "bus": (230, 130, 20), "pedestrian": (200, 110, 210)}
AGENT_SIZE = {"car": (4.5, 1.8, 1.5), "truck": (8.0, 2.5, 3.2), "bus": (12.0, 2.5, 3.0), "pedestrian": (0.5, 0.6, 1.75)}
LANE_W = 0.15


@dataclass
class Scene:
    lanes: List[LaneInstance]
    lane_curves: List[Tuple[np.ndarray, np.ndarray]]  # (x, y) ego samples per lane
    agents: list                                      # (name, (x, y, l, w, h))
    lights: list                                      # (x, y, z, colour)
    signs: list                                       # (x, y, z, name)
    direct: np.ndarray                                # polygons in ego (x, y)
    alternative: List[np.ndarray]


def _curve(offset: float, c: float, xs: np.ndarray) -> np.ndarray:
    return offset + c * (xs - 3.0) ** 2 / 60.0


def make_scene(rng: np.random.Generator) -> Scene:
    xs = np.linspace(3.0, 60.0, 58)
    c = rng.uniform(-0.6, 0.6)
    n_left, n_right = int(rng.integers(1, 3)), int(rng.integers(1, 3))  # markings on each side of the ego lane
    offsets = sorted([1.75 + 3.5 * k for k in range(n_left)] + [-(1.75 + 3.5 * k) for k in range(n_right)], reverse=True)
    yellow_at = 1.75 + 3.5 * (n_left - 1) if n_left >= 1 and rng.random() < 0.6 else None  # outermost left = opposing
    lanes, curves = [], []
    for off in offsets:
        ys = _curve(off, c, xs)
        curves.append((xs, ys))
        if off == yellow_at:
            ltype, cont, direction = "double_yellow", "solid", "opposing"
        elif off == max(offsets) or off == min(offsets):
            ltype, cont, direction = ("single_white", "solid", "boundary")
        else:
            ltype, cont, direction = ("single_white", "dashed", "same")
        k = int(round((abs(off) - 1.75) / 3.5)) + 1  # 1 = the ego lane's own boundary, 2 = the next lane's ...
        order = -k if off > 0 else k                   # left (ego y > 0) is negative
        lanes.append(LaneInstance(np.zeros((2, 2)), order=order, direction=LANE_DIRECTIONS.index(direction),
                                  continuity=LANE_CONTINUITY.index(cont), ltype=LANE_TYPES.index(ltype)))
    # drivable area: ego lane between the two innermost markings; alternative = other same-direction lanes
    def region(y_hi, y_lo):
        top = np.stack([xs, _curve(y_hi, c, xs)], 1)
        bot = np.stack([xs, _curve(y_lo, c, xs)], 1)[::-1]
        return np.concatenate([top, bot])
    direct = region(1.75, -1.75)
    alternative = []
    for off in offsets:
        if off > 1.75 and off != yellow_at and (yellow_at is None or off < yellow_at):
            alternative.append(region(off, off - 3.5))
        if off < -1.75:
            alternative.append(region(off + 3.5, off))
    # agents in lanes, far first
    lane_centres = [0.0] + [off - 1.75 for off in offsets if off > 1.75] + [off + 1.75 for off in offsets if off < -1.75]
    agents = []
    for _ in range(int(rng.integers(1, 5))):
        name = rng.choice(["car", "car", "car", "truck", "bus"])
        l, w, h = AGENT_SIZE[name]
        x = float(rng.uniform(8, 50))
        yc = float(rng.choice(lane_centres))
        agents.append((name, (x, _curve(yc, c, np.array([x]))[0], l, w, h)))
    for _ in range(int(rng.integers(0, 3))):
        side = max(offsets) + 1.0 if rng.random() < 0.5 else min(offsets) - 1.0
        x = float(rng.uniform(6, 30))
        agents.append(("pedestrian", (x, _curve(side, c, np.array([x]))[0], *AGENT_SIZE["pedestrian"])))
    agents.sort(key=lambda a: -a[1][0])
    lights = [(float(rng.uniform(15, 45)), float(rng.choice([-1, 1]) * rng.uniform(2.5, 6)), float(rng.uniform(4.5, 5.5)),
               str(rng.choice(list(LIGHT_COLORS)))) for _ in range(int(rng.integers(0, 3)))]
    signs = [(float(rng.uniform(8, 40)), float(rng.choice([-1, 1]) * rng.uniform(3, 5)), float(rng.uniform(2, 3)),
              str(rng.choice(SIGN_NAMES))) for _ in range(int(rng.integers(0, 3)))]
    return Scene(lanes, curves, agents, lights, signs, direct, alternative)


def _box_corners(x, y, l, w, h, yaw=0.0) -> np.ndarray:
    dx, dy = l / 2, w / 2
    base = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = base @ R.T + [x, y]
    return np.concatenate([np.concatenate([xy, np.zeros((4, 1))], 1), np.concatenate([xy, np.full((4, 1), h)], 1)])


def _poly_px(poly_ego: np.ndarray, K, T, z: float = 0.0) -> Optional[np.ndarray]:
    xyz = np.concatenate([poly_ego, np.full((len(poly_ego), 1), z)], 1)
    uv, zc = project_ego(xyz, K, T)
    ok = zc > 0.5
    return uv[ok] if ok.sum() >= 3 else None


def render(scene: Scene, K: np.ndarray, T: np.ndarray, hw: Tuple[int, int], rng: np.random.Generator):
    """Returns (image RGB, da_mask, lane_mask, sample fields)."""
    H, W = hw
    img = np.zeros((H, W, 3), np.uint8)
    cy = int(round(K[1, 2]))
    img[:cy] = (150, 190, 240)
    img[cy:] = (90, 140, 70)
    road = _poly_px(np.concatenate([np.stack([scene.lane_curves[0][0], scene.lane_curves[0][1] + 1.0], 1),
                                    np.stack([scene.lane_curves[-1][0], scene.lane_curves[-1][1] - 1.0], 1)[::-1]]), K, T)
    if road is not None:
        cv2.fillPoly(img, [np.round(road).astype(np.int32)], (85, 85, 88))
    # lanes
    lanes_out = []
    for lane, (xs, ys) in zip(scene.lanes, scene.lane_curves):
        xyz = np.stack([xs, ys, np.zeros_like(xs)], 1)
        uv, zc = project_ego(xyz, K, T)
        ok = zc > 0.5
        uv, zc = uv[ok], zc[ok]
        colour = (235, 235, 235) if "white" in LANE_TYPES[lane.ltype] else (225, 200, 40)
        dashed = LANE_CONTINUITY[lane.continuity] == "dashed"
        for i in range(len(uv) - 1):
            if dashed and (i // 3) % 2:
                continue
            t = max(1, int(round(K[0, 0] * LANE_W / zc[i])))
            if "double" in LANE_TYPES[lane.ltype]:
                t = max(2, 2 * t)
            cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[i + 1]).astype(int)), colour, t)
        lanes_out.append(LaneInstance(uv[::-1], points3d=xyz[ok][::-1], order=lane.order, direction=lane.direction,
                                      continuity=lane.continuity, ltype=lane.ltype))
    # masks
    da = np.zeros((H, W), np.uint8)
    for poly in scene.alternative:
        p = _poly_px(poly, K, T)
        if p is not None:
            fill_polygons(da, [p], 2)
    p = _poly_px(scene.direct, K, T)
    if p is not None:
        fill_polygons(da, [p], 1)
    ll = np.zeros((H, W), np.uint8)
    draw_polylines(ll, [l.points for l in lanes_out], 1, line_thickness(hw))
    # agents (far first)
    boxes, cls, box3d = [], [], []
    for name, (x, y, l, w, h) in scene.agents:
        uv, zc = project_ego(_box_corners(x, y, l, w, h), K, T)
        if (zc <= 0.5).any():
            continue
        x1, y1 = uv.min(0)
        x2, y2 = uv.max(0)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cv2.rectangle(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), AGENT_COLORS[name], -1)
        boxes.append([x1, y1, x2, y2])
        cls.append(AGENT_CLASSES.index(name))
        pc = np.linalg.inv(T) @ np.array([x, y, h / 2, 1.0])
        box3d.append([pc[2], 0.0, l, w, h, x, y])
    # elements
    el_boxes, el_cls, el_attr, el_sign = [], [], [], []
    for x, y, z, colour in scene.lights:
        uv, zc = project_ego(_box_corners(x, y, 0.4, 0.4, 1.2)[:, :] + [0, 0, z], K, T)
        x1, y1 = uv.min(0)
        x2, y2 = uv.max(0)
        if (zc <= 0.5).any() or x1 < 0 or y1 < 0 or x2 > W - 1 or y2 > H - 1 or x2 - x1 < 3:
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (20, 20, 20), -1)
        r = max(1, int((x2 - x1) / 3))
        cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), r, LIGHT_COLORS[colour], -1)
        el_boxes.append([x1, y1, x2, y2])
        el_cls.append(EL_LIGHT)
        el_attr.append(ELEMENT_ATTRS.index(colour))
        el_sign.append("")
    for x, y, z, name in scene.signs:
        uv, zc = project_ego(_box_corners(x, y, 0.1, 0.8, 0.8) + [0, 0, z], K, T)
        x1, y1 = uv.min(0)
        x2, y2 = uv.max(0)
        if (zc <= 0.5).any() or x1 < 0 or y1 < 0 or x2 > W - 1 or y2 > H - 1 or x2 - x1 < 4:
            continue
        c = SIGN_COLORS[name]
        cx_, cy_, rad = int((x1 + x2) / 2), int((y1 + y2) / 2), int((x2 - x1) / 2)
        if name == "syn:stop":
            cv2.circle(img, (cx_, cy_), rad, c, -1)
        elif name == "syn:yield":
            cv2.fillPoly(img, [np.array([[cx_ - rad, cy_ - rad], [cx_ + rad, cy_ - rad], [cx_, cy_ + rad]])], c)
        elif name == "syn:info":
            cv2.rectangle(img, (cx_ - rad, cy_ - rad), (cx_ + rad, cy_ + rad), c, -1)
        else:  # chiral arrows: a triangle pointing left or right on a blue square
            cv2.rectangle(img, (cx_ - rad, cy_ - rad), (cx_ + rad, cy_ + rad), (30, 60, 200), -1)
            tip = cx_ - rad if name.endswith("left") else cx_ + rad
            back = cx_ + rad // 2 if name.endswith("left") else cx_ - rad // 2
            cv2.fillPoly(img, [np.array([[tip, cy_], [back, cy_ - rad // 2], [back, cy_ + rad // 2]])], c)
        el_boxes.append([x1, y1, x2, y2])
        el_cls.append(EL_SIGN)
        el_attr.append(-1)
        el_sign.append(name)
    noise = rng.integers(-6, 7, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    fields = dict(boxes2d=np.array(boxes, np.float32).reshape(-1, 4), cls=cls, box3d=np.array(box3d, np.float32).reshape(-1, 7),
                  el_boxes=np.array(el_boxes, np.float32).reshape(-1, 4), el_cls=el_cls, el_attr=el_attr, el_sign=el_sign,
                  lanes=lanes_out)
    return img, da, ll, fields


def make_dataset(dst: str | Path, n_train: int = 64, n_val: int = 16, seed: int = 0, hw: Tuple[int, int] = (720, 1280),
                 partial: bool = True, overwrite: bool = False) -> Path:
    dst = Path(dst)
    w = Writer(dst, "synthetic", "own", copy_images=True, overwrite=overwrite)
    rng = np.random.default_rng(seed)
    K = nominal_K(hw, hfov_deg=65.0)
    T = ground_cam2ego(1.5, 0.0)
    tmp = dst / "_render"
    tmp.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            scene = make_scene(rng)
            img, da, ll, fields = render(scene, K, T, hw, rng)
            has_da = not (partial and rng.random() < 0.2)
            has_lanes = not (partial and rng.random() < 0.2)
            has_el = not (partial and rng.random() < 0.1)
            if not has_el:
                fields.update(el_boxes=np.zeros((0, 4), np.float32), el_cls=[], el_attr=[], el_sign=[])
            if not has_lanes:
                fields["lanes"] = []
            lm = LabelMask(agents=True, cls_vocab=range(len(AGENT_CLASSES)), box3d=bool(len(fields["boxes2d"])),
                           elements=has_el, el_vocab=(EL_LIGHT, EL_SIGN) if has_el else (), el_attr=has_el, el_sign=has_el,
                           da=has_da, lane_geom=has_lanes, lane_3d=has_lanes, lane_order=has_lanes, lane_dir=has_lanes,
                           lane_cont=has_lanes, lane_type=has_lanes, lane_mask=has_lanes)
            stem = f"{split}_{i:05d}"
            path = tmp / f"{stem}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            w.add(split, stem, path, lm, da_mask=da if has_da else None, lane_mask=ll if has_lanes else None, hw=hw,
                  K=K, cam2ego=T, calib="nominal", **fields)
            path.unlink()
    tmp.rmdir()
    w.finish()
    return dst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dst", required=True)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hw", type=int, nargs=2, default=[720, 1280])
    p.add_argument("--no-partial", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    make_dataset(a.dst, a.n_train, a.n_val, a.seed, tuple(a.hw), not a.no_partial, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
