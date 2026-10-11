"""Rasterising polygons / polylines into class-id masks (drivable area, lane markings)."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


def line_thickness(image_hw: Sequence[int], ref_px: float = 8.0) -> int:
    """Lane line thickness: ``ref_px`` for a 1280 px long side, scaled with the long side (min 2 px)."""
    return max(2, int(round(ref_px * max(image_hw) / 1280)))


def bezier(p0, p1, p2, p3, n: int = 16) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def flatten_poly2d(vertices: Sequence[Sequence[float]], types: str = "", closed: bool = False) -> np.ndarray:
    """BDD100K / Scalabel ``poly2d``: vertices typed ``L`` (line) or ``C`` (bezier control point; two ``C`` between
    two ``L`` form a cubic segment) -> dense (M, 2) polyline."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    if not types or "C" not in types:
        return v.astype(np.float32)
    types = (types + "L" * len(v))[: len(v)]
    pts, i = [], 0
    if closed:
        v = np.concatenate([v, v[:1]])
        types = types + "L"
    while i < len(v):
        if types[i] == "L":
            pts.append(v[i])
            if i + 3 < len(v) and types[i + 1] == "C" and types[i + 2] == "C" and types[i + 3] == "L":
                pts.extend(bezier(v[i], v[i + 1], v[i + 2], v[i + 3])[1:])
                i += 3
                continue
        i += 1
    if len(pts) < 2:
        return v.astype(np.float32)
    return np.asarray(pts, dtype=np.float32)


def fill_polygons(mask: np.ndarray, polys: Iterable[np.ndarray], value: int) -> None:
    for p in polys:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
        if len(p) >= 3:
            cv2.fillPoly(mask, [np.round(p).astype(np.int32)], int(value))


def draw_polylines(mask: np.ndarray, lines: Iterable[np.ndarray], value: int, thickness: int) -> None:
    for p in lines:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
        if len(p) >= 2:
            cv2.polylines(mask, [np.round(p).astype(np.int32)], False, int(value), int(thickness), lineType=cv2.LINE_8)
