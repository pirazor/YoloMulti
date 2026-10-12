"""Camera / ego geometry shared by converters, the dataset, the lane head and the evaluators.

Camera frame x right, y down, z forward; ego frame x forward, y left, z up with the origin on the ground below
the camera (``cam2ego``). Lanes are learned in the ego frame at fixed forward distances (:data:`LANE_X_ANCHORS`):
a 2D-only source supplies them through the ground-plane lift (flat road, height from the sample's ``cam2ego``), a
3D source supplies them directly.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from sparc.data.schema import LANE_X_ANCHORS, LaneInstance


def rays_from_pixels(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 2) pixels -> (N, 3) camera-frame directions with z = 1."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], 1)


def lift_to_ground(uv: np.ndarray, K: np.ndarray, cam2ego: np.ndarray, z_ground: float = 0.0,
                   max_range: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect pixel rays with the ego plane ``z = z_ground``.

    Returns ``(xy (N, 2) ego metres, valid (N,))``; a ray is invalid when it does not hit the ground in front of
    the camera within ``max_range``."""
    T = np.asarray(cam2ego, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    d = rays_from_pixels(uv, K) @ R.T  # directions in the ego frame
    dz = d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z_ground - t[2]) / dz
    valid = (dz < -1e-9) & (s > 0) & np.isfinite(s)
    P = t[None] + s[:, None] * d
    valid &= P[:, 0] > 0
    valid &= np.hypot(P[:, 0], P[:, 1]) <= max_range
    return P[:, :2], valid


def project_ego(xyz: np.ndarray, K: np.ndarray, cam2ego: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(N, 3) ego points -> ``(uv (N, 2) pixels, z_cam (N,))``; points with z_cam <= 0 get NaN pixels."""
    T = np.asarray(cam2ego, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    P = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pc = (P - T[:3, 3]) @ T[:3, :3]  # R^T (P - t)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], 1)
    uv[z <= 1e-6] = np.nan
    return uv, z


def _monotone(xs: np.ndarray, *cols: np.ndarray):
    """Sort by x and average duplicate x so ``np.interp`` is well defined."""
    order = np.argsort(xs, kind="stable")
    xs = xs[order]
    cols = [c[order] for c in cols]
    ux, inv = np.unique(xs, return_inverse=True)
    counts = np.bincount(inv, minlength=len(ux))
    out = [np.bincount(inv, weights=c, minlength=len(ux)) / counts for c in cols]
    return ux, out


def lane_anchor_targets(lane: LaneInstance, K: np.ndarray, cam2ego: np.ndarray,
                        anchors: np.ndarray = LANE_X_ANCHORS, prefer_3d: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lateral offset ``y``, height ``z`` and visibility at the forward anchors for one lane.

    3D sources use ``points3d``; otherwise the 2D polyline is lifted to the ground plane. Anchors outside the
    lane's forward extent are not visible (y, z = NaN)."""
    if prefer_3d and lane.points3d is not None:
        xyz = lane.points3d.astype(np.float64)
        xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    else:
        xy, valid = lift_to_ground(lane.points, K, cam2ego)
        xs, ys = xy[valid, 0], xy[valid, 1]
        zs = np.zeros_like(xs)
    n = len(anchors)
    y, z, vis = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool)
    if len(xs) < 2:
        return y, z, vis
    ux, (uy, uz) = _monotone(xs, ys, zs)
    if len(ux) < 2:
        return y, z, vis
    vis = (anchors >= ux[0]) & (anchors <= ux[-1])
    y[vis] = np.interp(anchors[vis], ux, uy)
    z[vis] = np.interp(anchors[vis], ux, uz)
    return y, z, vis


def anchors_to_polyline(y: np.ndarray, z: np.ndarray, vis: np.ndarray, K: np.ndarray, cam2ego: np.ndarray,
                        hw: Sequence[int], anchors: np.ndarray = LANE_X_ANCHORS, margin: float = 0.0
                        ) -> np.ndarray:
    """Visible anchor points -> (M, 2) image polyline (near to far), points outside the image dropped."""
    sel = np.asarray(vis, dtype=bool)
    if not sel.any():
        return np.zeros((0, 2), np.float32)
    xyz = np.stack([anchors[sel], np.asarray(y)[sel], np.asarray(z)[sel]], 1)
    uv, zc = project_ego(xyz, K, cam2ego)
    h, w = hw
    ok = (zc > 0.5) & np.isfinite(uv).all(1) & (uv[:, 0] >= -margin) & (uv[:, 0] <= w - 1 + margin) & \
         (uv[:, 1] >= -margin) & (uv[:, 1] <= h - 1 + margin)
    return uv[ok].astype(np.float32)


def resample_by_rows(pts: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """x of a polyline at the given image rows (NaN where the polyline does not cover the row)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return np.full(len(rows), np.nan)
    vy, (vx,) = _monotone(p[:, 1], p[:, 0])
    out = np.full(len(rows), np.nan)
    inside = (rows >= vy[0]) & (rows <= vy[-1])
    out[inside] = np.interp(rows[inside], vy, vx)
    return out
