import math

import numpy as np
import pytest

from sparc.data import virtual_camera as vc
from sparc.data.geometry import anchors_to_polyline, lane_anchor_targets, lift_to_ground, project_ego, resample_by_rows
from sparc.data.schema import LANE_X_ANCHORS, LaneInstance, ground_cam2ego


def test_virtual_camera_constants():
    cam = vc.VirtualCamera()
    assert cam.K[0, 0] == pytest.approx(320 / math.tan(math.radians(30)), rel=1e-6)
    assert cam.K[0, 2] == 319.5 and cam.K[1, 2] == 191.5 and 37 < cam.vfov_deg < 39
    Kn = vc.nominal_K((720, 1280), hfov_deg=65, horizon_row=330)
    assert Kn[0, 2] == 639.5 and Kn[1, 2] == 330 and Kn[0, 0] == pytest.approx(640 / math.tan(math.radians(32.5)))


def test_intrinsics_affine_matches_projection():
    """Warping a pixel with K_v K_s^-1 equals projecting the same ray with K_v."""
    rng = np.random.default_rng(0)
    Ks = vc.nominal_K((720, 1280), hfov_deg=70, horizon_row=300)
    Kv = vc.VirtualCamera().K
    A = vc.intrinsics_affine(Ks, Kv)
    uv = rng.uniform(0, (1279, 719), (50, 2))
    rays = np.concatenate([(uv[:, :1] - Ks[0, 2]) / Ks[0, 0], (uv[:, 1:] - Ks[1, 2]) / Ks[1, 1], np.ones((50, 1))], 1)
    want = (rays @ Kv.T)[:, :2]
    np.testing.assert_allclose(vc.warp_points(uv, A), want, atol=1e-3)
    np.testing.assert_allclose(vc.apply_to_K(A, Ks), Kv, atol=1e-9)
    with pytest.raises(ValueError):
        vc.intrinsics_affine(np.array([[1, 0, 0], [0, 1, 0], [0.1, 0, 1]]), Kv)


def test_warps_are_consistent_for_image_boxes_points_and_masks():
    img = np.zeros((100, 160, 3), np.uint8)
    img[40:60, 50:90] = 255
    A = vc.compose(vc.hflip(160), vc.scale_translate(1.5, 10, -5, about=(79.5, 49.5)))
    out = vc.warp_image(img, A, (100, 160))
    boxes, keep = vc.warp_boxes(np.array([[50, 40, 90, 60]]), A, (100, 160))
    assert keep.all()
    x1, y1, x2, y2 = boxes[0]
    inside = out[int(y1) + 2:int(y2) - 1, int(x1) + 2:int(x2) - 1]
    assert (inside == 255).mean() > 0.95 and (out[:, :, 0] == 255).sum() < 1.2 * inside[:, :, 0].size
    # scale about the centre then flip: a corner point maps where the composed matrix says
    p = vc.warp_points([[50, 40]], A)[0]
    s = vc.warp_points([[50, 40]], vc.scale_translate(1.5, 10, -5, about=(79.5, 49.5)))[0]
    assert p[0] == pytest.approx(159 - s[0]) and p[1] == pytest.approx(s[1])
    mask = np.full((100, 160), 2, np.uint8)
    m = vc.warp_mask(mask, vc.scale_translate(0.5), (100, 160))
    assert m[10, 10] == 2 and m[90, 150] == 255  # uncovered area is ignore


def test_warp_boxes_drops_clipped_and_tiny_boxes():
    A = vc.scale_translate(1.0, -100, 0)
    boxes, keep = vc.warp_boxes(np.array([[0, 0, 50, 50], [120, 0, 160, 40], [90, 0, 101, 5]]), A, (100, 160))
    assert keep.tolist() == [False, True, False]
    assert boxes[1].tolist() == [20, 0, 60, 40]


def test_lift_and_project_are_inverse():
    rng = np.random.default_rng(1)
    K = vc.VirtualCamera().K
    T = ground_cam2ego(1.4, 0.04, 0.01)
    xy = np.stack([rng.uniform(4, 80, 40), rng.uniform(-10, 10, 40)], 1)
    uv, z = project_ego(np.concatenate([xy, np.zeros((40, 1))], 1), K, T)
    assert (z > 0).all()
    back, valid = lift_to_ground(uv, K, T)
    assert valid.all()
    np.testing.assert_allclose(back, xy, atol=1e-6)
    # a pixel above the horizon never hits the ground
    _, v = lift_to_ground(np.array([[320.0, 10.0]]), K, T)
    assert not v.any()


def test_lane_anchor_targets_from_2d_and_3d():
    K = vc.VirtualCamera().K
    T = ground_cam2ego(1.5, 0.02)
    # a straight lane 1.75 m left of the ego from 4 m to 50 m
    xs = np.linspace(4, 50, 30)
    xyz = np.stack([xs, np.full_like(xs, 1.75), np.zeros_like(xs)], 1)
    uv, _ = project_ego(xyz, K, T)
    lane2d = LaneInstance(uv[::-1])  # near to far in the image = bottom to top
    y, z, vis = lane_anchor_targets(lane2d, K, T)
    inside = (LANE_X_ANCHORS >= 4) & (LANE_X_ANCHORS <= 50)
    assert vis.tolist() == inside.tolist()
    np.testing.assert_allclose(y[vis], 1.75, atol=1e-4)
    np.testing.assert_allclose(z[vis], 0.0, atol=1e-6)
    assert np.isnan(y[~vis]).all()
    # a 3D lane with height is used as is (not lifted)
    lane3d = LaneInstance(uv[::-1], points3d=np.stack([xs, np.full_like(xs, -1.5), 0.01 * xs], 1))
    y3, z3, v3 = lane_anchor_targets(lane3d, K, T)
    np.testing.assert_allclose(y3[v3], -1.5, atol=1e-6)
    np.testing.assert_allclose(z3[v3], 0.01 * LANE_X_ANCHORS[v3], atol=1e-6)
    # back to the image: the polyline from the anchors lies on the original lane
    poly = anchors_to_polyline(y, z, vis, K, T, (384, 640))
    assert len(poly) == vis.sum()
    xs_at = resample_by_rows(uv, poly[:, 1])
    np.testing.assert_allclose(poly[:, 0], xs_at, atol=0.05)


def test_lane_anchor_targets_degenerate_inputs():
    K = vc.VirtualCamera().K
    T = ground_cam2ego(1.5)
    y, z, vis = lane_anchor_targets(LaneInstance([[300, 50], [320, 60]]), K, T)  # above the horizon
    assert not vis.any() and np.isnan(y).all()
    y, z, vis = lane_anchor_targets(LaneInstance([[300, 300], [300, 300.0001]]), K, T)  # one distinct point
    assert not vis.any()
