from pathlib import Path

import numpy as np
import pytest

from sparc.data.schema import (AGENT_CLASSES, ATTR_FLIP, BDD_CLASSES, EL_LIGHT, EL_SIGN, ELEMENT_ATTRS, LabelMask,
                               LaneInstance, Sample, flip_cam2ego, ground_cam2ego)
from sparc.rig.stereo import StereoRig

CALIB = Path(__file__).parent / "data" / "stereo_calib_example.yaml"
K = np.array([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]])


def _sample(**kw):
    base = dict(image="img/a.jpg", hw=(720, 1280), K=K, cam2ego=ground_cam2ego(1.4, 0.02), source="bdd100k",
                licence="research", calib="nominal",
                label_mask=LabelMask(agents=True, cls_vocab=range(8), elements=True, el_vocab=(0, 1), el_attr=True,
                                     lane_geom=True, lane_dir=True, da=True, lane_mask=True),
                boxes2d=[[10, 20, 50, 80]], cls=[2],
                el_boxes=[[600, 100, 620, 140], [700, 90, 730, 120]], el_cls=[EL_LIGHT, EL_SIGN], el_attr=[1, -1],
                el_sign=["", "regulatory--stop--g1"],
                lanes=[LaneInstance([[100, 700], [400, 400]], order=-1, direction=1, continuity=0, ltype=1, weak=True)],
                da_mask="da/a.png", lane_mask="ll/a.png", clip_id="c1", frame_idx=3, timestamp_ns=123)
    base.update(kw)
    return Sample(**base)


def test_sample_round_trip(tmp_path):
    s = _sample(box3d=[[12.0, 0.1, 4.5, 1.8, 1.5, 12.3, -1.0]], track_id=[5],
                world_from_ego=np.eye(4), image_right="img/a_r.jpg", baseline_m=0.0998, depth="d/a.png",
                depth_source="stereo", el_lane=[0, -1],
                lanes=[LaneInstance([[100, 700], [400, 400]], points3d=[[3, 1.7, 0], [30, 1.6, 0.1]], order=-1,
                                    direction=1, continuity=0, ltype=1, weak=True),
                       LaneInstance([[800, 700], [500, 400]], order=1)],
                label_mask=LabelMask(agents=True, cls_vocab=range(8), box3d=True, depth=True, track=True, elements=True,
                                     el_vocab=(0, 1), el_attr=True, el_sign=True, topology=True, lane_geom=True,
                                     lane_3d=False, da=True, lane_mask=True))
    p = s.save(tmp_path / "s" / "a")
    t = Sample.load(p)
    for k in ("image", "hw", "source", "licence", "calib", "image_right", "baseline_m", "depth", "depth_source", "clip_id",
              "frame_idx", "timestamp_ns", "label_mask"):
        assert getattr(t, k) == getattr(s, k), k
    for k in ("K", "cam2ego", "boxes2d", "cls", "track_id", "world_from_ego", "el_boxes", "el_cls", "el_attr", "el_lane"):
        np.testing.assert_array_equal(getattr(t, k), getattr(s, k))
    assert t.el_sign.tolist() == ["", "regulatory--stop--g1"]
    np.testing.assert_array_equal(np.isnan(t.box3d), np.isnan(s.box3d))
    assert len(t.lanes) == 2 and t.lanes[0].weak and t.lanes[0].order == -1 and t.lanes[1].points3d is None
    np.testing.assert_array_equal(t.lanes[0].points, s.lanes[0].points)
    np.testing.assert_array_equal(t.lanes[0].points3d, s.lanes[0].points3d)


def test_sample_without_optional_parts_round_trips(tmp_path):
    s = Sample(image="a.jpg", hw=(590, 1640), K=K, cam2ego=np.eye(4), source="culane", licence="research",
               label_mask=LabelMask(lane_geom=True, lane_order=True),
               lanes=[LaneInstance([[1, 2], [3, 4], [5, 6]], order=1)])
    t = Sample.load(s.save(tmp_path / "x"))
    assert len(t.boxes2d) == 0 and t.box3d.shape == (0, 7) and len(t.el_boxes) == 0 and t.el_sign.shape == (0,)
    assert t.lanes[0].order == 1


@pytest.mark.parametrize("kw, msg", [
    (dict(boxes2d=[[10, 20, 5, 80]]), "x2 > x1"),
    (dict(cls=[11]), "cls ids"),
    (dict(el_sign=["x", ""]), "not a traffic sign"),
    (dict(el_attr=[13, -1]), "el_attr id"),
    (dict(el_lane=[0, 3]), "el_lane"),
    (dict(da_mask=None), "da file is missing"),
    (dict(licence="cc"), "licence"),
    (dict(calib="guess"), "calib"),
    (dict(image_right="r.jpg"), "image_right and baseline_m"),
    (dict(label_mask=LabelMask(agents=True, cls_vocab=[7])), "cls_vocab"),
    (dict(label_mask=LabelMask(elements=True, el_vocab=[1])), "el_vocab"),
    (dict(label_mask=LabelMask(agents=True, cls_vocab=range(8), lane_geom=True, lane_3d=True)), "no points3d"),
    (dict(box3d=[[12.0, 0.1, 4.5, 1.8, 1.5, np.nan, -1.0]]), "all finite"),
])
def test_sample_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _sample(**kw)


def test_label_mask_rules():
    with pytest.raises(ValueError, match="cls_vocab"):
        LabelMask(agents=True)
    with pytest.raises(ValueError, match="need agents"):
        LabelMask(track=True)
    with pytest.raises(ValueError, match="need lane_geom"):
        LabelMask(lane_geom=False, lane_type=True)
    with pytest.raises(ValueError, match="need elements"):
        LabelMask(el_attr=True)
    assert LabelMask(agents=True, cls_vocab=[7]).vocab_mask().tolist() == [False] * 7 + [True]
    assert LabelMask(elements=True, el_vocab=[1]).el_vocab_mask().tolist() == [False, True]
    with pytest.raises(ValueError, match="order"):
        LaneInstance([[0, 0], [1, 1]], order=0)
    assert len(AGENT_CLASSES) == 8 and len(BDD_CLASSES) == 10 and len(ELEMENT_ATTRS) == 13
    assert ATTR_FLIP[ELEMENT_ATTRS.index("turn_left")] == ELEMENT_ATTRS.index("turn_right")


def test_ground_cam2ego_axes():
    T = ground_cam2ego(1.5, pitch_rad=0.1)
    fwd = T[:3, :3] @ np.array([0, 0, 1.0])  # optical axis in ego coordinates
    assert fwd[0] > 0.99 and abs(fwd[1]) < 1e-9 and fwd[2] < 0  # forward and slightly down
    right = T[:3, :3] @ np.array([1.0, 0, 0])
    assert right[1] == pytest.approx(-1.0)  # camera x (right) is ego -y
    # a ground point straight ahead at 10 m projects below the principal point of a level camera
    T0 = ground_cam2ego(1.5)
    p_cam = np.linalg.inv(T0) @ np.array([10.0, 0, 0, 1])
    assert p_cam[2] == pytest.approx(10.0) and p_cam[1] == pytest.approx(1.5)
    # mirrored scene: a point 2 m left of the camera (ego y = +2) becomes 2 m right, and stays in front
    Tf = flip_cam2ego(T)
    p = np.linalg.inv(T) @ np.array([10.0, 2.0, 0.0, 1.0])     # camera coordinates of the point
    pf = Tf @ np.array([-p[0], p[1], p[2], 1.0])               # same pixel column mirrored -> camera x negated
    assert pf[0] == pytest.approx(10.0) and pf[1] == pytest.approx(-2.0) and pf[2] == pytest.approx(0.0, abs=1e-9)


def test_example_rig_geometry():
    r = StereoRig.from_yaml(CALIB)
    assert r.eye_hw == (900, 1440)
    assert r.baseline_m == pytest.approx(0.0998, abs=1e-4)
    assert r.bf == pytest.approx(1331.02 * 0.0998, rel=1e-3)
    # the shipped file is the alpha=0 rectification
    np.testing.assert_allclose(r.with_alpha(0.0).P1, r.P1, rtol=1e-6, atol=1e-6)
    hr, _ = r.raw_fov_deg()
    hc, _ = r.rect_fov_deg()
    assert hr > 75 and hc < 60
    assert r.with_alpha(0.5).rect_fov_deg()[0] > hc + 15
    assert any("zooms in" in w for w in r.check())
    assert not any("expected" in w for w in r.check(expected_baseline_m=0.1))
    assert any("expected" in w for w in r.check(expected_baseline_m=0.2))
    # network intrinsics: 1440x900 -> 640 long side -> letterboxed into 384x640 (height-limited)
    Kn, lb = r.network_K((384, 640))
    assert lb["unpad_wh"][1] == 384 and lb["left"] > 0
    assert Kn[0, 0] == pytest.approx(r.P1[0, 0] * lb["scale_xy"][0])
    sx = lb["scale_xy"][0]
    assert Kn[0, 2] == pytest.approx((r.P1[0, 2] + 0.5) * sx - 0.5 + lb["left"])  # pixel-centre convention
    # a rectified image already resized to 720x450 gives the same network K
    Kn2, _ = r.network_K((384, 640), src_hw=(450, 720))
    np.testing.assert_allclose(Kn2, Kn, atol=1e-9)
    # depth error grows with Z^2 and shrinks with resolution
    t = r.depth_table(640, ranges_m=(10, 20))
    e10, e20 = (row["err_m"][0.25] for row in t["rows"])
    assert e20 == pytest.approx(4 * e10)
    assert r.max_range_for(1440) == pytest.approx(r.max_range_for(640) * 1440 / 640)
    assert "rectified baseline" in r.report()


def test_scale_K_matches_cv2_resize():
    """A blob at pixel-centre x in a 1440-wide image lands where scale_K predicts after cv2.resize (half-pixel rule)."""
    import cv2

    from sparc.rig.stereo import scale_K

    img = np.zeros((90, 144), np.float32)
    x0, y0 = 100.0, 40.0
    yy, xx = np.mgrid[:90, :144]
    img[:] = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 3.0 ** 2))
    small = cv2.resize(img, (64, 40), interpolation=cv2.INTER_AREA)
    yy2, xx2 = np.mgrid[:40, :64]
    cx, cy = (small * xx2).sum() / small.sum(), (small * yy2).sum() / small.sum()
    K = np.array([[500.0, 0, x0], [0, 500.0, y0], [0, 0, 1]])
    Ks = scale_K(K, (90, 144), (40, 64))
    assert abs(Ks[0, 2] - cx) < 0.05 and abs(Ks[1, 2] - cy) < 0.05
    assert Ks[0, 2] != pytest.approx(x0 * 64 / 144, abs=0.1)  # naive scaling is 0.28 px off here


def test_rectify_maps_agree_with_projection():
    """A 3D point projected with the raw model lands, after remapping, where the rectified K puts it."""
    import cv2

    from sparc.rig.stereo import scale_K

    r = StereoRig.from_yaml(CALIB)
    out_hw = (450, 720)
    mx, my = r.rectify_maps(out_hw)
    X = np.array([[0.8, 0.3, 6.0]])
    # rectified pixel at out_hw
    Kn = scale_K(r.K_rect, r.eye_hw, out_hw)
    p_rect = Kn @ (r.R1 @ X.T)
    u, v = (p_rect[:2] / p_rect[2]).ravel()
    # the map at that pixel must point at the raw projection of X
    raw, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), r.K_left, r.D_left)
    iu, iv = int(round(u)), int(round(v))
    assert abs(mx[iv, iu] - raw[0, 0, 0]) < 3 and abs(my[iv, iu] - raw[0, 0, 1]) < 3
