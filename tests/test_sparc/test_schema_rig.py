from pathlib import Path

import numpy as np
import pytest

from sparc.data.schema import (CLS_TRAFFIC_LIGHT, DET_CLASSES, LabelMask, LaneInstance, Sample, ground_cam2ego)
from sparc.rig.stereo import StereoRig

CALIB = Path(__file__).parent / "data" / "stereo_calib_example.yaml"
K = np.array([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]])


def _sample(**kw):
    base = dict(image="img/a.jpg", hw=(720, 1280), K=K, cam2ego=ground_cam2ego(1.4, 0.02), source="bdd100k",
                licence="research", label_mask=LabelMask(det=True, tl=True, cls_vocab=range(10), lane_geom=True,
                                                         lane_dir=True, da=True, lane_mask=True),
                boxes2d=[[10, 20, 50, 80], [600, 100, 620, 140]], cls=[2, CLS_TRAFFIC_LIGHT], tl_color=[-1, 1],
                lanes=[LaneInstance([[100, 700], [400, 400]], order=-1, direction=1, continuity=0, ltype=1, weak=True)],
                da_mask="da/a.png", lane_mask="ll/a.png", clip_id="c1", frame_idx=3, timestamp_ns=123)
    base.update(kw)
    return Sample(**base)


def test_sample_round_trip(tmp_path):
    s = _sample(box3d=[[12.0, 0.1, 4.5, 1.8, 1.5, 12.3, -1.0], [np.nan] * 7], track_id=[5, -1],
                world_from_ego=np.eye(4), image_right="img/a_r.jpg", baseline_m=0.0998, depth="d/a.png",
                depth_source="stereo", label_mask=LabelMask(det=True, cls_vocab=range(10), box3d=True, depth=True,
                                                            track=True, tl=True))
    p = s.save(tmp_path / "s" / "a")
    t = Sample.load(p)
    for k in ("image", "hw", "source", "licence", "image_right", "baseline_m", "depth", "depth_source", "clip_id",
              "frame_idx", "timestamp_ns", "label_mask"):
        assert getattr(t, k) == getattr(s, k), k
    for k in ("K", "cam2ego", "boxes2d", "cls", "tl_color", "sign_cls", "track_id", "world_from_ego"):
        np.testing.assert_array_equal(getattr(t, k), getattr(s, k))
    np.testing.assert_array_equal(np.isnan(t.box3d), np.isnan(s.box3d))
    assert len(t.lanes) == 1 and t.lanes[0].weak and t.lanes[0].order == -1
    np.testing.assert_array_equal(t.lanes[0].points, s.lanes[0].points)


def test_sample_without_optional_parts_round_trips(tmp_path):
    s = Sample(image="a.jpg", hw=(590, 1640), K=K, cam2ego=np.eye(4), source="culane", licence="research",
               label_mask=LabelMask(lane_geom=True, lane_order=True),
               lanes=[LaneInstance([[1, 2], [3, 4], [5, 6]], order=1)])
    t = Sample.load(s.save(tmp_path / "x"))
    assert len(t.boxes2d) == 0 and t.box3d.shape == (0, 7) and t.lanes[0].order == 1


@pytest.mark.parametrize("kw, msg", [
    (dict(cls=[2, 3]), "not a traffic light"),
    (dict(boxes2d=[[10, 20, 5, 80], [600, 100, 620, 140]]), "x2 > x1"),
    (dict(cls=[2, 11]), "cls ids"),
    (dict(da_mask=None), "da file is missing"),
    (dict(licence="cc"), "licence"),
    (dict(image_right="r.jpg"), "image_right and baseline_m"),
    (dict(label_mask=LabelMask(det=True, cls_vocab=[9])), "cls_vocab"),
])
def test_sample_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _sample(**kw)


def test_label_mask_rules():
    with pytest.raises(ValueError, match="cls_vocab"):
        LabelMask(det=True)
    assert LabelMask(det=True, cls_vocab=[9]).vocab_mask().tolist() == [False] * 9 + [True]
    with pytest.raises(ValueError, match="order"):
        LaneInstance([[0, 0], [1, 1]], order=0)
    assert len(DET_CLASSES) == 10


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
    # depth error grows with Z^2 and shrinks with resolution
    t = r.depth_table(640, ranges_m=(10, 20))
    e10, e20 = (row["err_m"][0.25] for row in t["rows"])
    assert e20 == pytest.approx(4 * e10)
    assert r.max_range_for(1440) == pytest.approx(r.max_range_for(640) * 1440 / 640)
    assert "rectified baseline" in r.report()


def test_rectify_maps_agree_with_projection():
    """A 3D point projected with the raw model lands, after remapping, where the rectified K puts it."""
    import cv2

    r = StereoRig.from_yaml(CALIB)
    out_hw = (450, 720)
    mx, my = r.rectify_maps(out_hw)
    X = np.array([[0.8, 0.3, 6.0]])
    # rectified pixel at out_hw
    Kn = r.K_rect.copy()
    Kn[0] *= out_hw[1] / r.eye_hw[1]
    Kn[1] *= out_hw[0] / r.eye_hw[0]
    p_rect = Kn @ (r.R1 @ X.T)
    u, v = (p_rect[:2] / p_rect[2]).ravel()
    # the map at that pixel must point at the raw projection of X
    raw, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), r.K_left, r.D_left)
    iu, iv = int(round(u)), int(round(v))
    assert abs(mx[iv, iu] - raw[0, 0, 0]) < 3 and abs(my[iv, iu] - raw[0, 0, 1]) < 3
