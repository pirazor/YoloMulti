"""Pipeline geometry, overlays, sources, predict and bench."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from adas_mt.deploy.preprocess import letterbox_bgr
from adas_mt.deploy.runner import (
    Pipeline, Result, bench, class_palette, format_bench, iter_frames, overlay, predict, summarize,
)
from adas_mt.deploy.trt_build import build_engine
from adas_mt.export import export_onnx

from .conftest import nontrivial_model
from .fake_trt import make_fake_trt

HW = (96, 160)


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    return export_onnx(nontrivial_model("n", nc=3, imgsz=HW), tmp_path_factory.mktemp("r") / "m.onnx", imgsz=HW).onnx


def _frame(h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 6)  # smooth, so detections vary
    return np.ascontiguousarray(img)


class _FakeBackend:
    """Returns hand-made network outputs so the host-side mapping back to the frame can be checked exactly."""

    def __init__(self, det, da, ll, hw=HW):
        self.det, self.da, self.ll, self.input_hw, self.batch = det, da, ll, hw, 1

    def infer(self, x):
        return {"det": self.det[None], "da": self.da[None], "ll": self.ll[None]}

    def close(self):
        pass


def _pipe_with(onnx_path, backend, conf=0.25, masks_at="original"):
    p = Pipeline(onnx_path, conf=conf, masks_at=masks_at)
    p.backend = backend
    return p


@pytest.mark.parametrize("hw0", [(720, 1280), (480, 640), (1080, 1920), (600, 800)])
def test_boxes_and_masks_map_back_to_the_original_frame(onnx_path, hw0):
    """A box/mask drawn in original coordinates, pushed through the letterbox, must come back unchanged."""
    h0, w0 = hw0
    frame = np.zeros((h0, w0, 3), np.uint8)
    _, info = letterbox_bgr(frame, HW)
    sx, sy = info.scale_xy
    truth = np.array([[w0 * 0.25, h0 * 0.30, w0 * 0.60, h0 * 0.70]], np.float32)
    net = truth.copy()
    net[:, [0, 2]] = net[:, [0, 2]] * sx + info.left
    net[:, [1, 3]] = net[:, [1, 3]] * sy + info.top
    det = np.zeros((300, 6), np.float32)
    det[0] = [*net[0], 0.9, 2]
    det[1] = [*net[0], 0.1, 1]  # below the confidence threshold
    da = np.zeros(HW, np.int32)
    w, h = info.unpad_wh
    da[info.top : info.top + h // 2, info.left : info.left + w] = 1  # top half of the image area is "direct"
    ll = np.zeros(HW, np.int32)
    ll[: info.top, :] = 2  # the padding must never leak into the output
    res = _pipe_with(onnx_path, _FakeBackend(det, da, ll))(frame)
    assert res.boxes.shape == (1, 4) and np.allclose(res.boxes, truth, atol=1.0)
    assert res.scores[0] == pytest.approx(0.9) and res.classes.tolist() == [2]
    assert res.da.shape == res.ll.shape == (h0, w0) and res.da.dtype == np.uint8
    assert (res.da[: h0 // 2 - 2] == 1).all() and (res.da[h0 // 2 + 2 :] == 0).all()
    assert (res.ll == 0).all() or info.top == 0


def test_boxes_clipped_to_nothing_are_dropped_and_others_clipped(onnx_path):
    det = np.zeros((300, 6), np.float32)
    det[0] = [-50, -40, -10, -5, 0.9, 0]  # entirely in the padding / outside the frame
    det[1] = [-20, 10, 40, 60, 0.8, 1]  # sticks out on the left
    det[2] = [60, 20, 70, 20.2, 0.7, 1]  # degenerate (height < 1 px)
    zeros = np.zeros(HW, np.int32)
    res = _pipe_with(onnx_path, _FakeBackend(det, zeros, zeros))(np.zeros((HW[0], HW[1], 3), np.uint8))
    assert len(res) == 1 and res.boxes[0, 0] == 0 and res.classes.tolist() == [1]


def test_no_detections_and_logit_outputs(onnx_path):
    det = np.zeros((300, 6), np.float32)  # all scores 0 -> nothing above conf
    logits = np.zeros((3, *HW), np.float32)
    logits[2] = 5.0  # class 2 everywhere
    res = _pipe_with(onnx_path, _FakeBackend(det, logits, logits))(np.zeros((HW[0], HW[1], 3), np.uint8))
    assert len(res) == 0 and res.boxes.shape == (0, 4)
    assert (res.da == 2).all() and (res.ll == 2).all()


def test_masks_at_network_resolution(onnx_path):
    zeros = np.zeros(HW, np.int32)
    res = _pipe_with(onnx_path, _FakeBackend(np.zeros((300, 6), np.float32), zeros, zeros), masks_at="network")(_frame(720, 1280))
    w, h = res.info.unpad_wh  # the unpadded network area
    assert (w, h) == (160, 90) and res.da.shape == (h, w) == res.ll.shape


def test_real_model_end_to_end_with_onnxruntime(onnx_path):
    pipe = Pipeline(onnx_path, conf=0.05)
    f = _frame(360, 640)
    res = pipe(f)
    assert isinstance(res, Result) and res.da.shape == res.ll.shape == f.shape[:2]
    assert set(np.unique(res.da)) <= {0, 1, 2} and res.boxes.shape[1] == 4
    assert set(res.timings) == {"pre", "infer", "post", "total"} and res.timings["total"] >= res.timings["infer"] > 0
    assert (res.boxes[:, [0, 2]] <= f.shape[1] + 1e-3).all() and (res.boxes[:, [1, 3]] <= f.shape[0] + 1e-3).all()
    assert (res.boxes >= 0).all()
    other = pipe(_frame(480, 854, seed=2))  # frame size may change between calls
    assert other.da.shape == (480, 854)


def test_runner_and_validation_see_the_same_input(onnx_path):
    """The pipeline's network input is exactly what deploy.preprocess produces (which equals the val pipeline)."""
    from adas_mt.deploy.preprocess import preprocess

    class Capture(_FakeBackend):
        def infer(self, x):
            self.x = np.array(x)
            return super().infer(x)

    cap = Capture(np.zeros((300, 6), np.float32), np.zeros(HW, np.int32), np.zeros(HW, np.int32))
    f = _frame(720, 1280)
    _pipe_with(onnx_path, cap)(f)
    assert np.array_equal(cap.x[0], preprocess(f, HW)[0])


def test_metadata_mismatch_is_rejected(onnx_path, tmp_path):
    bad = tmp_path / "wrong.onnx"
    bad.write_bytes(onnx_path.read_bytes())
    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    meta["imgsz"] = [128, 160]
    bad.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="metadata says"):
        Pipeline(bad)
    with pytest.raises(ValueError, match="masks_at"):
        Pipeline(onnx_path, masks_at="x")
    with pytest.raises(FileNotFoundError, match="no metadata"):
        orphan = tmp_path / "orphan.engine"
        orphan.write_bytes(b"x")
        Pipeline(orphan)


def test_trt_pipeline_equals_ort_pipeline(onnx_path, tmp_path):
    trt = make_fake_trt()
    eng = build_engine(onnx_path, tmp_path / "m.engine", trt_module=trt).engine
    a = Pipeline(onnx_path, conf=0.05)
    b = Pipeline(eng, backend="trt", conf=0.05, device="cpu", trt_module=trt)
    f = _frame(360, 640, seed=3)
    ra, rb = a(f), b(f)
    assert np.array_equal(ra.da, rb.da) and np.array_equal(ra.ll, rb.ll)
    assert np.allclose(ra.boxes, rb.boxes) and np.array_equal(ra.classes, rb.classes)
    assert b.meta["engine"]["precision"] == "fp16"


# --------------------------------------------------------------------------- overlay / sources
def test_overlay_colours_classes_and_leaves_the_input_alone():
    f = np.full((100, 200, 3), 100, np.uint8)
    da = np.zeros((100, 200), np.uint8)
    da[50:, :] = 1
    ll = np.zeros((100, 200), np.uint8)
    ll[:, 100:102] = 1
    res = Result(np.array([[10, 10, 60, 40]], np.float32), np.array([0.9], np.float32), np.array([1]), da, ll,
                 letterbox_bgr(f, (96, 160))[1])
    before = f.copy()
    out = overlay(f, res, ["car", "person"])
    assert np.array_equal(f, before) and out.shape == f.shape
    assert out[20, 150].tolist() == [100, 100, 100]  # untouched background
    assert out[80, 20].tolist() != [100, 100, 100]  # tinted drivable area
    assert out[10, 101].tolist() == [40, 40, 240]  # solid lane = red
    assert (out[10:40, 10] != 100).any()  # box outline drawn
    small = Result(res.boxes, res.scores, res.classes, da[::2, ::2], ll[::2, ::2], res.info)  # network-res masks are resized
    assert overlay(f, small, ["car", "person"]).shape == f.shape
    assert len(set(class_palette(8))) == 8


def test_iter_frames_images_dirs_videos(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"{i}.png"), _frame(40, 60, seed=i))
    assert [n for n, _ in iter_frames(d)] == ["0.png", "1.png", "2.png"]
    assert [n for n, _ in iter_frames(d / "1.png")] == ["1.png"]
    vid = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (60, 40))
    for i in range(4):
        w.write(_frame(40, 60, seed=i))
    w.release()
    frames = list(iter_frames(vid))
    assert len(frames) == 4 and frames[0][1].shape == (40, 60, 3)
    with pytest.raises(FileNotFoundError):
        list(iter_frames(tmp_path / "nothing.png"))
    with pytest.raises(FileNotFoundError, match="no images"):
        (tmp_path / "empty").mkdir()
        list(iter_frames(tmp_path / "empty"))


def test_predict_writes_overlays_masks_and_video(onnx_path, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"f{i}.png"), _frame(72, 128, seed=i))
    out = tmp_path / "out"
    stats = predict(onnx_path, d, out, conf=0.05, save_masks=True)
    assert stats["frames"] == 3 and stats["latency_ms"]["p50"] > 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["f0.jpg", "f0_da.png", "f0_ll.png", "f1.jpg", "f1_da.png", "f1_ll.png", "f2.jpg", "f2_da.png", "f2_ll.png"]
    assert cv2.imread(str(out / "f0_da.png"), cv2.IMREAD_UNCHANGED).shape == (72, 128)

    vid = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 72))
    for i in range(5):
        w.write(_frame(72, 128, seed=i))
    w.release()
    s2 = predict(onnx_path, vid, tmp_path / "vout", conf=0.05, max_frames=4, fps=10)
    assert s2["frames"] == 4 and (tmp_path / "vout" / "predictions.mp4").stat().st_size > 0


def test_bench_reports_stage_statistics(onnx_path):
    pipe = Pipeline(onnx_path)
    r = bench(pipe, _frame(180, 320), n=6, warmup=2)
    s = r["stages_ms"]
    assert set(s) == {"pre", "infer", "post", "total"} and r["n"] == 6
    assert s["total"]["p50"] >= s["infer"]["p50"] > 0 and s["total"]["p99"] >= s["total"]["p50"]
    assert r["fps_mean"] > 0 and "FPS" in format_bench(r)
    syn = bench(pipe, None, n=2, warmup=0)  # synthetic 720p default frame
    assert syn["frame_hw"] == [720, 1280]
    assert np.isnan(summarize([])["p50"]) and summarize([1, 2, 3, 4])["p50"] == 2.5


def test_gpu_preprocess_path_matches_cpu_path(onnx_path):
    """The torch letterbox (used on the Jetson GPU) feeds the same network input to within 8-bit rounding."""
    f = _frame(360, 640, seed=5)
    cpu = Pipeline(onnx_path, conf=0.05)
    gpu = Pipeline(onnx_path, conf=0.05, gpu_preprocess=True, device="cpu")  # torch path, no CUDA in CI
    a, b = cpu(f), gpu(f)
    assert (a.da == b.da).mean() > 0.99 and (a.ll == b.ll).mean() > 0.99
    assert a.info == b.info
