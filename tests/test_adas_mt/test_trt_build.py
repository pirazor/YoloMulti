"""Engine builder and TensorRT backend, against a fake ``tensorrt`` that mimics the 8.6 and 10.x differences."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from adas_mt.deploy.meta import meta_path
from adas_mt.deploy.runner import OrtBackend, TrtBackend, open_backend
from adas_mt.deploy.trt_build import (
    KEEP_FP16_PRESETS, TrtBuildError, build_engine, calibration_images, trtexec_command,
)
from adas_mt.export import export_onnx

from .conftest import nontrivial_model, structured_frame
from .fake_trt import BuilderFlag, DataType, MemoryPoolType, make_fake_trt

HW = (96, 160)
VERSIONS = ["8.6.1", "10.3.0", "10.16.0"]


@pytest.fixture(scope="module")
def model():
    return nontrivial_model("n", nc=3, imgsz=HW)


@pytest.fixture(scope="module")
def onnx_static(model, tmp_path_factory):
    return export_onnx(model, tmp_path_factory.mktemp("s") / "m.onnx", imgsz=HW).onnx


@pytest.fixture(scope="module")
def onnx_dynamic(model, tmp_path_factory):
    return export_onnx(model, tmp_path_factory.mktemp("d") / "m.onnx", imgsz=HW, dynamic=True).onnx


@pytest.fixture()
def images(tmp_path):
    d = tmp_path / "calib"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        cv2.imwrite(str(d / f"{i:02d}.jpg"), rng.integers(0, 256, (200, 320, 3), dtype=np.uint8))
    return d


@pytest.mark.parametrize("version", VERSIONS)
def test_fp16_build_per_tensorrt_version(onnx_static, tmp_path, version):
    trt = make_fake_trt(version)
    r = build_engine(onnx_static, tmp_path / "m.engine", precision="fp16", workspace_mb=512, trt_module=trt)
    b = trt.builders[0]
    assert b.network.flags == (0 if version.startswith("10") else 1)  # EXPLICIT_BATCH only before TensorRT 10
    cfg = b.config
    assert BuilderFlag.FP16 in cfg.flags and BuilderFlag.INT8 not in cfg.flags
    assert cfg.pools[MemoryPoolType.WORKSPACE] == 512 << 20
    assert not hasattr(cfg, "max_workspace_size") or cfg.max_workspace_size == 0  # the deprecated knob is never used
    assert cfg.profiles == [] and r.engine.is_file()
    side = json.loads(meta_path(r.engine).read_text())
    assert side["engine"]["precision"] == "fp16" and side["engine"]["tensorrt"] == version and side["imgsz"] == list(HW)


def test_fp32_build_sets_no_precision_flags(onnx_static, tmp_path):
    trt = make_fake_trt()
    build_engine(onnx_static, tmp_path / "m.engine", precision="fp32", trt_module=trt)
    assert trt.builders[0].config.flags == set()


def test_parse_failure_is_reported_with_a_hint(onnx_static, tmp_path):
    trt = make_fake_trt("8.6.1")
    orig = trt.OnnxParser
    trt.OnnxParser = lambda net, log: type("P", (), {  # noqa: E731
        "parse_from_file": lambda self, p: False, "num_errors": 1, "get_error": lambda self, i: "In node 5: DepthToSpace unsupported"})()
    with pytest.raises(TrtBuildError, match="decompose-pixel-shuffle"):
        build_engine(onnx_static, tmp_path / "m.engine", trt_module=trt)
    assert orig  # keep the reference used


def test_uint8_outputs_are_rejected_before_tensorrt_10(model, tmp_path):
    p = export_onnx(model, tmp_path / "u8.onnx", imgsz=HW, seg_dtype="uint8").onnx
    with pytest.raises(TrtBuildError, match="TensorRT >= 10"):
        build_engine(p, tmp_path / "u8.engine", trt_module=make_fake_trt("8.6.1"))
    build_engine(p, tmp_path / "u8.engine", trt_module=make_fake_trt("10.3.0"))  # fine on 10


def test_dynamic_batch_gets_an_optimisation_profile(onnx_dynamic, onnx_static, tmp_path):
    trt = make_fake_trt()
    r = build_engine(onnx_dynamic, tmp_path / "d.engine", opt_batch=2, max_batch=6, trt_module=trt)
    prof = trt.builders[0].config.profiles[0].shapes["images"]
    assert prof == ((1, 3, *HW), (2, 3, *HW), (6, 3, *HW))
    assert r.info["dynamic_batch"] and (r.info["opt_batch"], r.info["max_batch"]) == (2, 6)
    trt2 = make_fake_trt()
    r2 = build_engine(onnx_static, tmp_path / "s.engine", opt_batch=2, trt_module=trt2)  # ignored for a static ONNX
    assert trt2.builders[0].config.profiles == [] and not r2.info["dynamic_batch"]


def test_int8_needs_calibration_data(onnx_static, tmp_path):
    with pytest.raises(TrtBuildError, match="--calib"):
        build_engine(onnx_static, tmp_path / "m.engine", precision="int8", trt_module=make_fake_trt())


def test_int8_calibration_feeds_deployment_preprocessing(onnx_static, tmp_path, images):
    from adas_mt.deploy.preprocess import preprocess

    trt = make_fake_trt()
    cache = tmp_path / "m.calib"
    build_engine(onnx_static, tmp_path / "m.engine", precision="int8", calib=images, calib_n=6, calib_cache=cache,
                 trt_module=trt, calib_device="cpu")
    cfg = trt.builders[0].config
    assert {BuilderFlag.INT8, BuilderFlag.FP16} <= cfg.flags
    seen = cfg.int8_calibrator.calibration_seen
    assert len(seen) == 6 and all(a.shape == (1, 3, *HW) for a in seen)
    assert 0.0 <= min(a.min() for a in seen) and max(a.max() for a in seen) <= 1.0
    first = cfg.int8_calibrator.images[0]
    assert np.array_equal(seen[0][0], preprocess(cv2.imread(str(first)), HW)[0])  # same pixels the runner feeds
    assert cache.read_bytes() == b"calibration-cache"
    # an existing cache is reused, but only because it was asked for
    assert cfg.int8_calibrator.read_calibration_cache() == b"calibration-cache"
    trt2 = make_fake_trt()
    build_engine(onnx_static, tmp_path / "n.engine", precision="int8", calib=images, calib_n=3, trt_module=trt2,
                 calib_device="cpu")
    assert trt2.builders[0].config.int8_calibrator.read_calibration_cache() is None  # no implicit stale cache


def test_int8_pins_float_head_layers_to_fp16_but_not_integer_ones(onnx_static, tmp_path, images):
    trt = make_fake_trt()
    build_engine(onnx_static, tmp_path / "m.engine", precision="int8", calib=images, calib_n=2, keep_fp16="heads",
                 trt_module=trt, calib_device="cpu")
    net = trt.builders[0].network
    pinned = [net.get_layer(i) for i in range(net.num_layers) if net.get_layer(i).precision == DataType.HALF]
    assert pinned and all(any(s in l.name for s in ("/model.", "/da_head/", "/ll_head/")) for l in pinned)
    assert any("/da_head/" in l.name for l in pinned) and any("/ll_head/" in l.name for l in pinned)
    for l in pinned:  # never an integer-typed layer (TopK indices, Shape, Gather, ...)
        assert all(l.get_output(j).dtype in (DataType.FLOAT, DataType.HALF) for j in range(l.num_outputs))
    backbone = [net.get_layer(i) for i in range(net.num_layers)
                if "/model.5/" in net.get_layer(i).name or "/model.2/" in net.get_layer(i).name]
    assert backbone and all(l.precision is None for l in backbone)  # the backbone stays INT8-eligible
    assert BuilderFlag.OBEY_PRECISION_CONSTRAINTS in trt.builders[0].config.flags

    trt_none = make_fake_trt()
    build_engine(onnx_static, tmp_path / "n.engine", precision="int8", calib=images, calib_n=2, keep_fp16="none",
                 trt_module=trt_none, calib_device="cpu")
    net = trt_none.builders[0].network
    assert all(net.get_layer(i).precision is None for i in range(net.num_layers))
    assert set(KEEP_FP16_PRESETS) == {"none", "seg", "heads"}


def test_calibration_image_selection(tmp_path, images):
    files = calibration_images(images, n=4)
    assert len(files) == 4 and len(set(files)) == 4
    assert calibration_images(images, n=4) == files  # deterministic
    assert len(calibration_images(images, n=100)) == 10
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    cv2.imwrite(str(root / "images/train/a.jpg"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(root / "images/val/b.jpg"), np.zeros((8, 8, 3), np.uint8))
    (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\n")
    assert [p.name for p in calibration_images(root / "data.yaml")] == ["a.jpg"]  # train split only, never val
    with pytest.raises(TrtBuildError, match="no calibration images"):
        calibration_images(tmp_path / "ds" / "images" / "empty_dir_that_does_not_exist")


def test_timing_cache_roundtrip(onnx_static, tmp_path):
    trt = make_fake_trt()
    tc = tmp_path / "t.cache"
    build_engine(onnx_static, tmp_path / "m.engine", timing_cache=tc, trt_module=trt)
    assert tc.read_bytes() == b"timing:"
    build_engine(onnx_static, tmp_path / "m2.engine", timing_cache=tc, trt_module=make_fake_trt())
    assert tc.read_bytes() == b"timing:timing:"


def test_trtexec_command_matches_the_precisions():
    assert "--fp16" in trtexec_command("a.onnx", "a.engine", "fp16") and "--int8" not in trtexec_command("a.onnx", "a.engine", "fp16")
    assert "--int8" in trtexec_command("a.onnx", "a.engine", "int8") and "--saveEngine=a.engine" in trtexec_command("a.onnx", "a.engine")


# --------------------------------------------------------------------------- the runtime backend
@pytest.mark.parametrize("version", ["8.6.1", "10.3.0"])
def test_trt_backend_matches_onnxruntime(onnx_static, tmp_path, version):
    import torch

    trt = make_fake_trt(version)
    eng = build_engine(onnx_static, tmp_path / "m.engine", trt_module=trt).engine
    be = TrtBackend(eng, device="cpu", trt_module=trt)
    ref = OrtBackend(onnx_static)
    assert be.input_hw == HW and be.batch == 1 and not be.dynamic
    assert be.output_names == ["det", "da", "ll"]
    x = structured_frame(HW).numpy()
    got, want = be.infer(x), ref.infer(x)
    for k in want:
        assert got[k].dtype == want[k].dtype and got[k].shape == want[k].shape, k
        assert np.array_equal(got[k], want[k]), k
    # torch input, repeated calls reuse the same bound buffers and stay correct
    for seed in (2, 3):
        xt = structured_frame(HW, seed=seed)
        assert np.array_equal(be.infer(xt)["da"], ref.infer(xt.numpy())["da"])
    first = be.infer(structured_frame(HW, seed=7))
    snapshot = {k: v.copy() for k, v in first.items()}
    be.infer(structured_frame(HW, seed=8))  # results of an earlier call must survive later calls
    assert all(np.array_equal(first[k], snapshot[k]) for k in first)
    with pytest.raises(ValueError, match="engine takes batch 1"):
        be.infer(torch.zeros(2, 3, *HW))
    be.close()


def test_trt_backend_dynamic_batch(onnx_dynamic, tmp_path):
    import torch

    trt = make_fake_trt()
    eng = build_engine(onnx_dynamic, tmp_path / "d.engine", opt_batch=2, max_batch=4, trt_module=trt).engine
    be = TrtBackend(eng, device="cpu", trt_module=trt)
    ref = OrtBackend(onnx_dynamic)
    assert be.dynamic and be.batch is None and be.max_batch == 4
    for b in (1, 3, 2, 4, 1):  # shrinking and growing, each checked against ORT
        x = torch.cat([structured_frame(HW, seed=i) for i in range(b)])
        out = be.infer(x)
        want = ref.infer(x.numpy())
        assert out["det"].shape == (b, 300, 6) and out["da"].shape == (b, *HW)
        assert np.array_equal(out["da"], want["da"]) and np.array_equal(out["ll"], want["ll"])
    with pytest.raises(ValueError, match="1..4"):
        be.infer(torch.zeros(5, 3, *HW))


def test_trt_backend_uint8_outputs(model, tmp_path):
    trt = make_fake_trt("10.3.0")
    p = export_onnx(model, tmp_path / "u8.onnx", imgsz=HW, seg_dtype="uint8").onnx
    be = TrtBackend(build_engine(p, tmp_path / "u8.engine", trt_module=trt).engine, device="cpu", trt_module=trt)
    out = be.infer(structured_frame(HW).numpy())
    assert out["da"].dtype == np.uint8 and out["det"].dtype == np.float32


def test_engine_that_does_not_load_gives_a_clear_error(tmp_path):
    bad = tmp_path / "bad.engine"
    bad.write_bytes(b"\x00not an engine")
    with pytest.raises(RuntimeError, match="same TensorRT|TensorRT version"):
        TrtBackend(bad, device="cpu", trt_module=make_fake_trt())


def test_open_backend_picks_by_suffix(onnx_static, tmp_path):
    trt = make_fake_trt()
    eng = build_engine(onnx_static, tmp_path / "m.plan", trt_module=trt).engine
    assert isinstance(open_backend(onnx_static), OrtBackend)
    assert isinstance(open_backend(eng, device="cpu", trt_module=trt), TrtBackend)
    with pytest.raises(ValueError):
        open_backend(onnx_static, backend="nope")
