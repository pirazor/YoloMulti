"""The deployment sub-commands through the real CLI entry point."""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import pytest
import torch

from adas_mt.cli import main
from adas_mt.engine.config import MTConfig

from .conftest import nontrivial_model

HW = (96, 160)


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_run")
    (d / "weights").mkdir()
    torch.save({"model": nontrivial_model("n", nc=2, imgsz=HW), "ema": None, "train_args": {}}, d / "weights" / "best.pt")
    MTConfig.from_dict({"imgsz": list(HW)}).save(d / "mt.yaml")
    return d / "weights" / "best.pt"


def test_export_predict_bench_eval_roundtrip(weights, synth_root, tmp_path, capsys):
    img = tmp_path / "f.png"
    cv2.imwrite(str(img), cv2.GaussianBlur(np.random.default_rng(0).integers(0, 256, (72, 128, 3), dtype=np.uint8), (0, 0), 5))
    assert main(["export", "--weights", str(weights), "--verify-image", str(img)]) == 0
    onnx = weights.with_suffix(".onnx")
    assert onnx.is_file() and onnx.with_suffix(".json").is_file()
    assert "parity:" in capsys.readouterr().out

    out = tmp_path / "pred"
    assert main(["predict", "--model", str(onnx), "--source", str(img), "--out", str(out), "--conf", "0.05", "--save-masks",
                 "--device", "cpu"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["f.jpg", "f_da.png", "f_ll.png"]

    rep = tmp_path / "bench.json"
    assert main(["bench", "--model", str(onnx), "--source", str(img), "--n", "3", "--warmup", "1", "--json", str(rep),
                 "--device", "cpu"]) == 0
    data = json.loads(rep.read_text())
    assert data["n"] == 3 and data["stages_ms"]["total"]["p50"] > 0
    assert "FPS" in capsys.readouterr().out

    assert main(["eval", "--model", str(onnx), "--data", str(synth_root / "data.yaml"), "--batch", "2", "--workers", "0"]) == 0
    assert "metrics/da_mIoU" in capsys.readouterr().out


def test_export_options_reach_the_exporter(weights, tmp_path):
    out = tmp_path / "d.onnx"
    assert main(["export", "--weights", str(weights), "--out", str(out), "--dynamic", "--seg-dtype", "uint8",
                 "--decompose-pixel-shuffle", "--no-simplify"]) == 0
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["dynamic_batch"] and meta["seg_dtype"] == "uint8" and meta["decompose_pixel_shuffle"]


def test_trt_build_without_tensorrt_explains_what_to_do(weights, tmp_path, monkeypatch):
    out = tmp_path / "m.onnx"
    main(["export", "--weights", str(weights), "--out", str(out)])
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # make `import tensorrt` fail
    from adas_mt.deploy.trt_build import TrtBuildError

    with pytest.raises(TrtBuildError, match="system-site-packages"):
        main(["trt-build", "--onnx", str(out)])
    with pytest.raises(SystemExit):  # argparse: --calib alone is fine, an unknown precision is not
        main(["trt-build", "--onnx", str(out), "--precision", "int4"])


def test_jetson_runtime_modules_do_not_import_the_training_stack():
    """The Jetson only needs the engine, the sidecar JSON and torch for CUDA buffers: importing the runner, the engine
    builder, the metadata helpers and the CLI must not pull in ultralytics or torch."""
    import subprocess

    code = ("import sys, adas_mt.cli, adas_mt.deploy.runner, adas_mt.deploy.trt_build, adas_mt.deploy.meta, adas_mt.deploy.preprocess;"
            "bad = [m for m in ('ultralytics', 'torch', 'timm', 'onnxruntime') if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]", out
