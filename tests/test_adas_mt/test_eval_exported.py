"""Exported models are evaluated by the training validator: ONNX / engine metrics must equal the .pt's."""

from __future__ import annotations

import pytest
import torch

from adas_mt.deploy.evaluate import evaluate, run_batched
from adas_mt.deploy.runner import OrtBackend
from adas_mt.deploy.trt_build import build_engine
from adas_mt.engine.config import MTConfig
from adas_mt.engine.val import run_validation
from adas_mt.export import export_onnx

from .conftest import nontrivial_model, structured_frame
from .fake_trt import make_fake_trt

HW = (96, 160)
KEYS = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/da_mIoU", "metrics/da_IoU_fg", "metrics/da_pixel_acc",
        "metrics/ll_mIoU", "metrics/ll_IoU_fg", "metrics/ll_pixel_acc", "fitness"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    (d / "weights").mkdir()
    model = nontrivial_model("n", nc=2, imgsz=HW)
    torch.save({"model": model, "ema": None, "train_args": {}}, d / "weights" / "best.pt")
    MTConfig.from_dict({"imgsz": list(HW)}).save(d / "mt.yaml")
    return d


def test_onnx_metrics_equal_the_checkpoint_metrics(run, synth_root):
    data = synth_root / "data.yaml"
    pt, _ = run_validation(run / "weights" / "best.pt", data, batch=2, device="cpu")
    onnx = export_onnx(run / "weights" / "best.pt").onnx
    ox, speed = evaluate(onnx, data, batch=2, device="cpu")
    for k in KEYS:
        assert ox[k] == pytest.approx(pt[k], abs=2e-2), (k, pt[k], ox[k])  # argmax ties / fp32 reordering only
    assert pt["metrics/da_pixel_acc"] > 0 and speed["images"] == 4 and speed["infer_ms_per_img"] > 0
    assert 0.0 < ox["metrics/da_mIoU"] <= 1.0 and 0.0 <= ox["metrics/ll_IoU_fg"] <= 1.0


def test_static_batch_one_engine_is_evaluated_with_batch_two_loader(run, synth_root, tmp_path):
    """Engines are built for batch 1: the evaluator must split loader batches instead of failing."""
    trt = make_fake_trt()
    onnx = export_onnx(run / "weights" / "best.pt", tmp_path / "m.onnx").onnx
    eng = build_engine(onnx, tmp_path / "m.engine", trt_module=trt).engine
    a, _ = evaluate(onnx, synth_root / "data.yaml", batch=2)
    b, _ = evaluate(eng, synth_root / "data.yaml", batch=2, trt_module=trt)
    for k in KEYS:
        assert b[k] == pytest.approx(a[k], abs=1e-6), k


def test_run_batched_splits_and_concatenates(run, tmp_path):
    onnx = export_onnx(run / "weights" / "best.pt", tmp_path / "m.onnx").onnx
    be = OrtBackend(onnx)
    assert be.batch == 1
    x = torch.cat([structured_frame(HW, seed=i) for i in range(3)])
    out = run_batched(be, x)
    assert out["det"].shape == (3, 300, 6) and out["da"].shape == (3, *HW)
    assert (out["da"][1] == be.infer(x[1:2])["da"][0]).all()


def test_class_count_mismatch_with_the_dataset_is_an_error(run, synth_root, tmp_path):
    import yaml

    onnx = export_onnx(run / "weights" / "best.pt", tmp_path / "m.onnx").onnx
    bad = yaml.safe_load((synth_root / "data.yaml").read_text())
    bad["nc"], bad["names"] = 3, ["a", "b", "c"]
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="nc=3"):
        evaluate(onnx, tmp_path / "bad.yaml", batch=2)
