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
    b, _ = evaluate(eng, synth_root / "data.yaml", batch=2, device="cpu", trt_module=trt)
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


# --------------------------------------------------------------------------- findings of the independent review
def test_static_batch_greater_than_one_handles_a_short_last_batch(run, synth_root, tmp_path):
    """8 train images, loader batch 3 -> chunks 3, 3, 2 against a static-batch-3 graph (and a static-batch-1 one)."""
    data = synth_root / "data.yaml"
    import yaml

    d = yaml.safe_load(data.read_text())
    d["val"] = "images/train"  # 8 images
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(d))
    b3 = export_onnx(run / "weights" / "best.pt", tmp_path / "b3.onnx", batch=3).onnx
    b1 = export_onnx(run / "weights" / "best.pt", tmp_path / "b1.onnx", batch=1).onnx
    s3, _ = evaluate(b3, tmp_path / "d.yaml", batch=3)
    s1, _ = evaluate(b1, tmp_path / "d.yaml", batch=3)
    for k in KEYS:
        assert s3[k] == pytest.approx(s1[k], abs=1e-5), k
    trt = make_fake_trt()
    eng = build_engine(b3, tmp_path / "b3.engine", trt_module=trt).engine  # the same, through the engine backend
    se, _ = evaluate(eng, tmp_path / "d.yaml", batch=3, device="cpu", trt_module=trt)
    for k in KEYS:
        assert se[k] == pytest.approx(s3[k], abs=1e-6), k


def test_raw_head_evaluates_like_the_topk_head(run, synth_root, tmp_path):
    a = export_onnx(run / "weights" / "best.pt", tmp_path / "t.onnx").onnx
    b = export_onnx(run / "weights" / "best.pt", tmp_path / "r.onnx", det_head="raw").onnx
    sa, _ = evaluate(a, synth_root / "data.yaml", batch=2)
    sb, _ = evaluate(b, synth_root / "data.yaml", batch=2)
    for k in KEYS:
        assert sb[k] == pytest.approx(sa[k], abs=1e-4), k


def test_engines_default_to_the_gpu_and_onnx_to_the_cpu(run, synth_root, tmp_path, monkeypatch):
    """eval --device must not default to the CPU for an engine (a real engine would be given CPU pointers)."""
    seen = {}

    def spy(model, backend, device, providers, trt_module):
        seen["device"] = device
        raise RuntimeError("stop")

    import adas_mt.deploy.evaluate as ev

    monkeypatch.setattr(ev, "open_backend", spy)
    trt = make_fake_trt()
    onnx = export_onnx(run / "weights" / "best.pt", tmp_path / "m.onnx").onnx
    eng = build_engine(onnx, tmp_path / "m.engine", trt_module=trt).engine
    for path, want in ((onnx, "cpu"), (eng, "cuda:0")):
        with pytest.raises(RuntimeError, match="stop"):
            evaluate(path, synth_root / "data.yaml")
        assert seen["device"] == want


def test_detections_agree_on_frames_from_the_real_validation_pipeline(run, synth_root, tmp_path):
    """Frames through MultiTaskDataset's val transforms (not random tensors): detections confident, parity holds."""
    import yaml

    from adas_mt.data.dataset import MultiTaskDataset
    from adas_mt.export import DeployWrapper, _head_mode, load_deploy_model, verify_onnx

    from .conftest import make_hyp

    from .conftest import nontrivial_model

    data = yaml.safe_load((synth_root / "data.yaml").read_text())
    ds = MultiTaskDataset(img_path=str(synth_root / "images" / "val"), data=data, imgsz=HW, augment=False, hyp=make_hyp(),
                          batch_size=1, cache=False, rect=False, prefix="")
    frames = torch.stack([ds[i]["img"].float() / 255 for i in range(len(ds))])
    # BatchNorm statistics calibrated on these frames: well-conditioned activations, like a trained model on its own domain
    trained_like = nontrivial_model("n", nc=2, imgsz=HW, calib=frames)
    r = export_onnx(trained_like, tmp_path / "m.onnx", verify=False, imgsz=HW)
    model = load_deploy_model(trained_like)
    wrapper = DeployWrapper(model).eval()
    confident = 0
    with _head_mode(model.model[-1], "topk", True):
        for i in range(len(ds)):
            x = frames[i : i + 1]
            confident += verify_onnx(wrapper, r.onnx, HW, batch=1, x=x)["det_confident"]
    assert confident > 0
