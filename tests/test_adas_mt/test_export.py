"""ONNX export: graph contract, parity with PyTorch, op audit, metadata."""

from __future__ import annotations

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from adas_mt.deploy.meta import meta_path, read_meta
from adas_mt.export import (
    TRT_OPS, DeployWrapper, ExportParityError, ReshapePixelShuffle, _det_parity, audit_ops, export_onnx,
    load_deploy_model, onnx_ops, verify_onnx,
)

from .conftest import nontrivial_model, structured_frame

HW = (96, 160)


@pytest.fixture(scope="module")
def model():
    return nontrivial_model("n", nc=3, imgsz=HW)


def _session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_graph_contract_int32(model, tmp_path):
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW)
    g = onnx.load(str(r.onnx)).graph
    ins = {i.name: i for i in g.input}
    outs = {o.name: o for o in g.output}
    assert list(ins) == ["images"] and list(outs) == ["det", "da", "ll"]
    dims = lambda t: [d.dim_value for d in t.type.tensor_type.shape.dim]  # noqa: E731
    assert dims(ins["images"]) == [1, 3, *HW]
    assert dims(outs["det"]) == [1, 300, 6] and dims(outs["da"]) == dims(outs["ll"]) == [1, *HW]
    assert outs["det"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert outs["da"].type.tensor_type.elem_type == onnx.TensorProto.INT32 == outs["ll"].type.tensor_type.elem_type
    assert r.parity["det_confident"] > 0, "the parity input must produce confident detections"
    assert r.parity["da_agree"] == r.parity["ll_agree"] == 1.0


def test_inference_graph_only_has_the_deployed_network(model, tmp_path):
    """No distillation projector, no auxiliary DA classifier, no one-to-many Detect branch, no BatchNorm."""
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW)
    m = onnx.load(str(r.onnx))
    names = " ".join(n.name for n in m.graph.node) + " " + " ".join(i.name for i in m.graph.initializer)
    for banned in ("kd_proj", "aux", "one2many", "BatchNormalization"):
        assert banned not in names and banned not in r.ops, banned
    lean = load_deploy_model(model)
    assert not hasattr(lean, "kd_proj") and lean.da_head.aux is None
    assert not lean.training and all(not p.requires_grad for p in lean.parameters())


def test_source_model_is_not_modified(model, tmp_path):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    n_mod = sum(1 for _ in model.modules())
    export_onnx(model, tmp_path / "m.onnx", imgsz=HW)
    assert sum(1 for _ in model.modules()) == n_mod and model.da_head.aux is not None
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert not getattr(model.model[-1], "export", False)


@pytest.mark.parametrize("seg_dtype,elem", [("uint8", onnx.TensorProto.UINT8), ("int32", onnx.TensorProto.INT32),
                                            ("logits", onnx.TensorProto.FLOAT)])
def test_seg_dtypes(model, tmp_path, seg_dtype, elem):
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW, seg_dtype=seg_dtype)
    outs = {o.name: o for o in onnx.load(str(r.onnx)).graph.output}
    assert outs["da"].type.tensor_type.elem_type == outs["ll"].type.tensor_type.elem_type == elem
    x = structured_frame(HW).numpy()
    da = _session(r.onnx).run(["da"], {"images": x})[0]
    assert da.shape == ((1, 3, *HW) if seg_dtype == "logits" else (1, *HW))
    assert r.meta["outputs"]["da"]["dtype"] == ("float32" if seg_dtype == "logits" else seg_dtype)


def test_dynamic_batch_runs_other_batch_sizes(model, tmp_path):
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW, dynamic=True)
    sess = _session(r.onnx)
    wrapper = DeployWrapper(load_deploy_model(model))
    wrapper.model.model[-1].export = True
    for b in (1, 3):
        x = torch.cat([structured_frame(HW, seed=i) for i in range(b)])
        det, da, ll = sess.run(None, {"images": x.numpy()})
        assert det.shape == (b, 300, 6) and da.shape == ll.shape == (b, *HW)
        wd, wda, _ = wrapper(x)
        assert (da == wda.numpy()).mean() > 0.999
    assert r.meta["dynamic_batch"] and r.meta["outputs"]["det"]["shape"][0] == "dynamic"
    g = onnx.load(str(r.onnx)).graph  # declared shapes are exact (the tracer leaves stale symbolic dims otherwise)
    declared = {v.name: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.input) + list(g.output)}
    assert declared == {"images": ["batch", 3, *HW], "det": ["batch", 300, 6], "da": ["batch", *HW], "ll": ["batch", *HW]}


def test_decomposed_pixel_shuffle_is_equivalent_and_avoids_depthtospace(model, tmp_path):
    a = export_onnx(model, tmp_path / "a.onnx", imgsz=HW, seg_dtype="logits")
    b = export_onnx(model, tmp_path / "b.onnx", imgsz=HW, seg_dtype="logits", decompose_pixel_shuffle=True)
    assert "DepthToSpace" in a.ops and "DepthToSpace" not in b.ops
    assert a.audit["check_on_device"] == ["DepthToSpace"] and b.audit["check_on_device"] == []
    x = structured_frame(HW).numpy()
    la = _session(a.onnx).run(["ll"], {"images": x})[0]
    lb = _session(b.onnx).run(["ll"], {"images": x})[0]
    assert np.allclose(la, lb, atol=1e-4 * max(1.0, np.abs(la).max()))
    r = 4
    t = torch.randn(2, 3 * r * r, 5, 7)
    assert torch.equal(ReshapePixelShuffle(r)(t), torch.nn.PixelShuffle(r)(t))


def test_op_audit_only_uses_tensorrt_safe_ops(model, tmp_path):
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW, decompose_pixel_shuffle=True)
    assert r.audit["unsupported"] == [], r.audit
    assert set(r.ops) <= TRT_OPS
    # the audit itself flags an unknown op
    assert audit_ops({"Conv": 3, "NonMaxSuppression": 1})["unsupported"] == ["NonMaxSuppression"]
    assert "NonMaxSuppression" not in onnx_ops(r.onnx), "the head must be NMS-free"


def test_metadata_sidecar_and_embedded_copy(model, tmp_path):
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW)
    side = json.loads(meta_path(r.onnx).read_text())
    assert side == r.meta == read_meta(r.onnx)
    assert side["imgsz"] == list(HW) and side["names"] == ["0", "1", "2"] and side["max_det"] == 300
    assert side["input"]["color"] == "RGB" and side["layers"]["da"] == "/da_head/"
    meta_path(r.onnx).unlink()  # the copy embedded in the graph still describes it
    assert read_meta(r.onnx)["imgsz"] == list(HW)
    assert meta_path("x/y.engine").name == "y.engine.json" and meta_path("x/y.onnx").name == "y.json"


def test_checkpoint_export_reads_geometry_from_the_run(model, tmp_path):
    from adas_mt.engine.config import MTConfig

    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    torch.save({"model": model, "ema": None}, run / "weights" / "best.pt")
    MTConfig.from_dict({"imgsz": [96, 160]}).save(run / "mt.yaml")
    r = export_onnx(run / "weights" / "best.pt")  # imgsz from mt.yaml, output next to the weights
    assert r.onnx == run / "weights" / "best.onnx" and r.meta["imgsz"] == [96, 160]
    with pytest.raises(ValueError, match="multiples of 32"):
        export_onnx(run / "weights" / "best.pt", imgsz=(100, 160))


def test_rejects_foreign_checkpoints(tmp_path):
    torch.save({"model": torch.nn.Linear(2, 2)}, tmp_path / "x.pt")
    with pytest.raises(TypeError, match="MultiTaskModel"):
        export_onnx(tmp_path / "x.pt", imgsz=HW)


def test_parity_check_detects_a_broken_graph(model, tmp_path):
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW, verify=False)
    wrapper = DeployWrapper(load_deploy_model(model))
    wrapper.model.model[-1].export = True
    for p in wrapper.parameters():  # a different network than the one that was exported
        p.data.add_(torch.randn_like(p) * 0.2)
    with pytest.raises(ExportParityError):
        verify_onnx(wrapper, r.onnx, HW)


def test_det_parity_is_robust_to_ties_but_not_to_real_differences():
    rng = np.random.default_rng(0)
    base = np.zeros((1, 300, 6), np.float32)
    base[0, :, :4] = rng.uniform(0, 100, (300, 4))
    base[0, :, 4] = np.linspace(0.9, 0.001, 300)
    base[0, :, 5] = rng.integers(0, 3, 300)
    assert _det_parity(base, base.copy())["det_confident"] > 0
    shuffled = base.copy()
    shuffled[0] = shuffled[0][rng.permutation(300)]  # same rows, different order
    _det_parity(base, shuffled)
    tied = base.copy()
    tied[0, 250:, 4] = 0.001  # tie at the cutoff: the pick among tied rows is arbitrary
    other = tied.copy()
    other[0, 250:, :4] += 50
    _det_parity(tied, other)
    moved = base.copy()
    moved[0, 0, 0] += 5.0  # a confident box moved
    with pytest.raises(ExportParityError, match="boxes"):
        _det_parity(base, moved)
    wrong_cls = base.copy()
    wrong_cls[0, 0, 5] = (wrong_cls[0, 0, 5] + 1) % 3
    with pytest.raises(ExportParityError, match="class"):
        _det_parity(base, wrong_cls)
