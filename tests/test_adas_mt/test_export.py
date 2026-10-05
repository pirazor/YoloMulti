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
    TRT_OPS, DeployWrapper, ExportParityError, ReshapePixelShuffle, _det_check, audit_ops, export_onnx,
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


def test_det_check_is_robust_to_ties_but_not_to_real_differences():
    rng = np.random.default_rng(0)
    nc, anchors = 3, 400
    dense = np.concatenate([rng.uniform(0, 600, (anchors, 4)), rng.uniform(0, 1, (anchors, nc)) ** 2], 1).astype(np.float32)
    from adas_mt.deploy.postprocess import topk_det

    want = topk_det(dense, 300)[None]
    dense = dense[None]
    assert _det_check(want.copy(), dense, want)["det_confident"] > 0
    shuffled = want.copy()
    shuffled[0] = shuffled[0][rng.permutation(300)]  # same rows, any order
    _det_check(shuffled, dense, want)
    low = dense.copy()  # exact score ties among the confident rows: which tied anchor is picked is arbitrary
    low[0, :, 4:] = np.round(low[0, :, 4:], 1)
    tied_want = topk_det(low[0], 300)[None]
    tied_got = tied_want.copy()
    tied_got[0] = tied_got[0][np.lexsort((tied_got[0][:, 0], tied_got[0][:, 5], -tied_got[0][:, 4]))]  # another tie order
    _det_check(tied_got, low, tied_want)
    moved = want.copy()
    moved[0, 0, 0] += 5.0  # a confident box moved by 5 px: no PyTorch prediction has it
    with pytest.raises(ExportParityError, match="boxes"):
        _det_check(moved, dense, want)
    wrong_cls = want.copy()
    wrong_cls[0, 0, 5] = (wrong_cls[0, 0, 5] + 1) % nc
    with pytest.raises(ExportParityError, match="class ids or scores"):
        _det_check(wrong_cls, dense, want)
    inflated = want.copy()
    inflated[0, :, 4] = np.minimum(inflated[0, :, 4] + 0.2, 1.0)  # scores systematically off
    with pytest.raises(ExportParityError):
        _det_check(inflated, dense, want)


# --------------------------------------------------------------------------- findings of the independent review
def test_raw_head_graph_has_no_topk_and_the_host_topk_equals_the_in_graph_one(model, tmp_path):
    from adas_mt.deploy.postprocess import topk_det_batch

    a = export_onnx(model, tmp_path / "topk.onnx", imgsz=HW)
    b = export_onnx(model, tmp_path / "raw.onnx", imgsz=HW, det_head="raw")
    assert "TopK" in a.ops and "TopK" not in b.ops and "GatherElements" not in b.ops and "Mod" not in b.ops
    assert b.audit["unsupported"] == [] and b.meta["det_head"] == "raw"
    anchors = sum((HW[0] // s) * (HW[1] // s) for s in (8, 16, 32))
    assert b.meta["outputs"]["det"]["shape"] == [1, anchors, 4 + 3] and a.meta["outputs"]["det"]["shape"] == [1, 300, 6]
    x = structured_frame(HW).numpy()
    raw = _session(b.onnx).run(["det"], {"images": x})[0]
    ref = _session(a.onnx).run(["det"], {"images": x})[0]
    assert raw.shape == (1, anchors, 7) and (raw[..., 4:] >= 0).all() and (raw[..., 4:] <= 1).all()
    host = topk_det_batch(raw, 300)
    assert np.allclose(np.sort(host[0, :, 4]), np.sort(ref[0, :, 4]), atol=1e-4)  # same selection, any tie order
    assert b.parity["det_confident"] > 0 and "raw_box_max_abs" in b.parity


def test_raw_head_with_dynamic_batch_and_other_options(model, tmp_path):
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW, det_head="raw", dynamic=True, seg_dtype="uint8",
                    decompose_pixel_shuffle=True)
    out = _session(r.onnx).run(None, {"images": torch.cat([structured_frame(HW, seed=i) for i in range(3)]).numpy()})
    assert out[0].shape[0] == 3 and out[1].dtype == np.uint8
    with pytest.raises(ValueError, match="det_head"):
        export_onnx(model, tmp_path / "x.onnx", imgsz=HW, det_head="nms")


def test_grouped_topk_graph_is_equivalent(model, tmp_path):
    """Ultralytics' TensorRT export uses a grouped exact top-k: more, smaller TopK layers; same detections."""
    a = export_onnx(model, tmp_path / "g.onnx", imgsz=HW, trt_topk=True)
    b = export_onnx(model, tmp_path / "u.onnx", imgsz=HW, trt_topk=False)
    assert a.ops["TopK"] > b.ops["TopK"] and a.meta["topk_groups"] == 8 and b.meta["topk_groups"] == 1
    x = structured_frame(HW).numpy()
    da = _session(a.onnx).run(["det"], {"images": x})[0]
    db = _session(b.onnx).run(["det"], {"images": x})[0]
    assert np.allclose(np.sort(da[0, :, 4]), np.sort(db[0, :, 4]), atol=1e-5)


def test_export_restores_the_detect_head_even_on_failure(model, tmp_path):
    det = model.model[-1]
    before = {k: det.__dict__.get(k, "missing") for k in ("export", "format", "postprocess")}
    export_onnx(model, tmp_path / "a.onnx", imgsz=HW, det_head="raw")
    assert {k: det.__dict__.get(k, "missing") for k in before} == before
    with pytest.raises(ValueError):
        export_onnx(model, tmp_path / "b.onnx", imgsz=(100, 160))
    assert {k: det.__dict__.get(k, "missing") for k in before} == before


def test_det_shape_when_there_are_fewer_anchors_than_max_det(model, tmp_path):
    r = export_onnx(model, tmp_path / "tiny.onnx", imgsz=(32, 64))  # 4*8 + 2*4 + 1*2 = 42 anchors
    assert r.meta["outputs"]["det"]["shape"] == [1, 42, 6]
    assert _session(r.onnx).run(["det"], {"images": np.zeros((1, 3, 32, 64), np.float32)})[0].shape == (1, 42, 6)


def test_parity_inputs_exercise_every_head(model, tmp_path):
    """A constant class map would make its parity check vacuous."""
    x = structured_frame(HW).numpy()
    r = export_onnx(model, tmp_path / "m.onnx", imgsz=HW)
    da, ll = _session(r.onnx).run(["da", "ll"], {"images": x})
    assert len(np.unique(da)) > 1 and len(np.unique(ll)) > 1 and r.parity["det_confident"] > 0


def test_opset_above_17_warns(model, tmp_path, caplog):
    with caplog.at_level("WARNING", logger="adas_mt.export"):
        export_onnx(model, tmp_path / "m.onnx", imgsz=HW, opset=18, verify=False)
    assert "opset 18" in caplog.text
