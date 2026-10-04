"""Export a trained checkpoint to a deployable ONNX graph.

The graph is the *deployed* network, nothing else: fused Conv+BN, one-to-one detection branch only
(NMS-free), no distillation projector, no auxiliary DA classifier. Post-processing that belongs on the GPU is
baked in so the host never touches the 3 x H x W logits::

    images  float32 (B, 3, H, W)   RGB, /255, letterboxed (see adas_mt.deploy.preprocess)
    det     float32 (B, 300, 6)    x1, y1, x2, y2, score, class   (network pixels, already NMS-free)
    da      int32   (B, H, W)      drivable-area class per pixel (0 = background)
    ll      int32   (B, H, W)      lane class per pixel (0 = background)

``seg_dtype`` selects ``int32`` (default; every TensorRT version), ``uint8`` (4x smaller output copy; TensorRT >= 10
only) or ``logits`` (float32 (B, C, H, W), for debugging and calibration studies).
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from adas_mt.deploy.meta import meta_path, read_meta  # noqa: F401  (re-exported)

LOG = logging.getLogger("adas_mt.export")

SEG_DTYPES = ("int32", "uint8", "logits")
INPUT_NAME = "images"
OUTPUT_NAMES = ("det", "da", "ll")
META_VERSION = 1

# ONNX ops the TensorRT ONNX parser handles in every supported version (8.6 .. 10.x). The set the exported graph is
# allowed to contain: anything else is reported at export time, long before an engine build fails on the device.
TRT_OPS = frozenset({
    "Add", "ArgMax", "Cast", "Clip", "Concat", "Constant", "ConstantOfShape", "Conv", "Div", "Equal", "Exp", "Expand",
    "Flatten", "Floor", "Gather", "GatherElements", "Greater", "Identity", "Less", "MatMul", "MaxPool", "Mod", "Mul",
    "Neg", "Pad", "Pow", "ReduceMax", "ReduceMean", "Relu", "Reshape", "Resize", "Shape", "Sigmoid", "Slice", "Softmax",
    "Split", "Sqrt", "Squeeze", "Sub", "Tile", "TopK", "Transpose", "Unsqueeze", "Where",
})
# Valid ONNX but not verifiable without a device: ``DepthToSpace(mode=CRD)`` is the lane head's PixelShuffle.
# ``decompose_pixel_shuffle=True`` replaces it with Reshape + Transpose + Reshape (always supported).
NEEDS_DEVICE_CHECK = frozenset({"DepthToSpace"})


class ExportParityError(RuntimeError):
    """The exported graph does not reproduce the PyTorch model."""


# --------------------------------------------------------------------------- graph pieces
class ReshapePixelShuffle(nn.Module):
    """``nn.PixelShuffle`` as Reshape + Transpose + Reshape (identical result, no ``DepthToSpace`` op)."""

    def __init__(self, r: int):
        super().__init__()
        self.r = int(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r = self.r
        x = x.reshape(b, c // (r * r), r, r, h, w).permute(0, 1, 4, 2, 5, 3)
        return x.reshape(b, c // (r * r), h * r, w * r)


class DeployWrapper(nn.Module):
    """``images -> (det, da, ll)`` with the argmax (and its integer cast) inside the graph."""

    def __init__(self, model: nn.Module, seg_dtype: str = "int32"):
        super().__init__()
        if seg_dtype not in SEG_DTYPES:
            raise ValueError(f"seg_dtype must be one of {SEG_DTYPES}, got {seg_dtype!r}")
        self.model = model
        self.seg_dtype = seg_dtype

    def _seg(self, logits: torch.Tensor) -> torch.Tensor:
        if self.seg_dtype == "logits":
            return logits
        cls = logits.argmax(1)
        return cls.to(torch.uint8 if self.seg_dtype == "uint8" else torch.int32)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        det = out["det"]
        det = det[0] if isinstance(det, (tuple, list)) else det
        return det, self._seg(out["da"]), self._seg(out["ll"])


# --------------------------------------------------------------------------- model loading
def load_deploy_model(weights: str | Path | nn.Module) -> nn.Module:
    """Checkpoint (or model) -> fused, float32, eval-mode copy without any training-only module."""
    from adas_mt.nn.model import MultiTaskModel

    if isinstance(weights, nn.Module):
        model = weights
    else:
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        model = (ckpt.get("ema") or ckpt["model"]) if isinstance(ckpt, dict) else ckpt
    if not isinstance(model, MultiTaskModel):
        raise TypeError(f"{weights} does not contain an adas_mt MultiTaskModel (got {type(model).__name__}); "
                        "was it trained with `python -m adas_mt train`?")
    model = deepcopy(model).float().eval()
    det = model.model[-1]
    if getattr(det, "one2one_cv2", None) is None and not getattr(det, "end2end", False):
        raise ValueError("the checkpoint has no NMS-free (end2end) detection head; only YOLO26-style models are supported")
    model.strip_training_only()
    model.fuse(verbose=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _resolve_imgsz(weights, imgsz: Optional[Sequence[int]]) -> tuple[int, int]:
    from adas_mt.engine.config import MTConfig

    if imgsz:
        out = tuple(int(v) for v in imgsz)
    else:
        run_cfg = MTConfig.find_run_cfg(weights) if isinstance(weights, (str, Path)) else None
        if run_cfg is None:
            out = tuple(MTConfig().imgsz)
            LOG.warning("no mt.yaml next to %s: exporting at %s (pass imgsz to be sure; it must equal the training geometry)",
                        weights, out)
        else:
            out = tuple(MTConfig.load(run_cfg).imgsz)
    if len(out) != 2 or any(v <= 0 or v % 32 for v in out):
        raise ValueError(f"imgsz must be (h, w), both positive multiples of 32, got {out}")
    return out  # type: ignore[return-value]


# --------------------------------------------------------------------------- audits / io
def onnx_ops(path: str | Path) -> Counter:
    import onnx

    return Counter(n.op_type for n in onnx.load(str(path)).graph.node)


def audit_ops(ops: Counter | Dict[str, int]) -> Dict[str, list]:
    """``{'unsupported': [...], 'check_on_device': [...]}`` against :data:`TRT_OPS`."""
    return {
        "unsupported": sorted(o for o in ops if o not in TRT_OPS and o not in NEEDS_DEVICE_CHECK),
        "check_on_device": sorted(o for o in ops if o in NEEDS_DEVICE_CHECK),
    }


def _names(d: Any) -> list:
    if isinstance(d, dict):
        return [d[k] for k in sorted(d)]
    return list(d)


def build_meta(model: nn.Module, imgsz, batch: int, dynamic: bool, opset: int, seg_dtype: str, source: str,
               decompose_pixel_shuffle: bool) -> Dict[str, Any]:
    import ultralytics

    det = model.model[-1]
    h, w = imgsz
    b: Any = "dynamic" if dynamic else batch
    seg_shape = [b, h, w] if seg_dtype != "logits" else [b, "C", h, w]
    return {
        "format": "adas_mt", "version": META_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(source),
        "imgsz": [h, w], "batch": batch, "dynamic_batch": bool(dynamic), "opset": opset,
        "seg_dtype": seg_dtype, "decompose_pixel_shuffle": bool(decompose_pixel_shuffle),
        "max_det": int(det.max_det), "nc": int(det.nc),
        "names": _names(model.names), "da_names": list(model.da_names), "ll_names": list(model.ll_names),
        "da_classes": int(model.da_classes), "ll_classes": int(model.ll_classes),
        # substrings of the ONNX node names of each component (used to pin precision when building INT8 engines)
        "layers": {"det": f"/model.{len(model.model) - 1}/", "da": "/da_head/", "ll": "/ll_head/"},
        "input": {"name": INPUT_NAME, "dtype": "float32", "layout": "NCHW", "color": "RGB", "range": [0.0, 1.0]},
        "preprocess": {
            "resize": "long side to max(imgsz), bilinear (cv2.INTER_LINEAR), never upscale",
            "pad": "centred letterbox, value 114", "equals": "validation pipeline (tests/test_adas_mt/test_deploy_preprocess.py)",
        },
        "outputs": {
            "det": {"shape": [b, int(det.max_det), 6], "dtype": "float32",
                    "columns": "x1 y1 x2 y2 score class, network pixels, NMS-free (one-to-one head)"},
            "da": {"shape": seg_shape, "dtype": "float32" if seg_dtype == "logits" else seg_dtype},
            "ll": {"shape": seg_shape, "dtype": "float32" if seg_dtype == "logits" else seg_dtype},
        },
        "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__},
    }


def _embed_meta(onnx_path: Path, meta: Dict[str, Any]) -> None:
    import onnx

    m = onnx.load(str(onnx_path))
    for p in list(m.metadata_props):
        if p.key == "adas_mt":
            m.metadata_props.remove(p)
    prop = m.metadata_props.add()
    prop.key, prop.value = "adas_mt", json.dumps(meta)
    onnx.save(m, str(onnx_path))


# --------------------------------------------------------------------------- verification
def _smooth_random(batch: int, imgsz) -> torch.Tensor:
    """Spatially varying seeded test input (iid per-pixel noise averages out in the first convolutions)."""
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(0)
    low = torch.rand(batch, 3, max(2, imgsz[0] // 16), max(2, imgsz[1] // 16), generator=g)
    return F.interpolate(low, size=tuple(imgsz), mode="bilinear", align_corners=False)


def _det_parity(got: np.ndarray, want: np.ndarray, conf: float = 0.05) -> Dict[str, float]:
    """Tie-robust comparison of two (B, N, 6) detection tensors.

    Top-k breaks exact (and, across runtimes, near-exact) score ties arbitrarily, so row ``i`` of one tensor need not
    be row ``i`` of the other. What must agree: the sorted score vector, and the confident rows (``score > conf`` and
    strictly above the top-k cutoff) after a canonical sort."""
    stats = {"det_score_max_abs": 0.0, "det_box_max_abs": 0.0, "det_confident": 0.0}
    for g, w in zip(got, want):
        sg, sw = np.sort(g[:, 4])[::-1], np.sort(w[:, 4])[::-1]
        stats["det_score_max_abs"] = max(stats["det_score_max_abs"], float(np.abs(sg - sw).max()))
        # rows at the top-k cutoff are an arbitrary pick among tied scores: only rows strictly above it are comparable
        floor = max(conf, float(max(g[:, 4].min(), w[:, 4].min())) + 1e-4)
        gc, wc = g[g[:, 4] > floor], w[w[:, 4] > floor]
        if len(gc) != len(wc):
            # a score within float noise of the threshold may fall on either side: allow a one-row slack
            if abs(len(gc) - len(wc)) > 1:
                raise ExportParityError(f"{len(gc)} confident detections in ONNX vs {len(wc)} in PyTorch")
            continue
        order = lambda a: np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0], a[:, 5], -a[:, 4]))  # noqa: E731
        gc, wc = gc[order(gc)], wc[order(wc)]
        if len(gc):
            stats["det_box_max_abs"] = max(stats["det_box_max_abs"], float(np.abs(gc[:, :4] - wc[:, :4]).max()))
            if not (gc[:, 5] == wc[:, 5]).all():
                raise ExportParityError("class ids of the confident detections differ between ONNX and PyTorch")
        stats["det_confident"] += len(gc)
    if stats["det_score_max_abs"] > 1e-3:
        raise ExportParityError(f"detection scores differ from PyTorch: max abs diff {stats['det_score_max_abs']:.4g}")
    if stats["det_box_max_abs"] > 0.1:  # pixels
        raise ExportParityError(f"detection boxes differ from PyTorch: max abs diff {stats['det_box_max_abs']:.4g} px")
    return stats


@torch.no_grad()
def verify_onnx(wrapper: nn.Module, onnx_path: str | Path, imgsz, batch: int = 1, seg_dtype: str = "int32",
                x: Optional[torch.Tensor] = None, min_agree: float = 0.999) -> Dict[str, float]:
    """Run PyTorch and ONNX Runtime on the same input and compare every output.

    Detections are compared tie-robustly (:func:`_det_parity`); class maps may differ on exact argmax ties only
    (>= ``min_agree`` of the pixels must agree); logits must match numerically. Pass a real, letterboxed frame as
    ``x`` for a meaningful check of a trained model (the default is a smooth random image)."""
    import onnxruntime as ort

    if x is None:
        x = _smooth_random(batch, imgsz)
    elif x.shape[0] != batch:
        x = x[:1].expand(batch, -1, -1, -1).contiguous()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = dict(zip(OUTPUT_NAMES, sess.run(list(OUTPUT_NAMES), {INPUT_NAME: x.numpy()})))
    want = dict(zip(OUTPUT_NAMES, (t.numpy() for t in wrapper(x))))
    stats = _det_parity(got["det"], want["det"])
    for k in ("da", "ll"):
        if seg_dtype == "logits":
            diff = float(np.abs(got[k] - want[k]).max())
            scale = max(1.0, float(np.abs(want[k]).max()))
            stats[f"{k}_max_abs"] = diff
            if diff > 1e-3 * scale:  # relative to the logit range: ~100 fp32 layers drift by ~1e-4 relative
                raise ExportParityError(f"{k} logits differ from PyTorch: max abs diff {diff:.4g} (logit range {scale:.3g})")
        else:
            agree = float((got[k] == want[k]).mean())
            stats[f"{k}_agree"] = agree
            if agree < min_agree:
                raise ExportParityError(f"{k} class map agrees with PyTorch on only {agree:.4%} of the pixels")
    return stats


# --------------------------------------------------------------------------- export
@dataclass
class ExportResult:
    onnx: Path
    meta_file: Path
    meta: Dict[str, Any]
    parity: Dict[str, float] = field(default_factory=dict)
    ops: Dict[str, int] = field(default_factory=dict)
    audit: Dict[str, list] = field(default_factory=dict)


def export_onnx(
    weights: str | Path | nn.Module,
    out: str | Path | None = None,
    imgsz: Optional[Sequence[int]] = None,
    batch: int = 1,
    dynamic: bool = False,
    seg_dtype: str = "int32",
    opset: int = 17,
    simplify: bool = True,
    decompose_pixel_shuffle: bool = False,
    verify: bool = True,
    verify_image: str | Path | None = None,
) -> ExportResult:
    """Export ``weights`` (a ``.pt`` written by the trainer, or a model) to ``out`` (default: next to the weights).

    ``verify_image``: a real frame for the PyTorch-vs-ONNX parity check (default: seeded noise, which exercises the
    segmentation heads but yields no confident detections)."""
    if opset < 13:
        raise ValueError("opset must be >= 13 (ArgMax/Resize semantics the graph relies on)")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    imgsz = _resolve_imgsz(weights, imgsz)
    model = load_deploy_model(weights)
    if (model.da_head.aux is not None) or hasattr(model, "kd_proj"):
        raise RuntimeError("training-only modules survived strip_training_only()")
    if decompose_pixel_shuffle:
        model.ll_head.shuffle = ReshapePixelShuffle(model.ll_head.SCALE)
    if out is None:
        if not isinstance(weights, (str, Path)):
            raise ValueError("out is required when exporting an in-memory model")
        out = Path(weights).with_suffix(".onnx")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    wrapper = DeployWrapper(model, seg_dtype).eval()
    det = model.model[-1]
    prev_export, det.export = getattr(det, "export", False), True  # Detect returns the decoded tensor only
    try:
        x = torch.zeros(batch, 3, *imgsz)
        with torch.no_grad():
            wrapper(x)  # warm-up / shape check before tracing
        axes = {INPUT_NAME: {0: "batch"}, "det": {0: "batch"}, "da": {0: "batch"}, "ll": {0: "batch"}} if dynamic else None
        with warnings.catch_warnings():
            # the tracer's "converted a tensor to a Python bool" notes are about shape constants (anchors, top-k size)
            # that are intentionally frozen for the static spatial size; the parity check below covers the result
            warnings.simplefilter("ignore", (torch.jit.TracerWarning, DeprecationWarning))
            torch.onnx.export(
                wrapper, (x,), str(out), input_names=[INPUT_NAME], output_names=list(OUTPUT_NAMES), opset_version=opset,
                do_constant_folding=True, dynamic_axes=axes, dynamo=False,
            )
        if simplify:
            _simplify(out)
        _pin_io_shapes(out, imgsz, "batch" if dynamic else batch, int(det.max_det), seg_dtype,
                       (model.da_classes, model.ll_classes))
        meta = build_meta(model, imgsz, batch, dynamic, opset, seg_dtype, str(weights) if isinstance(weights, (str, Path)) else "<model>",
                          decompose_pixel_shuffle)
        _embed_meta(out, meta)
        meta_file = meta_path(out)
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        ops = dict(onnx_ops(out))
        audit = audit_ops(ops)
        if audit["unsupported"]:
            LOG.warning("ops outside the TensorRT-safe set: %s (engine build may fail)", audit["unsupported"])
        if audit["check_on_device"]:
            LOG.info("op(s) to confirm on the device: %s (use --decompose-pixel-shuffle if the engine build rejects them)",
                     audit["check_on_device"])
        parity: Dict[str, float] = {}
        if verify:
            vx = _load_verify_image(verify_image, imgsz) if verify_image else None
            parity = verify_onnx(wrapper, out, imgsz, batch=batch, seg_dtype=seg_dtype, x=vx)
            if dynamic and batch != 2:  # the batch axis must really be free
                parity.update({f"b2_{k}": v for k, v in
                               verify_onnx(wrapper, out, imgsz, batch=2, seg_dtype=seg_dtype, x=vx).items()})
            if not parity.get("det_confident"):
                LOG.warning("the verification input produced no confident detections: boxes were only checked through "
                            "the score vector. Pass verify_image=<a real frame> for a full check of a trained model.")
            LOG.info("ONNX parity vs PyTorch: %s", {k: round(v, 6) for k, v in parity.items()})
    finally:
        det.export = prev_export
    LOG.info("exported %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return ExportResult(out, meta_file, meta, parity, ops, audit)


def _load_verify_image(path: str | Path, imgsz) -> torch.Tensor:
    import cv2

    from adas_mt.deploy.preprocess import preprocess

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read verify image {path}")
    return torch.from_numpy(preprocess(img, imgsz)[0])[None]


def _pin_io_shapes(path: Path, imgsz, batch, max_det: int, seg_dtype: str, seg_classes) -> None:
    """Rewrite the declared input/output shapes. The tracer leaves stale symbolic dims on dynamic-batch graphs
    (even ``[batch, batch, W]``); consumers and ``onnx.checker`` should see the real ones."""
    import onnx

    h, w = imgsz
    shapes = {INPUT_NAME: [batch, 3, h, w], "det": [batch, max_det, 6]}
    for k, c in zip(("da", "ll"), seg_classes):
        shapes[k] = [batch, c, h, w] if seg_dtype == "logits" else [batch, h, w]
    m = onnx.load(str(path))
    for vi in list(m.graph.input) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        del dims[:]
        for d in shapes[vi.name]:
            dim = dims.add()
            if isinstance(d, str):
                dim.dim_param = d
            else:
                dim.dim_value = int(d)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def _simplify(path: Path) -> None:
    """onnxslim in place; a failure keeps the (valid) unsimplified graph."""
    try:
        import onnx
        from onnxslim import slim

        slimmed = slim(onnx.load(str(path)))
        onnx.checker.check_model(slimmed)
        onnx.save(slimmed, str(path))
    except Exception as e:  # noqa: BLE001 - optional optimisation, never fatal
        LOG.warning("onnxslim skipped (%s: %s)", type(e).__name__, e)
