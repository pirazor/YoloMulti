"""ONNX export of SPARC graphs (multi-input), onnxruntime parity and the TensorRT op audit.

The op audit extends ``adas_mt.export.TRT_OPS`` with the ops SPARC needs. Ops whose TensorRT support or FP16
behaviour can only be confirmed on the device (``NEEDS_DEVICE_CHECK``) are exactly what the Phase-0 probes exercise:
``GridSample`` (deformable attention, IPM), ``ScatterElements`` with ``reduction="add"`` (depth splat) and
``DepthToSpace`` (convex upsampling of the stereo head).
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from adas_mt.export import TRT_OPS as _ADAS_TRT_OPS

LOG = logging.getLogger("sparc.export")

# In the TensorRT ONNX parser's support matrix for 8.6 .. 10.x (opset <= 17 semantics)
SPARC_TRT_OPS = frozenset(_ADAS_TRT_OPS | {
    "Abs", "And", "AveragePool", "BatchNormalization", "ConvTranspose", "Cos", "Erf", "GreaterOrEqual",
    "LayerNormalization", "LessOrEqual", "Log", "Max", "Min", "Not", "Or", "Range", "ReduceSum", "Sin",
    "Gemm", "Einsum", "ReduceMin", "Sign", "Reciprocal",
})
NEEDS_DEVICE_CHECK = frozenset({"GridSample", "ScatterElements", "DepthToSpace", "ScatterND"})


class ExportParityError(RuntimeError):
    pass


def onnx_ops(path: str | Path) -> Counter:
    import onnx

    return Counter(n.op_type for n in onnx.load(str(path)).graph.node)


def audit_ops(ops: Counter | Dict[str, int]) -> Dict[str, list]:
    return {"unsupported": sorted(o for o in ops if o not in SPARC_TRT_OPS and o not in NEEDS_DEVICE_CHECK),
            "check_on_device": sorted(o for o in ops if o in NEEDS_DEVICE_CHECK)}


def scatter_reductions(path: str | Path) -> List[str]:
    """``reduction`` attribute of every ScatterElements node ("none" when absent)."""
    import onnx

    out = []
    for n in onnx.load(str(path)).graph.node:
        if n.op_type == "ScatterElements":
            red = [a.s.decode() for a in n.attribute if a.name == "reduction"]
            out.append(red[0] if red else "none")
    return out


# --------------------------------------------------------------------------- comparison policies
def compare(name: str, got: np.ndarray, ref: np.ndarray, policy: str) -> Dict[str, float]:
    """Numerical agreement of one output under its policy.

    * ``dense``: relative L2 error ``|got - ref| / |ref|`` and max abs error;
    * ``label``: fraction of equal elements (class maps);
    * ``ranked``: agents whose order comes from a top-k: compare the sorted score column (index 4) and the
      fraction of the reference's top-50 boxes matched by IoU > 0.9 (fp16 can reorder near-ties);
    * ``rowset``: per-agent vectors in top-k order (ID embeddings): fraction of reference rows that have a row in the
      output within 5 % relative L2 distance, whatever its position.
    """
    got = np.asarray(got, dtype=np.float64).reshape(ref.shape) if got.size == ref.size else np.asarray(got)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        return {"shape_mismatch": 1.0}
    if policy == "label":
        return {"agree": float((got == ref).mean())}
    if policy == "rowset":
        g, r = got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1])
        d = np.sqrt(((r[:, None, :] - g[None, :, :]) ** 2).sum(-1)).min(1) / (np.linalg.norm(r, axis=1) + 1e-12)
        return {"row_recall": float((d < 0.05).mean()), "row_rel_max": float(d.max())}
    if policy == "ranked":
        gs, rs = np.sort(got[..., 4].ravel())[::-1], np.sort(ref[..., 4].ravel())[::-1]
        out = {"score_max_abs": float(np.abs(gs - rs).max())}
        g, r = got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1])
        top = r[np.argsort(-r[:, 4])[:50], :4]
        out["top50_box_recall"] = float(np.mean([_best_iou(b, g[:, :4]) > 0.9 for b in top]))
        return out
    den = np.linalg.norm(ref) + 1e-12
    return {"rel_l2": float(np.linalg.norm(got - ref) / den), "max_abs": float(np.abs(got - ref).max())}


def _best_iou(b: np.ndarray, boxes: np.ndarray) -> float:
    x1, y1 = np.maximum(b[0], boxes[:, 0]), np.maximum(b[1], boxes[:, 1])
    x2, y2 = np.minimum(b[2], boxes[:, 2]), np.minimum(b[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1])
    ab = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return float((inter / np.maximum(a + ab - inter, 1e-9)).max()) if len(boxes) else 0.0


OUTPUT_POLICY = {"agents": "ranked", "agent_emb": "rowset", "da": "label"}


def policy_for(name: str) -> str:
    return OUTPUT_POLICY.get(name, "dense")


# --------------------------------------------------------------------------- export
@dataclass
class ExportInfo:
    path: str
    inputs: Dict[str, list]
    outputs: Dict[str, list]
    ops: Dict[str, int]
    audit: Dict[str, list]
    scatter_reductions: List[str]
    parity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ort_cpu_ms: Optional[float] = None
    mbytes: float = 0.0


def run_ort(path: str | Path, inputs: Dict[str, np.ndarray], runs: int = 0, threads: int = 0):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, inputs)
    ms = None
    if runs:
        t0 = time.perf_counter()
        for _ in range(runs):
            sess.run(names, inputs)
        ms = (time.perf_counter() - t0) / runs * 1e3
    return dict(zip(names, outs)), ms


def export_graph(module: nn.Module, inputs: Dict[str, torch.Tensor], path: str | Path, output_names: Sequence[str],
                 opset: int = 17, simplify: bool = True, check: bool = True, cpu_runs: int = 0,
                 tol: Dict[str, float] | None = None) -> ExportInfo:
    """Export ``module(*inputs.values())`` with static shapes; verify against PyTorch with onnxruntime.

    ``tol``: max ``rel_l2`` per dense output (default 1e-3) / min agreement for labels (0.999)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval()
    args = tuple(inputs.values())
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", (torch.jit.TracerWarning, DeprecationWarning, UserWarning, FutureWarning))
        torch.onnx.export(module, args, str(path), input_names=list(inputs), output_names=list(output_names),
                          opset_version=opset, do_constant_folding=True, **extra)
        want = module(*args)
    if simplify:
        _simplify(path)
    ops = onnx_ops(path)
    info = ExportInfo(
        path=str(path), inputs={k: list(v.shape) for k, v in inputs.items()},
        outputs={k: list(v.shape) for k, v in zip(output_names, want)}, ops=dict(sorted(ops.items())),
        audit=audit_ops(ops), scatter_reductions=scatter_reductions(path), mbytes=path.stat().st_size / 2**20,
    )
    if info.audit["unsupported"]:
        LOG.warning("%s: ops outside the TensorRT whitelist: %s", path.name, info.audit["unsupported"])
    if check or cpu_runs:
        np_in = {k: v.numpy() for k, v in inputs.items()}
        got, info.ort_cpu_ms = run_ort(path, np_in, runs=cpu_runs)
        tol = tol or {}
        bad = []
        for name, w in zip(output_names, want):
            m = compare(name, got[name], w.numpy(), policy_for(name))
            info.parity[name] = m
            if "rel_l2" in m and m["rel_l2"] > tol.get(name, 1e-3):
                bad.append(f"{name}: rel_l2 {m['rel_l2']:.2e}")
            if "agree" in m and m["agree"] < tol.get(name, 0.999):
                bad.append(f"{name}: agreement {m['agree']:.4f}")
            if "top50_box_recall" in m and m["top50_box_recall"] < tol.get(name, 0.98):
                bad.append(f"{name}: top-50 box recall {m['top50_box_recall']:.2f}")
            if "row_recall" in m and m["row_recall"] < tol.get(name, 0.98):
                bad.append(f"{name}: row recall {m['row_recall']:.2f}")
        if check and bad:
            raise ExportParityError(f"{path.name}: ONNX does not reproduce PyTorch: " + "; ".join(bad))
    return info


def _simplify(path: Path) -> None:
    try:
        import onnx
        from onnxslim import slim

        slimmed = slim(onnx.load(str(path)))
        onnx.checker.check_model(slimmed)
        onnx.save(slimmed, str(path))
    except Exception as e:  # noqa: BLE001 - optional optimisation
        LOG.warning("onnxslim skipped (%s: %s)", type(e).__name__, e)


def embed_meta(path: str | Path, meta: Dict) -> None:
    import onnx

    m = onnx.load(str(path))
    for p in list(m.metadata_props):
        if p.key == "sparc":
            m.metadata_props.remove(p)
    prop = m.metadata_props.add()
    prop.key, prop.value = "sparc", json.dumps(meta)
    onnx.save(m, str(path))
