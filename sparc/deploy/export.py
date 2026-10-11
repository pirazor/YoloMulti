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
    "Gemm", "Einsum", "ReduceMin", "Sign", "Reciprocal", "Tanh", "HardSigmoid", "LeakyRelu", "PRelu",
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
# Column roles of the table outputs: "coord" (box edges, pixels: matched by IoU), "score", "id" (argmax ids: compared by
# agreement), "p" (probabilities), "val" (regressed values). Agents are matched to the reference by box IoU before
# the columns are compared (fp16 can reorder near-tied top-k entries); lanes keep their slot order.
AGENT_ROLES = ("coord",) * 4 + ("score", "id") + ("val",) * 8
ELEMENT_ROLES = ("coord",) * 4 + ("score", "id", "id", "p", "id", "p")
LANE_META_ROLES = ("p", "id", "p", "id", "p", "id", "p", "id", "p")  # exist, order (value), p, dir, p, cont, p, type, p
TABLE_ROLES = {"agents": AGENT_ROLES, "elements": ELEMENT_ROLES}
OUTPUT_POLICY = {"agents": "agents", "elements": "elements", "agent_emb": "rowset", "lane_meta": "lane_meta", "da": "label"}


def policy_for(name: str) -> str:
    return OUTPUT_POLICY.get(name, "dense")


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _columns(got: np.ndarray, ref: np.ndarray, roles) -> Dict[str, float]:
    """Per-role agreement of two (N, C) tables whose rows correspond."""
    out: Dict[str, float] = {}
    for role in ("id", "p", "val", "score"):
        cols = [i for i, r in enumerate(roles) if r == role]
        if not cols:
            continue
        g, r = got[:, cols], ref[:, cols]
        if role == "id":
            out["id_agree"] = float((g == r).mean())
        else:
            out[f"{role}_max_abs"] = float(np.abs(g - r).max())
            if role == "val":
                out["val_rel_l2"] = float(np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-12))
    return out


def compare(name: str, got: np.ndarray, ref: np.ndarray, policy: str) -> Dict[str, float]:
    """Numerical agreement of one output under its policy.

    * ``dense``: relative L2 error ``|got - ref| / |ref|`` and max abs error;
    * ``label``: fraction of equal elements (class maps);
    * ``agents`` / ``elements``: the reference's top-50 boxes (by score) are matched to output boxes by IoU > 0.9
      (``top50_box_recall``); on the matched rows the id columns must agree and the probability / value / score
      columns are compared numerically (roles in :data:`TABLE_ROLES`);
    * ``lane_meta``: per slot, id columns by agreement, probability columns by max abs error;
    * ``rowset``: per-agent vectors (ID embeddings): fraction of reference rows that have a row in the output within
      5 % relative L2 distance, whatever its position.
    """
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.size != ref.size:
        return {"shape_mismatch": 1.0}
    got = got.reshape(ref.shape)
    if policy == "label":
        return {"agree": float((got == ref).mean())}
    if policy == "rowset":
        g, r = got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1])
        d = np.sqrt(((r[:, None, :] - g[None, :, :]) ** 2).sum(-1)).min(1) / (np.linalg.norm(r, axis=1) + 1e-12)
        return {"row_recall": float((d < 0.05).mean()), "row_rel_max": float(d.max())}
    if policy == "lane_meta":
        return _columns(got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1]), LANE_META_ROLES)
    if policy in TABLE_ROLES:
        g, r = got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1])
        top = np.argsort(-r[:, 4])[:50]
        iou = _iou_matrix(r[top, :4], g[:, :4])
        best = iou.argmax(1)
        matched = iou[np.arange(len(top)), best] > 0.9
        out = {"top50_box_recall": float(matched.mean())}
        if matched.any():
            out.update(_columns(g[best[matched]], r[top[matched]], TABLE_ROLES[policy]))
        return out
    den = np.linalg.norm(ref) + 1e-12
    return {"rel_l2": float(np.linalg.norm(got - ref) / den), "max_abs": float(np.abs(got - ref).max())}


# thresholds: (key, limit, "min" | "max")
PARITY_LIMITS_ORT = (("rel_l2", 1e-3, "max"), ("agree", 0.999, "min"), ("top50_box_recall", 0.98, "min"),
                     ("row_recall", 0.98, "min"), ("id_agree", 0.99, "min"), ("p_max_abs", 1e-2, "max"),
                     ("score_max_abs", 1e-2, "max"), ("val_rel_l2", 1e-2, "max"))
PARITY_LIMITS_FP16 = (("rel_l2", 0.05, "max"), ("agree", 0.97, "min"), ("top50_box_recall", 0.8, "min"),
                      ("row_recall", 0.8, "min"), ("id_agree", 0.9, "min"), ("p_max_abs", 0.1, "max"),
                      ("score_max_abs", 0.05, "max"), ("val_rel_l2", 0.05, "max"))


def parity_failures(metrics: Dict[str, Dict[str, float]], limits=PARITY_LIMITS_FP16) -> List[str]:
    """Human-readable violations of ``limits`` (empty = parity ok)."""
    bad = []
    for name, m in metrics.items():
        if m.get("shape_mismatch") or m.get("missing"):
            bad.append(f"{name}: missing or wrong shape")
            continue
        for key, lim, kind in limits:
            if key in m and ((kind == "max" and not m[key] <= lim) or (kind == "min" and not m[key] >= lim)):
                bad.append(f"{name}: {key} {m[key]:.3g} ({'>' if kind == 'max' else '<'} {lim})")
    return bad


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
                 limits=None) -> ExportInfo:
    """Export ``module(*inputs.values())`` with static shapes; verify against PyTorch with onnxruntime
    (``limits`` default :data:`PARITY_LIMITS_ORT`)."""
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
        for name, w in zip(output_names, want):
            info.parity[name] = compare(name, got[name], w.numpy(), policy_for(name))
        bad = parity_failures(info.parity, limits or PARITY_LIMITS_ORT)
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
