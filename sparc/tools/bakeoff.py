"""Phase-0 bake-off: profile the candidates on the CPU, export them for the Jetson, summarise device results.

``export_bundle`` writes a self-contained folder: one ONNX graph per (backbone, stage) and per op probe, its inputs as
raw float32 ``.bin`` files (what ``trtexec --loadInputs`` reads) plus ``.npy``, the onnxruntime fp32 reference outputs,
``manifest.json`` and ``run_on_device.py``. On the Jetson only ``python3``, ``numpy`` and ``trtexec`` are needed:
``python3 run_on_device.py`` builds an FP16 engine per graph, times it with CUDA graphs, compares the engine's
outputs with the reference and writes ``results.json`` / ``results.md``. ``python -m sparc bakeoff report`` turns that
into the per-stage latency table (stage cost = difference of the cumulative cuts).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from sparc.deploy.export import PARITY_LIMITS_FP16, export_graph, parity_failures, policy_for
from sparc.nn.model import STAGES, SparcCfg, SparcDeploy, SparcNet, example_inputs

LOG = logging.getLogger("sparc.bakeoff")
COMPONENTS = ("backbone", "encoder", "decoder", "da_head", "stereo", "bev")


# --------------------------------------------------------------------------- activation scale
@torch.no_grad()
def calibrate_bn(net: nn.Module, inputs: Dict[str, torch.Tensor], steps: int = 4, seed: int = 0) -> nn.Module:
    """Set every BatchNorm's running statistics from forward passes on perturbed example inputs.

    An untrained network in eval mode uses BN statistics (0, 1) that do not match its activations: a random YOLO26
    trunk shrinks P5 to ~1e-8, so the exported graph would be checked (and timed in FP16) on degenerate values.
    With calibrated statistics every layer sees unit-scale activations, as in a trained model."""
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(steps):
        x = {k: (v * (1 + 0.1 * torch.randn(v.shape, generator=g)) if k in ("image", "image_right") else v)
             for k, v in inputs.items()}
        if "image" in x:
            x["image"] = x["image"].clamp(0, 1)
        net(**x)
    for m in bns:
        m.momentum = 0.1
    return net.eval()


# --------------------------------------------------------------------------- CPU profile
@torch.no_grad()
def profile_net(net: SparcNet, latency_runs: int = 0) -> Dict[str, Dict[str, float]]:
    """Params (M) and GFLOPs (2 x MACs of conv / matmul / attention) per deployed component."""
    from torch.utils.flop_counter import FlopCounterMode

    dep = SparcDeploy(net)
    inp = example_inputs(net)
    with FlopCounterMode(display=False, depth=3) as fc:
        dep(*inp.values())
    counts = fc.get_flop_counts()
    out = {}
    for comp in COMPONENTS:
        mod = getattr(net, comp, None)
        if mod is None:
            continue
        key = f"SparcDeploy.net.{comp}"
        out[comp] = {"params_M": sum(p.numel() for p in mod.parameters()) / 1e6,
                     "gflops": sum(counts.get(key, {}).values()) / 1e9}
    out["total"] = {"params_M": sum(p.numel() for n, p in net.named_parameters() if not n.startswith("lane_aux")) / 1e6,
                    "gflops": sum(counts["Global"].values()) / 1e9}
    if latency_runs:
        dep(*inp.values())
        t0 = time.perf_counter()
        for _ in range(latency_runs):
            dep(*inp.values())
        out["total"]["torch_cpu_ms"] = (time.perf_counter() - t0) / latency_runs * 1e3
    return out


def profile_table(backbones: Sequence[str], imgsz=(384, 640), stereo: bool = False, latency_runs: int = 0) -> str:
    comps = [c for c in COMPONENTS if c != "stereo" or stereo]
    head = "| backbone | " + " | ".join(f"{c} M / GF" for c in comps) + " | total M / GF |" + (" torch CPU ms |" if latency_runs else "")
    lines = [head, "|" + "---|" * (len(comps) + 2 + bool(latency_runs))]
    for b in backbones:
        net = SparcNet(SparcCfg(backbone=b, imgsz=tuple(imgsz), stereo=stereo)).eval()
        p = profile_net(net, latency_runs)
        cells = [f"{p[c]['params_M']:.1f} / {p[c]['gflops']:.1f}" for c in comps]
        row = f"| {b} | " + " | ".join(cells) + f" | {p['total']['params_M']:.1f} / {p['total']['gflops']:.1f} |"
        if latency_runs:
            row += f" {p['total']['torch_cpu_ms']:.0f} |"
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- probes
class _Probe(nn.Module):
    """Wraps a callable so it exports with named positional inputs."""

    def __init__(self, fn, modules: Dict[str, nn.Module]):
        super().__init__()
        self.mods = nn.ModuleDict(modules)
        self.fn = fn

    def forward(self, *args):
        out = self.fn(self.mods, *args)
        return out if isinstance(out, tuple) else (out,)


def build_probes(imgsz=(384, 640), seed: int = 0) -> List[dict]:
    """Single-feature graphs at the deployed shapes: each isolates one op family TensorRT must handle."""
    from sparc.nn.bev import DepthSplat, IPMSampler
    from sparc.nn.deform_attn import MSDeformAttn
    from sparc.nn.stereo import StereoHead

    torch.manual_seed(seed)
    h, w = imgsz
    shapes = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
    s = sum(a * b for a, b in shapes)
    q, d = 224, 256
    base = SparcNet(SparcCfg(backbone="yolo26s", imgsz=imgsz, stereo=False))  # only for example_inputs geometry
    geo = example_inputs(base)
    probes = []
    for mode in ("grid_sample", "gather"):
        attn = MSDeformAttn(d, 8, 3, 4, mode)
        with torch.no_grad():  # spread the sampling points like a trained model would
            attn.offsets.weight.normal_(0, 0.02)
            attn.weights.weight.normal_(0, 0.02)
        probes.append({
            "name": f"probe_msda_{mode}", "what": f"deformable cross-attention, {mode} sampling, {q} queries x {s} tokens",
            "module": _Probe(lambda m, x, ref, mem: m["a"](x, ref, mem, shapes), {"a": attn}),
            "inputs": {"query": torch.randn(1, q, d), "ref": torch.rand(1, q, 4) * 0.8 + 0.1, "memory": torch.randn(1, s, d)},
            "outputs": ["out"]})
    splat = DepthSplat(256, 64)
    probes.append({
        "name": "probe_splat_scatter_add", "what": "depth splat: ScatterElements(add) of 15360 points into 240x160 cells",
        "module": _Probe(lambda m, f, dep, K, T: m["s"](f, dep, K, T), {"s": splat}),
        "inputs": {"feat": torch.randn(1, 256, h // 8, w // 8), "depth": geo["depth"], "K": geo["K"], "cam2ego": geo["cam2ego"]},
        "outputs": ["bev_feat"]})
    ipm = IPMSampler(256, 64)
    probes.append({
        "name": "probe_ipm_grid_sample", "what": "IPM: ground-plane projection + GridSample for 38400 BEV cells",
        "module": _Probe(lambda m, f, K, T: m["i"](f, K, T), {"i": ipm}),
        "inputs": {"feat": torch.randn(1, 256, h // 8, w // 8), "K": geo["K"], "cam2ego": geo["cam2ego"]},
        "outputs": ["bev_feat"]})
    stereo = StereoHead(192)
    img = geo["image"]
    probes.append({
        "name": "probe_stereo_head", "what": "in-network stereo: shared stem, 48-level group-wise correlation, convex upsampling",
        "module": _Probe(lambda m, l, r, K, b: (lambda disp: (disp, StereoHead.to_depth(disp, K, b)))(m["s"](l, r)),
                         {"s": stereo}),
        "inputs": {"left": img, "right": torch.roll(img, -6, -1), "K": geo["K"], "baseline": torch.full((1, 1), 0.1)},
        "outputs": ["disparity", "depth"]})
    return probes


# --------------------------------------------------------------------------- bundle
def _save_inputs(d: Path, name: str, inputs: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    out = {}
    for k, v in inputs.items():
        a = v.detach().contiguous().numpy().astype(np.float32)
        a.tofile(d / f"{name}.in.{k}.bin")
        np.save(d / f"{name}.in.{k}.npy", a)
        out[k] = {"bin": f"io/{name}.in.{k}.bin", "npy": f"io/{name}.in.{k}.npy", "shape": list(a.shape), "dtype": "float32"}
    return out


def _export_one(root: Path, name: str, module: nn.Module, inputs: Dict[str, torch.Tensor], outputs: Sequence[str],
                opset: int, cpu_runs: int, extra: dict) -> dict:
    from sparc.deploy.export import run_ort

    t0 = time.time()
    info = export_graph(module, inputs, root / "onnx" / f"{name}.onnx", outputs, opset=opset, cpu_runs=cpu_runs)
    ins = _save_inputs(root / "io", name, inputs)
    ref, _ = run_ort(info.path, {k: v.numpy() for k, v in inputs.items()})
    outs = {}
    for k in outputs:
        np.save(root / "io" / f"{name}.ref.{k}.npy", ref[k])
        outs[k] = {"ref": f"io/{name}.ref.{k}.npy", "shape": list(ref[k].shape), "dtype": str(ref[k].dtype),
                   "policy": policy_for(k)}
    LOG.info("%s: %.0fs, %.1f MB, device-check ops %s", name, time.time() - t0, info.mbytes, info.audit["check_on_device"])
    return {"name": name, "onnx": f"onnx/{name}.onnx", "inputs": ins, "outputs": outs, "ops": info.ops,
            "audit": info.audit, "scatter_reductions": info.scatter_reductions, "ort_parity": info.parity,
            "ort_cpu_ms": info.ort_cpu_ms, "mbytes": info.mbytes, **extra}


def export_bundle(out: str | Path, backbones: Sequence[str], imgsz=(384, 640), stages: Sequence[str] = ("backbone", "full"),
                  probes: bool = True, opset: int = 17, cpu_runs: int = 3) -> dict:
    root = Path(out)
    for sub in ("onnx", "io"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    bad = set(stages) - set(STAGES)
    if bad:
        raise ValueError(f"unknown stages {sorted(bad)}; choose from {STAGES}")
    models = []
    for b in backbones:
        net = SparcNet(SparcCfg(backbone=b, imgsz=tuple(imgsz))).eval()
        calibrate_bn(net, example_inputs(net))
        prof = profile_net(net)
        for st in [s for s in STAGES if s in stages]:
            dep = SparcDeploy(net, st)
            models.append(_export_one(root, f"{b}.{st}", dep, example_inputs(net, st), dep.output_names(), opset,
                                      cpu_runs, {"kind": "candidate", "backbone": b, "stage": st, "profile": prof}))
    if probes:
        for p in build_probes(tuple(imgsz)):
            models.append(_export_one(root, p["name"], p["module"], p["inputs"], p["outputs"], opset, cpu_runs,
                                      {"kind": "probe", "what": p["what"]}))
    manifest = {"format": "sparc-bakeoff", "version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "imgsz": list(imgsz), "opset": opset, "torch": torch.__version__, "models": models}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.copy(Path(__file__).with_name("run_on_device.py"), root / "run_on_device.py")
    return manifest


# --------------------------------------------------------------------------- report
def parity_ok(metrics: Dict[str, Dict[str, float]]) -> bool:
    return not parity_failures(metrics, PARITY_LIMITS_FP16)


def report(out: str | Path) -> str:
    root = Path(out)
    manifest = json.loads((root / "manifest.json").read_text())
    res_files = sorted(root.glob("results_*.json"))
    if not res_files:
        return "no results_*.json in the bundle yet: run run_on_device.py on the Jetson first"
    lines = []
    for rf in res_files:
        res = json.loads(rf.read_text())
        by = {r["name"]: r for r in res["results"]}
        dev = res.get("device", {})
        lines += [f"## {res['precision'].upper()} on {dev.get('model', '?')} (TensorRT {res.get('tensorrt') or '?'})", "",
                  "| backbone | backbone ms | rest ms | full ms | GFLOPs | params M | fp16 parity |", "|---|---|---|---|---|---|---|"]
        cands = [m for m in manifest["models"] if m["kind"] == "candidate"]
        for b in dict.fromkeys(m["backbone"] for m in cands):
            rb, rf_ = by.get(f"{b}.backbone", {}), by.get(f"{b}.full", {})
            tb, tf = rb.get("gpu_median_ms"), rf_.get("gpu_median_ms")
            prof = next(m["profile"] for m in cands if m["backbone"] == b)["total"]
            fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else (v or "-")
            rest = tf - tb if (tb is not None and tf is not None) else None
            par = "ok" if rf_.get("parity") and parity_ok(rf_["parity"]) else ("FAIL" if rf_.get("parity") else rf_.get("error", "-")[:40])
            lines.append(f"| {b} | {fmt(tb)} | {fmt(rest)} | {fmt(tf)} | {prof['gflops']:.1f} | {prof['params_M']:.1f} | {par} |")
        lines += ["", "| probe | builds | ms | parity | what |", "|---|---|---|---|---|"]
        for m in manifest["models"]:
            if m["kind"] != "probe":
                continue
            r = by.get(m["name"], {})
            built = "yes" if r.get("built") else ("NO" if r else "-")
            ms = f"{r['gpu_median_ms']:.2f}" if r.get("gpu_median_ms") is not None else "-"
            par = "ok" if r.get("parity") and parity_ok(r["parity"]) else ("FAIL" if r.get("parity") else "-")
            lines.append(f"| {m['name']} | {built} | {ms} | {par} | {m['what']} |")
        lines.append("")
    return "\n".join(lines)
