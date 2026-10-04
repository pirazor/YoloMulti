"""Per-component parameters / GFLOPs / activation size, measurable on CPU.

Components: ``backbone`` (layers 0..10), ``neck`` (11..22), ``det`` (Detect), ``da`` and ``ll`` heads.
GFLOPs = 2 x MACs counted by ``torch.utils.flop_counter`` (convs / matmuls; elementwise ignored).
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Dict, Sequence

import torch
from torch.utils.flop_counter import FlopCounterMode


def _gflops(fn) -> float:
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 1e9


@torch.no_grad()
def profile_model(
    model, imgsz: Sequence[int] = (384, 640), latency_runs: int = 0, fused: bool = True
) -> Dict[str, Dict[str, float]]:
    """Return ``{component: {params_M, gflops}}`` plus ``total`` (and CPU latency if requested).

    ``fused=True`` profiles a fused copy, i.e. what is deployed: ``fuse()`` folds Conv+BN and drops the
    one-to-many detection branch (~7% of the FLOPs of the training graph)."""
    model = deepcopy(model).eval().float()
    if fused:
        model.fuse(verbose=False)
    x = torch.zeros(1, 3, *imgsz)
    backbone_end = int(model.yaml["backbone"].__len__())  # layers [0, backbone_end) are the backbone
    layers = list(model.model)
    parts = {
        "backbone": layers[:backbone_end],
        "neck": layers[backbone_end:-1],
        "det": layers[-1:],
        "da": [model.da_head],
        "ll": [model.ll_head],
    }
    # per-layer FLOPs: run once with module-level hooks recording FLOPs between pre/post
    flops = {k: 0.0 for k in parts}
    owner = {}
    for name, mods in parts.items():
        for m in mods:
            owner[id(m)] = name

    def pre(m, _):
        m._fc = FlopCounterMode(display=False)
        m._fc.__enter__()

    def post(m, _, __):
        m._fc.__exit__(None, None, None)
        flops[owner[id(m)]] += m._fc.get_total_flops() / 1e9
        del m._fc

    handles = []
    for mods in parts.values():
        for m in mods:
            handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    model(x)
    for h in handles:
        h.remove()

    out: Dict[str, Dict[str, float]] = {}
    for name, mods in parts.items():
        out[name] = {
            "params_M": sum(p.numel() for m in mods for p in m.parameters()) / 1e6,
            "gflops": flops[name],
        }
    out["total"] = {k: sum(v[k] for v in out.values()) for k in ("params_M", "gflops")}
    if latency_runs:
        model(x)
        t0 = time.perf_counter()
        for _ in range(latency_runs):
            model(x)
        out["total"]["cpu_ms"] = (time.perf_counter() - t0) / latency_runs * 1e3
    return out


def format_profile(p: Dict[str, Dict[str, float]]) -> str:
    rows = [f"{'part':<9}{'params(M)':>10}{'GFLOPs':>9}"]
    rows += [f"{k:<9}{v['params_M']:>10.3f}{v['gflops']:>9.2f}" for k, v in p.items()]
    return "\n".join(rows)


if __name__ == "__main__":  # python -m adas_mt.utils.profile --scale s --imgsz 384 640
    import argparse

    from adas_mt.nn import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="s", choices=list("nsmlx"))
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    ap.add_argument("--nc", type=int, default=9)
    ap.add_argument("--latency_runs", type=int, default=0)
    a = ap.parse_args()
    print(format_profile(profile_model(build_model(a.scale, nc=a.nc), a.imgsz, a.latency_runs)))
