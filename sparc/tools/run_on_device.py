#!/usr/bin/env python3
"""Run a SPARC bake-off bundle on the Jetson: build TensorRT engines, time them, check their outputs.

Needs only python3, numpy and trtexec (JetPack: /usr/src/tensorrt/bin/trtexec). Run from anywhere:

    sudo nvpmodel -m 0 && sudo jetson_clocks          # fixed max clocks, or the timings mean little
    python3 run_on_device.py                          # FP16, every graph in manifest.json
    python3 run_on_device.py --precision fp32 --only probe_
    python3 run_on_device.py --precision best         # INT8 allowed (uncalibrated: timing only, accuracy is not checked)

Writes ``results_<precision>.json`` and ``results_<precision>.md`` next to this script; send those back (the engines in
``engines/`` are device-specific and not needed). The comparison logic mirrors ``sparc.deploy.export.compare``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLAGS = {"fp16": ["--fp16"], "fp32": [], "best": ["--best"]}
LIMITS = {"rel_l2": 0.05, "agree": 0.97, "top50_box_recall": 0.8, "row_recall": 0.8}
TIME_RE = re.compile(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms"
                     r"(?:, percentile\(90%\) = ([\d.]+) ms)?(?:, percentile\(95%\) = ([\d.]+) ms)?(?:, percentile\(99%\) = ([\d.]+) ms)?")
LAT_RE = re.compile(r"Latency: min = [\d.]+ ms, max = [\d.]+ ms, mean = [\d.]+ ms, median = ([\d.]+) ms")
BUILD_RE = re.compile(r"Engine built in ([\d.]+) sec")
TRT_RE = re.compile(r"TensorRT version: ([\d.]+)")


def find_trtexec(given: str | None) -> str:
    for c in [given, shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"]:
        if c and Path(c).exists():
            return c
    sys.exit("trtexec not found: pass --trtexec /path/to/trtexec")


def device_info() -> dict:
    info = {"machine": platform.machine(), "python": platform.python_version()}
    for key, path in (("model", "/proc/device-tree/model"), ("l4t", "/etc/nv_tegra_release")):
        try:
            info[key] = Path(path).read_text(errors="ignore").strip("\x00\n ").splitlines()[0]
        except OSError:
            pass
    try:
        info["nvpmodel"] = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


# ------------------------------------------------------------------ comparison (mirrors sparc.deploy.export.compare)
def _best_iou(b, boxes):
    x1, y1 = np.maximum(b[0], boxes[:, 0]), np.maximum(b[1], boxes[:, 1])
    x2, y2 = np.minimum(b[2], boxes[:, 2]), np.minimum(b[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1])
    ab = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return float((inter / np.maximum(a + ab - inter, 1e-9)).max()) if len(boxes) else 0.0


def compare(got, ref, policy):
    got, ref = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if got.size != ref.size:
        return {"shape_mismatch": 1.0}
    got = got.reshape(ref.shape)
    if policy == "label":
        return {"agree": float((got == ref).mean())}
    if policy == "rowset":
        g, r = got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1])
        d = np.sqrt(((r[:, None, :] - g[None, :, :]) ** 2).sum(-1)).min(1) / (np.linalg.norm(r, axis=1) + 1e-12)
        return {"row_recall": float((d < 0.05).mean()), "row_rel_max": float(d.max())}
    if policy == "ranked":
        gs, rs = np.sort(got[..., 4].ravel())[::-1], np.sort(ref[..., 4].ravel())[::-1]
        g, r = got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1])
        top = r[np.argsort(-r[:, 4])[:50], :4]
        return {"score_max_abs": float(np.abs(gs - rs).max()),
                "top50_box_recall": float(np.mean([_best_iou(b, g[:, :4]) > 0.9 for b in top]))}
    return {"rel_l2": float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)), "max_abs": float(np.abs(got - ref).max())}


def parity_ok(metrics):
    for m in metrics.values():
        if m.get("shape_mismatch") or m.get("missing"):
            return False
        if "rel_l2" in m and not m["rel_l2"] <= LIMITS["rel_l2"]:
            return False
        if "agree" in m and m["agree"] < LIMITS["agree"]:
            return False
        if "top50_box_recall" in m and m["top50_box_recall"] < LIMITS["top50_box_recall"]:
            return False
        if "row_recall" in m and m["row_recall"] < LIMITS["row_recall"]:
            return False
    return True


def read_export(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {d["name"]: np.asarray(d["values"]) for d in data}


# ------------------------------------------------------------------ one graph
def run_one(m: dict, a, trtexec: str, out_dir: Path) -> dict:
    name = m["name"]
    eng = out_dir / "engines" / f"{name}.{a.precision}.engine"
    eng.parent.mkdir(parents=True, exist_ok=True)
    exp = out_dir / "outputs" / f"{name}.{a.precision}.json"
    exp.parent.mkdir(parents=True, exist_ok=True)
    loads = ",".join(f"{k}:{HERE / v['bin']}" for k, v in m["inputs"].items())
    src = [f"--loadEngine={eng}"] if (a.reuse and eng.exists()) else [f"--onnx={HERE / m['onnx']}", f"--saveEngine={eng}"]
    cmd = [trtexec, *src, *FLAGS[a.precision], f"--loadInputs={loads}", f"--exportOutput={exp}", "--useCudaGraph",
           "--noDataTransfers", "--useSpinWait", f"--duration={a.duration}", f"--warmUp={a.warmup}",
           f"--memPoolSize=workspace:{a.workspace}M"]
    if a.layer_profile:
        cmd += ["--separateProfileRun", "--dumpProfile", f"--exportProfile={out_dir / 'outputs' / f'{name}.{a.precision}.profile.json'}"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        log, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        log, rc = f"timeout after {a.timeout}s\n{e.stdout or ''}", -1
    (out_dir / "outputs" / f"{name}.{a.precision}.log").write_text(log)
    r = {"name": name, "kind": m.get("kind"), "built": rc == 0,
         "returncode": rc, "wall_s": round(time.time() - t0, 1)}
    if TRT_RE.search(log):
        r["tensorrt"] = TRT_RE.search(log).group(1)
    if BUILD_RE.search(log):
        r["build_s"] = float(BUILD_RE.search(log).group(1))
    t = TIME_RE.search(log)
    if t:
        vals = [float(v) if v else None for v in t.groups()]
        r.update({"gpu_min_ms": vals[0], "gpu_max_ms": vals[1], "gpu_mean_ms": vals[2], "gpu_median_ms": vals[3],
                  "gpu_p90_ms": vals[4], "gpu_p99_ms": vals[6]})
    lat = LAT_RE.search(log)
    if lat:
        r["latency_median_ms"] = float(lat.group(1))
    if rc != 0:
        errs = [l for l in log.splitlines() if "ERROR" in l or "error" in l]
        r["error"] = (errs[-1] if errs else log.strip().splitlines()[-1] if log.strip() else "failed")[-300:]
        return r
    if a.precision == "best":
        return r  # uncalibrated INT8: numbers are meaningless, timing only
    if exp.exists():
        got = read_export(exp)
        r["parity"] = {k: (compare(got[k], np.load(HERE / v["ref"]), v["policy"]) if k in got else {"missing": 1.0})
                       for k, v in m["outputs"].items()}
        r["parity_ok"] = parity_ok(r["parity"])
    return r


def to_markdown(res: dict) -> str:
    dev = res["device"]
    lines = [f"# SPARC bake-off: {res['precision']} on {dev.get('model', '?')}", "",
             f"TensorRT {res.get('tensorrt', '?')}, {dev.get('l4t', '')}, nvpmodel: {dev.get('nvpmodel', '?').splitlines()[0] if dev.get('nvpmodel') else '?'}",
             "", "| graph | built | GPU median ms | p99 ms | parity | note |", "|---|---|---|---|---|---|"]
    for r in res["results"]:
        ms = f"{r['gpu_median_ms']:.2f}" if r.get("gpu_median_ms") is not None else "-"
        p99 = f"{r['gpu_p99_ms']:.2f}" if r.get("gpu_p99_ms") is not None else "-"
        par = "-" if "parity" not in r else ("ok" if r["parity_ok"] else "FAIL")
        note = r.get("error", "")
        if r.get("parity") and not r["parity_ok"]:
            note = "; ".join(f"{k}: {v}" for k, v in r["parity"].items())[:200]
        lines.append(f"| {r['name']} | {'yes' if r['built'] else 'NO'} | {ms} | {p99} | {par} | {note} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trtexec", default=None)
    ap.add_argument("--precision", choices=sorted(FLAGS), default="fp16")
    ap.add_argument("--only", nargs="*", default=None, help="run graphs whose name contains any of these substrings")
    ap.add_argument("--duration", type=int, default=10, help="seconds of timed inference per graph")
    ap.add_argument("--warmup", type=int, default=1000, help="warm-up in ms")
    ap.add_argument("--workspace", type=int, default=2048, help="builder workspace MiB")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per graph (engine builds can be slow)")
    ap.add_argument("--reuse", action="store_true", help="reuse engines built earlier instead of rebuilding")
    ap.add_argument("--layer-profile", action="store_true", help="also export a per-layer profile")
    ap.add_argument("--out", default=str(HERE), help="where results / engines go (default: the bundle)")
    a = ap.parse_args(argv)
    manifest = json.loads((HERE / "manifest.json").read_text())
    trtexec = find_trtexec(a.trtexec)
    out_dir = Path(a.out)
    models = [m for m in manifest["models"] if not a.only or any(s in m["name"] for s in a.only)]
    res = {"precision": a.precision, "device": device_info(), "trtexec": trtexec, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "results": []}
    for i, m in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {m['name']} ...", flush=True)
        r = run_one(m, a, trtexec, out_dir)
        res.setdefault("tensorrt", r.get("tensorrt"))
        res["results"].append(r)
        msg = f"   built={r['built']}"
        if r.get("gpu_median_ms") is not None:
            msg += f" median={r['gpu_median_ms']:.2f} ms"
        if "parity_ok" in r:
            msg += f" parity={'ok' if r['parity_ok'] else 'FAIL'}"
        if r.get("error"):
            msg += f" error={r['error'][:120]}"
        print(msg, flush=True)
        (out_dir / f"results_{a.precision}.json").write_text(json.dumps(res, indent=1))  # incremental
    (out_dir / f"results_{a.precision}.md").write_text(to_markdown(res))
    print(f"wrote {out_dir / f'results_{a.precision}.json'} and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
