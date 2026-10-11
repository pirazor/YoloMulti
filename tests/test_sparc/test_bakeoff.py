"""Bake-off bundle: export on the CPU, then the device runner against a fake trtexec (output format of TensorRT 10)."""

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from sparc.tools.bakeoff import export_bundle, parity_ok, report

FAKE_TRTEXEC = textwrap.dedent(r'''
    #!{python}
    """Fake trtexec: echoes the bundle's reference outputs (+ fp16-sized noise) in --exportOutput format."""
    import json, sys
    from pathlib import Path
    import numpy as np
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    onnx = args.get("onnx", "")
    if "splat" in onnx:
        print("[E] [TRT] ModelImporter.cpp:949: ERROR: ScatterElements: reduction add is not supported")
        sys.exit(1)
    root = Path(onnx).parent.parent
    name = Path(onnx).stem
    m = next(x for x in json.loads((root / "manifest.json").read_text())["models"] if x["name"] == name)
    rng = np.random.default_rng(0)
    out = []
    for k, v in m["outputs"].items():
        ref = np.load(root / v["ref"])
        if name == "probe_msda_gather":
            ref = ref.copy(); ref.flat[0] = float("nan")
        if v["policy"] == "dense" and "noisy" in sys.argv[0]:
            ref = ref + rng.normal(0, 0.2, ref.shape) * (np.abs(ref).mean() + 1e-6)
        elif v["policy"] == "dense":
            ref = ref * (1 + rng.normal(0, 1e-3, ref.shape))
        out.append({"name": k, "dimensions": "x".join(map(str, ref.shape)), "values": ref.ravel().tolist()})
    Path(args["exportOutput"]).write_text(json.dumps(out).replace("NaN", "-nan"))  # trtexec prints -nan / inf
    print("[I] TensorRT version: 10.3.0")
    print("[I] Engine built in 12.5 sec.")
    print("[I] Latency: min = 2.0 ms, max = 3.0 ms, mean = 2.5 ms, median = 2.4 ms, percentile(90%) = 2.8 ms, percentile(95%) = 2.9 ms, percentile(99%) = 3.0 ms")
    print("[I] GPU Compute Time: min = 1.5 ms, max = 2.5 ms, mean = 1.9 ms, median = 1.8 ms, percentile(90%) = 2.2 ms, percentile(95%) = 2.3 ms, percentile(99%) = 2.4 ms")
    print("&&&& PASSED TensorRT.trtexec")
''').lstrip()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle")
    export_bundle(out, backbones=["vit_t16_dinov3"], imgsz=(128, 192), stages=["backbone", "full"], cpu_runs=0)
    return out


def _fake(tmp_path, name="trtexec"):
    p = tmp_path / name
    p.write_text(FAKE_TRTEXEC.replace("{python}", sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_bundle_contents(bundle):
    m = json.loads((bundle / "manifest.json").read_text())
    names = [x["name"] for x in m["models"]]
    assert names[:2] == ["vit_t16_dinov3.backbone", "vit_t16_dinov3.full"]
    assert {"probe_msda_grid_sample", "probe_msda_gather", "probe_splat_scatter_add", "probe_ipm_grid_sample",
            "probe_stereo_head"} <= set(names)
    full = m["models"][1]
    assert set(full["inputs"]) == {"image", "K", "cam2ego", "depth"}
    assert full["audit"]["unsupported"] == [] and full["scatter_reductions"] == ["add"]
    assert "GridSample" in full["audit"]["check_on_device"]
    gather = next(x for x in m["models"] if x["name"] == "probe_msda_gather")
    assert "GridSample" not in gather["ops"]  # the fallback really avoids GridSample
    for x in m["models"]:  # raw .bin inputs are exactly what trtexec --loadInputs reads
        for k, v in x["inputs"].items():
            raw = np.fromfile(bundle / v["bin"], dtype=np.float32)
            assert raw.size == np.prod(v["shape"]) and np.array_equal(raw, np.load(bundle / v["npy"]).ravel())
    assert (bundle / "run_on_device.py").exists()


def test_device_runner_with_fake_trtexec(bundle, tmp_path):
    fake = _fake(tmp_path)
    r = subprocess.run([sys.executable, str(bundle / "run_on_device.py"), "--trtexec", str(fake), "--duration", "1",
                        "--out", str(tmp_path / "res")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "res" / "results_fp16.json").read_text())
    by = {x["name"]: x for x in res["results"]}
    assert res["tensorrt"] == "10.3.0"
    ok = by["vit_t16_dinov3.full"]
    assert ok["built"] and ok["gpu_median_ms"] == 1.8 and ok["gpu_p99_ms"] == 2.4 and ok["build_s"] == 12.5
    assert ok["parity_ok"] and ok["parity"]["da"]["agree"] == 1.0 and ok["parity"]["agents"]["top50_box_recall"] == 1.0
    bad = by["probe_splat_scatter_add"]
    assert not bad["built"] and "ScatterElements" in bad["error"]
    nan = by["probe_msda_gather"]  # a NaN in the export is parsed, fails parity, and does not stop the run
    assert nan["built"] and nan["parity_ok"] is False and len(res["results"]) == len(json.loads((bundle / "manifest.json").read_text())["models"])
    md = (tmp_path / "res" / "results_fp16.md").read_text()
    assert "| probe_splat_scatter_add | NO |" in md
    # the CLI report reads results next to the manifest
    (bundle / "results_fp16.json").write_text(json.dumps(res))
    rep = report(bundle)
    assert "| vit_t16_dinov3 | 1.80 | 0.00 | 1.80 |" in rep and "| probe_splat_scatter_add | NO |" in rep
    assert "TensorRT 10.3.0" in rep


def test_device_runner_flags_numerical_drift(bundle, tmp_path):
    fake = _fake(tmp_path, "trtexec_noisy")
    r = subprocess.run([sys.executable, str(bundle / "run_on_device.py"), "--trtexec", str(fake), "--only", "backbone",
                        "--duration", "1", "--out", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "results_fp16.json").read_text())
    assert len(res["results"]) == 1 and res["results"][0]["parity_ok"] is False


def test_runner_and_library_compare_agree():
    """run_on_device.py duplicates the comparison (no sparc import on the device): keep them identical."""
    sys.path.insert(0, str(Path(__file__).parents[2] / "sparc" / "tools"))
    import run_on_device as dev
    from sparc.deploy import export as export_mod
    from sparc.deploy.export import compare

    rng = np.random.default_rng(0)
    ref = rng.random((1, 200, 18)).astype(np.float32)
    ref[..., 2:4] += ref[..., 0:2] + 0.5
    got = ref + rng.normal(0, 1e-3, ref.shape)
    emb = rng.normal(size=(1, 200, 64))
    perm = emb[:, rng.permutation(200)]
    assert dev.compare(perm, emb, "rowset") == compare("agent_emb", perm, emb, "rowset")
    assert compare("agent_emb", perm, emb, "rowset")["row_recall"] == 1.0
    for policy in ("dense", "agents"):
        a, b = dev.compare(got, ref, policy), compare("x", got, ref, policy)
        assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-12 for k in a)
    lab = rng.integers(0, 3, (1, 32, 48))
    assert dev.compare(lab, lab, "label") == compare("da", lab, lab, "label") == {"agree": 1.0}
    from sparc.deploy.export import PARITY_LIMITS_FP16

    assert tuple(dev.LIMITS) == tuple(PARITY_LIMITS_FP16) and dev.AGENT_ROLES == export_mod.AGENT_ROLES
    assert parity_ok({"x": {"rel_l2": 0.01}}) and not parity_ok({"x": {"rel_l2": 0.5}})
    # agents: a reordered, fp16-noisy table with one flipped sign id is still matched row by row
    ag = rng.random((1, 200, 18)); ag[..., 2:4] += ag[..., 0:2] + 0.5
    ag[..., 5] = rng.integers(0, 10, (1, 200)); ag[..., 14] = rng.integers(0, 4, (1, 200)); ag[..., 16] = rng.integers(0, 401, (1, 200))
    p2 = ag[:, rng.permutation(200)].copy()
    p2[0, 0, 16] += 1  # one wrong sign id among the top-50 rows (or not; depends on where row 0 ranks)
    m = compare("agents", p2, ag, "agents")
    assert m == dev.compare(p2, ag, "agents") and m["top50_box_recall"] == 1.0 and m["val_rel_l2"] == 0.0 and m["id_agree"] >= 0.99
    lm = rng.random((1, 24, 9)); lm[..., [1, 3, 5, 7]] = rng.integers(0, 3, (1, 24, 4))
    lm2 = lm.copy(); lm2[0, 0, 7] += 1
    m = compare("lane_meta", lm2, lm, "lane_meta")
    assert m == dev.compare(lm2, lm, "lane_meta") and m["id_agree"] == pytest.approx(1 - 1 / 96) and m["p_max_abs"] == 0.0
    # trtexec writes nan / -nan / inf in its JSON export
    f = Path(__import__("tempfile").mkdtemp()) / "o.json"
    f.write_text('[{"name": "x", "dimensions": "3", "values": [1.5, -nan, inf]}, {"name": "y", "dimensions": "1", "values": [-inf]}]')
    got = dev.read_export(f)
    assert np.isnan(got["x"][1]) and np.isposinf(got["x"][2]) and np.isneginf(got["y"][0]) and got["x"][0] == 1.5
