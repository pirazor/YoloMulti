"""``python -m sparc <rig|profile|bakeoff> ...``"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence


def cmd_rig(a: argparse.Namespace) -> int:
    from sparc.rig.stereo import StereoRig

    rig = StereoRig.from_yaml(a.calib)
    if a.alpha is not None:
        rig = rig.with_alpha(a.alpha)
    widths = {f"width {w}": w for w in a.disparity_width}
    print(rig.report(tuple(a.imgsz), widths))
    for w in rig.check(a.baseline):
        logging.getLogger("sparc").warning(w)
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    from sparc.data import converters

    mod = converters.get(a.source)
    sub = argparse.ArgumentParser(prog=f"python -m sparc convert {a.source}", description=mod.__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    mod.add_args(sub)
    stats = mod.convert(sub.parse_args(a.rest))
    print(stats.summary())
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from sparc.engine.config import DataCfg, TrainCfg
    from sparc.engine.trainer import Trainer

    overrides = {"epochs": a.epochs, "batch": a.batch, "workers": a.workers, "lr": a.lr, "epoch_size": a.epoch_size,
                 "amp": a.amp, "model.backbone": a.backbone, "model.pretrained": a.pretrained, "distill.enabled": a.distill}
    cfg = TrainCfg.load(a.cfg, overrides)
    data = DataCfg.load(a.data)
    if a.signs:
        data.signs = a.signs
    tr = Trainer(cfg, data, a.out, device=a.device, resume=a.resume)
    last = tr.fit()
    if tr.main:
        print(json.dumps({k: v for k, v in last.items() if isinstance(v, (int, float))}, indent=1))
    return 0


def cmd_val(a: argparse.Namespace) -> int:
    from sparc.engine.config import DataCfg
    from sparc.engine.trainer import Trainer, load_for_eval

    model, vocab, cfg = load_for_eval(a.weights, a.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu"))
    data = DataCfg.load(a.data)
    data.signs = a.signs or str(Path(a.weights).resolve().parent.parent / "signs.json")
    if not Path(data.signs).exists():
        data.signs = None
    cfg.workers = a.workers if a.workers is not None else cfg.workers
    tr = Trainer(cfg, DataCfg(train=data.val, val=data.val, signs=data.signs), a.out, device=a.device)
    res = tr.validate(model)
    print(json.dumps(res, indent=1))
    (Path(a.out) / "val.json").write_text(json.dumps(res, indent=1))
    return 0


def cmd_synth(a: argparse.Namespace) -> int:
    from sparc.data.synthetic import main as synth_main

    return synth_main(a.rest)


def cmd_profile(a: argparse.Namespace) -> int:
    from sparc.tools.bakeoff import profile_table

    print(profile_table(a.backbones, tuple(a.imgsz), stereo=a.stereo, latency_runs=a.cpu_runs))
    return 0


def cmd_bakeoff(a: argparse.Namespace) -> int:
    from sparc.tools import bakeoff

    if a.action == "export":
        manifest = bakeoff.export_bundle(a.out, backbones=a.backbones, imgsz=tuple(a.imgsz), stages=a.stages,
                                         probes=not a.no_probes, opset=a.opset, cpu_runs=a.cpu_runs)
        print(json.dumps({k: v for k, v in manifest.items() if k != "models"}, indent=1))
        print(f"{len(manifest['models'])} graphs in {a.out}; copy the folder to the Jetson and run "
              f"`python3 {Path(a.out) / 'run_on_device.py'}`")
        return 0
    if a.action == "report":
        print(bakeoff.report(a.out))
        return 0
    raise ValueError(a.action)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sparc", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rig", help="stereo calibration report: FOV, network intrinsics, depth precision vs range")
    r.add_argument("--calib", required=True, help="stereo calibration YAML (OpenCV stereoCalibrate/stereoRectify keys)")
    r.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    r.add_argument("--disparity-width", type=int, nargs="*", default=[640, 320],
                   help="widths (px) at which disparity is estimated, for the precision table")
    r.add_argument("--alpha", type=float, default=None, help="re-run stereoRectify with this free-scaling alpha")
    r.add_argument("--baseline", type=float, default=None, help="expected baseline in metres (sanity check)")
    r.set_defaults(fn=cmd_rig)

    from sparc.data.converters import REGISTRY
    from sparc.nn.backbones_spec import BACKBONES

    c = sub.add_parser("convert", help="convert a public dataset to SPARC samples: convert <source> -- <source args>")
    c.add_argument("source", choices=sorted(REGISTRY))
    c.add_argument("rest", nargs=argparse.REMAINDER, help="arguments of the source converter (see convert <source> -- --help)")
    c.set_defaults(fn=cmd_convert)

    t = sub.add_parser("train", help="train SPARC on converted sources")
    t.add_argument("--cfg", required=True, help="training config YAML (sparc/cfg/stage_a.yaml)")
    t.add_argument("--data", required=True, help="data manifest YAML (sources, weights)")
    t.add_argument("--out", required=True, help="run directory")
    t.add_argument("--resume", default=None, help="checkpoint to resume (weights/last.pt)")
    t.add_argument("--signs", default=None, help="saved sign vocabulary (default: union of the manifests)")
    t.add_argument("--device", default=None)
    for k, typ in (("epochs", int), ("batch", int), ("workers", int), ("epoch-size", int), ("lr", float)):
        t.add_argument(f"--{k}", type=typ, default=None, dest=k.replace("-", "_"))
    t.add_argument("--amp", default=None, choices=["bf16", "fp16", "none"])
    t.add_argument("--backbone", default=None)
    t.add_argument("--pretrained", default=None, type=lambda v: v.lower() in ("1", "true", "yes"))
    t.add_argument("--distill", default=None, type=lambda v: v.lower() in ("1", "true", "yes"))
    t.set_defaults(fn=cmd_train)

    v = sub.add_parser("val", help="evaluate a checkpoint on the manifest's validation sources")
    v.add_argument("--weights", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--signs", default=None)
    v.add_argument("--device", default=None)
    v.add_argument("--workers", type=int, default=None)
    v.set_defaults(fn=cmd_val)

    sy = sub.add_parser("synth", help="write a synthetic dataset: synth -- --dst DIR [--n-train N --n-val M]")
    sy.add_argument("rest", nargs=argparse.REMAINDER)
    sy.set_defaults(fn=cmd_synth)

    pr = sub.add_parser("profile", help="params / GFLOPs per stage on CPU for each backbone candidate")
    pr.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    pr.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    pr.add_argument("--stereo", action="store_true", help="include the in-network stereo head")
    pr.add_argument("--cpu-runs", type=int, default=0, help="also time the PyTorch forward on this CPU")
    pr.set_defaults(fn=cmd_profile)

    b = sub.add_parser("bakeoff", help="export the backbone bake-off bundle / summarise device results")
    b.add_argument("action", choices=["export", "report"])
    b.add_argument("--out", default="runs/bakeoff", help="bundle directory")
    b.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    b.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    b.add_argument("--stages", nargs="+", default=["backbone", "full"],
                   help="cumulative cut points to export per candidate: backbone encoder decoder full")
    b.add_argument("--no-probes", action="store_true", help="skip the single-op TensorRT probes")
    b.add_argument("--opset", type=int, default=17)
    b.add_argument("--cpu-runs", type=int, default=3, help="onnxruntime CPU timing runs per graph (0 = skip)")
    b.set_defaults(fn=cmd_bakeoff)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    a = build_parser().parse_args(argv)
    if getattr(a, "rest", None) and a.rest[:1] == ["--"]:
        a.rest = a.rest[1:]
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
