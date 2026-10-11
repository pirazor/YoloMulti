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

    from sparc.nn.backbone import BACKBONES

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
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
