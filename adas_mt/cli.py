"""``python -m adas_mt <train|val|convert|profile> ...``"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

DEFAULT_CFG = Path(__file__).parent / "cfg" / "default.yaml"


def _load_cfg(path: str | Path | None) -> Dict[str, Any]:
    with open(path or DEFAULT_CFG, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return {"train": dict(d.get("train") or {}), "mt": dict(d.get("mt") or {})}


def _cli_overrides(a: argparse.Namespace) -> Dict[str, Any]:
    """Only the flags that were actually given override the YAML."""
    keys = ("epochs", "batch", "device", "workers", "project", "name", "optimizer", "lr0", "amp", "cache", "seed", "patience")
    return {k: getattr(a, k) for k in keys if getattr(a, k, None) is not None}


def cmd_train(a: argparse.Namespace) -> int:
    from adas_mt.engine import MTConfig, MultiTaskTrainer

    cfg = _load_cfg(a.cfg)
    mt = dict(cfg["mt"])
    if a.imgsz:
        mt["imgsz"] = list(a.imgsz)
    if a.scale:
        mt["scale"] = a.scale
    if a.distill is not None:
        mt.setdefault("distill", {})["enabled"] = a.distill
    if a.teacher:
        mt.setdefault("distill", {})["teacher"] = a.teacher
    if a.teacher_ckpt:
        mt.setdefault("distill", {})["teacher_ckpt"] = a.teacher_ckpt
    overrides = {**cfg["train"], **_cli_overrides(a), "data": str(a.data), "model": a.model}
    if a.resume:
        overrides["resume"] = str(a.resume)
    trainer = MultiTaskTrainer(overrides=overrides, mt=MTConfig.from_dict(mt))
    trainer.train()
    return 0


def cmd_val(a: argparse.Namespace) -> int:
    from ultralytics.cfg import get_cfg

    from adas_mt.engine import MTConfig, MultiTaskValidator

    cfg = _load_cfg(a.cfg)
    mt = MTConfig.from_dict({**cfg["mt"], **({"imgsz": list(a.imgsz)} if a.imgsz else {})})
    args = get_cfg(overrides={"model": str(a.weights), "data": str(a.data), "batch": a.batch, "imgsz": max(mt.imgsz),
                              "device": a.device or "", "split": a.split, "nms": False, "plots": False, "conf": 0.001})
    stats = MultiTaskValidator(args=args, mt=mt)(model=str(a.weights))
    print(yaml.safe_dump({k: round(float(v), 5) for k, v in stats.items()}, sort_keys=False))
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    from adas_mt.data.convert_supervisely import main as convert_main

    return convert_main(a.rest)


def cmd_profile(a: argparse.Namespace) -> int:
    from adas_mt.nn import build_model
    from adas_mt.utils.profile import format_profile, profile_model

    print(format_profile(profile_model(build_model(a.scale, nc=a.nc, weights=a.weights), a.imgsz)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adas_mt", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train (stock Ultralytics trainer + multi-task extras)")
    t.add_argument("--data", required=True, type=Path, help="data.yaml written by the converter")
    t.add_argument("--model", default="yolo26s.pt", help="yolo26{n,s,m}.pt / .yaml, or a Stage-A / previous last.pt")
    t.add_argument("--cfg", type=Path, default=None, help="yaml with `train:` and `mt:` sections (default: adas_mt/cfg/default.yaml)")
    t.add_argument("--resume", type=Path, default=None, help="runs/<name>/weights/last.pt")
    t.add_argument("--imgsz", type=int, nargs=2, default=None, metavar=("H", "W"))
    t.add_argument("--scale", choices=list("nsmlx"), default=None)
    t.add_argument("--distill", action=argparse.BooleanOptionalAction, default=None, help="DINOv3 feature distillation")
    t.add_argument("--teacher", default=None)
    t.add_argument("--teacher_ckpt", default=None)
    for k, ty in (("epochs", int), ("batch", int), ("workers", int), ("seed", int), ("patience", int), ("lr0", float)):
        t.add_argument(f"--{k}", type=ty, default=None)
    for k in ("device", "project", "name", "optimizer"):
        t.add_argument(f"--{k}", default=None)
    t.add_argument("--amp", default=None, help="true | false | bf16")
    t.add_argument("--cache", default=None)
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("val", help="validate a checkpoint on the val split")
    v.add_argument("--weights", required=True, type=Path)
    v.add_argument("--data", required=True, type=Path)
    v.add_argument("--cfg", type=Path, default=None)
    v.add_argument("--imgsz", type=int, nargs=2, default=None, metavar=("H", "W"))
    v.add_argument("--batch", type=int, default=16)
    v.add_argument("--device", default=None)
    v.add_argument("--split", default="val")
    v.set_defaults(func=cmd_val)

    c = sub.add_parser("convert", help="Supervisely -> dataset (pass converter flags after --)")
    c.add_argument("rest", nargs=argparse.REMAINDER)
    c.set_defaults(func=cmd_convert)

    f = sub.add_parser("profile", help="params / GFLOPs per component of the deployed (fused) graph")
    f.add_argument("--scale", default="s", choices=list("nsmlx"))
    f.add_argument("--nc", type=int, default=9)
    f.add_argument("--weights", default=None)
    f.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    f.set_defaults(func=cmd_profile)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = build_parser().parse_args(argv)
    if a.cmd == "convert" and a.rest and a.rest[0] == "--":
        a.rest = a.rest[1:]
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
