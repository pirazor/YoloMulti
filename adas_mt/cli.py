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
    extra = set(d) - {"train", "mt"}
    if extra:
        raise ValueError(f"unknown top-level sections {sorted(extra)} in {path or DEFAULT_CFG} (valid: train, mt)")
    return {"train": dict(d.get("train") or {}), "mt": dict(d.get("mt") or {})}


def _bool_or_str(v: Any, allowed: tuple = ()) -> Any:
    """'true'/'false' (any case) -> bool; other strings must be in ``allowed`` (Ultralytics rejects the rest)."""
    if isinstance(v, str) and v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if isinstance(v, str) and allowed and v.lower() not in allowed:
        raise ValueError(f"invalid value {v!r}; expected true, false or one of {allowed}")
    return v.lower() if isinstance(v, str) else v


def _cli_overrides(a: argparse.Namespace) -> Dict[str, Any]:
    """Only the flags that were actually given override the YAML."""
    keys = ("epochs", "batch", "device", "workers", "project", "name", "optimizer", "lr0", "amp", "cache", "seed", "patience")
    out = {k: getattr(a, k) for k in keys if getattr(a, k, None) is not None}
    if "amp" in out:
        out["amp"] = _bool_or_str(out["amp"], ("bf16", "fp16", "fp32"))
    if "cache" in out:
        out["cache"] = _bool_or_str(out["cache"], ("ram", "disk"))  # cache=True means RAM; the string 'true' means nothing
    return out


def cmd_train(a: argparse.Namespace) -> int:
    from adas_mt.engine import MTConfig, MultiTaskTrainer

    cfg = _load_cfg(a.cfg)
    mt = dict(cfg["mt"])
    given = {}
    if a.imgsz:
        given["imgsz"] = list(a.imgsz)
    if a.scale:
        given["scale"] = a.scale
    dist = {}
    if a.distill is not None:
        dist["enabled"] = a.distill
    if a.teacher:
        dist["teacher"] = a.teacher
    if a.teacher_ckpt:
        dist["teacher_ckpt"] = a.teacher_ckpt
    overrides = {**cfg["train"], **_cli_overrides(a), "data": str(a.data), "model": a.model}
    overrides.pop("imgsz", None)  # the geometry is mt.imgsz
    if a.resume:
        # The run's own mt.yaml is authoritative (see MultiTaskTrainer); multi-task flags would conflict with it.
        if given or dist:
            logging.getLogger("adas_mt").warning("--resume: ignoring --imgsz/--scale/--distill/--teacher flags; "
                                                 "the run's mt.yaml is used")
        overrides["resume"] = str(a.resume)
        trainer = MultiTaskTrainer(overrides=overrides, mt=None)
    else:
        mt.update(given)
        if dist:
            mt["distill"] = {**(mt.get("distill") or {}), **dist}
        trainer = MultiTaskTrainer(overrides=overrides, mt=MTConfig.from_dict(mt))
    trainer.train()
    return 0


def cmd_val(a: argparse.Namespace) -> int:
    from adas_mt.engine.val import run_validation

    stats, _ = run_validation(a.weights, a.data, batch=a.batch, device=a.device or "", split=a.split,
                              imgsz=a.imgsz, cfg_path=a.cfg)
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
    t.add_argument("--amp", default=None, help="true | false | bf16 | fp16 | fp32")
    t.add_argument("--cache", default=None, help="true (=ram) | false | ram | disk")
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
