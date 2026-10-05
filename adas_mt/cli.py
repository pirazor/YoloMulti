"""``python -m adas_mt <train|val|convert|profile|export|trt-build|predict|bench|eval> ...``"""

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


def cmd_export(a: argparse.Namespace) -> int:
    from adas_mt.export import export_onnx

    r = export_onnx(a.weights, a.out, imgsz=a.imgsz, batch=a.batch, dynamic=a.dynamic, seg_dtype=a.seg_dtype, opset=a.opset,
                    simplify=a.simplify, decompose_pixel_shuffle=a.decompose_pixel_shuffle, verify=a.verify,
                    verify_image=a.verify_image, det_head=a.det_head, trt_topk=a.trt_topk)
    print(f"{r.onnx}\n{r.meta_file}\nparity: " + ", ".join(f"{k}={v:.3g}" for k, v in r.parity.items()))
    return 0


def cmd_trt_build(a: argparse.Namespace) -> int:
    from adas_mt.deploy.trt_build import build_engine, trtexec_command, trtexec_timing_command

    r = build_engine(a.onnx, a.engine, a.precision, a.workspace_mb, a.calib, a.calib_n, a.calib_cache, a.keep_fp16,
                     a.timing_cache, a.opt_batch, a.max_batch, a.opt_level, a.verbose)
    print(f"{r.engine} ({r.seconds:.0f} s)\nbuild equivalent: " + trtexec_command(a.onnx, r.engine, a.precision, a.workspace_mb, a.calib_cache)
          + "\ntime the engine: " + trtexec_timing_command(r.engine))
    return 0


def cmd_predict(a: argparse.Namespace) -> int:
    from adas_mt.deploy.runner import predict

    stats = predict(a.model, a.source, a.out, a.conf, a.backend, a.device, a.gpu_preprocess, a.save_masks, a.show,
                    a.max_frames, a.fps)
    print(yaml.safe_dump(stats, sort_keys=False))
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    import json

    from adas_mt.deploy.runner import Pipeline, bench, format_bench, iter_frames

    frame = next(iter(iter_frames(a.source)))[1] if a.source else None
    pipe = Pipeline(a.model, a.backend, a.conf, a.device, a.gpu_preprocess)
    r = bench(pipe, frame, a.n, a.warmup)
    print(format_bench(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from adas_mt.deploy.evaluate import evaluate

    stats, speed = evaluate(a.model, a.data, a.batch, a.device, a.split, a.backend, a.conf, a.workers)
    print(yaml.safe_dump({**{k: round(float(v), 5) for k, v in stats.items()}, **{k: round(v, 3) for k, v in speed.items()}},
                         sort_keys=False))
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

    e = sub.add_parser("export", help="checkpoint -> ONNX (fused, NMS-free, argmax in the graph) + metadata JSON")
    e.add_argument("--weights", required=True, type=Path, help="runs/<name>/weights/best.pt (imgsz is read from the run's mt.yaml)")
    e.add_argument("--out", type=Path, default=None, help="default: next to the weights")
    e.add_argument("--imgsz", type=int, nargs=2, default=None, metavar=("H", "W"))
    e.add_argument("--batch", type=int, default=1)
    e.add_argument("--dynamic", action="store_true", help="free batch axis (engines then need an optimisation profile)")
    e.add_argument("--seg-dtype", dest="seg_dtype", choices=["int32", "uint8", "logits"], default="int32",
                   help="class-map dtype; uint8 needs TensorRT >= 10, logits is for debugging")
    e.add_argument("--opset", type=int, default=17)
    e.add_argument("--no-simplify", dest="simplify", action="store_false")
    e.add_argument("--no-verify", dest="verify", action="store_false", help="skip the ONNX Runtime parity check")
    e.add_argument("--verify-image", default=None, help="a real frame for the parity check (recommended for trained models)")
    e.add_argument("--decompose-pixel-shuffle", action="store_true",
                   help="replace DepthToSpace by Reshape/Transpose/Reshape if the engine build rejects it")
    e.add_argument("--det-head", dest="det_head", choices=["topk", "raw"], default="topk",
                   help="topk: NMS-free top-k in the graph (default). raw: dense predictions, top-k on the host "
                        "(fallback for INT8 on TensorRT 10.3.0 / JetPack 6.x)")
    e.add_argument("--no-trt-topk", dest="trt_topk", action="store_false",
                   help="one big TopK instead of the grouped exact top-k Ultralytics uses for TensorRT")
    e.set_defaults(func=cmd_export)

    b = sub.add_parser("trt-build", help="ONNX -> TensorRT engine (run ON the Jetson)")
    b.add_argument("--onnx", required=True, type=Path)
    b.add_argument("--engine", type=Path, default=None)
    b.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    b.add_argument("--workspace-mb", type=int, default=1024)
    b.add_argument("--calib", default=None, help="INT8: image directory or data.yaml (its train split)")
    b.add_argument("--calib-n", type=int, default=512)
    b.add_argument("--calib-cache", type=Path, default=None, help="reuse/write the calibration cache (delete it when the model changes)")
    b.add_argument("--keep-fp16", nargs="+", default=["heads"], metavar="X",
                   help="INT8: components kept in FP16: heads | seg | none, or ONNX node-name substrings")
    b.add_argument("--timing-cache", type=Path, default=None)
    b.add_argument("--opt-batch", type=int, default=None)
    b.add_argument("--max-batch", type=int, default=None)
    b.add_argument("--opt-level", type=int, default=None, help="builder optimisation level (TensorRT default 3; 5 = slowest build)")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_trt_build)

    def runner_args(sp, source_required: bool) -> None:
        sp.add_argument("--model", required=True, type=Path, help=".engine (TensorRT) or .onnx (ONNX Runtime)")
        sp.add_argument("--source", required=source_required, default=None, help="image | directory | video | camera index | gstreamer pipeline")
        sp.add_argument("--conf", type=float, default=0.25)
        sp.add_argument("--backend", choices=["auto", "trt", "ort"], default="auto")
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--gpu-preprocess", action="store_true", help="letterbox on the GPU (saves the CPU resize on a Jetson)")

    r = sub.add_parser("predict", help="run an exported model on images / video / a camera and save overlays")
    runner_args(r, True)
    r.add_argument("--out", type=Path, default=Path("runs/predict"))
    r.add_argument("--save-masks", action="store_true")
    r.add_argument("--show", action="store_true")
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--fps", type=float, default=None, help="output video frame rate")
    r.set_defaults(func=cmd_predict)

    n = sub.add_parser("bench", help="end-to-end latency (pre / infer / post, p50-p99) of an exported model")
    runner_args(n, False)
    n.add_argument("--n", type=int, default=300)
    n.add_argument("--warmup", type=int, default=50)
    n.add_argument("--json", default=None, help="also write the report as JSON")
    n.set_defaults(func=cmd_bench)

    ev = sub.add_parser("eval", help="mAP / DA mIoU / lane IoU of an exported model (what export + quantisation cost)")
    ev.add_argument("--model", required=True, type=Path)
    ev.add_argument("--data", required=True, type=Path)
    ev.add_argument("--split", default="val")
    ev.add_argument("--batch", type=int, default=16)
    ev.add_argument("--device", default=None, help="default: 0 for an .engine, cpu for an .onnx")
    ev.add_argument("--backend", choices=["auto", "trt", "ort"], default="auto")
    ev.add_argument("--conf", type=float, default=0.001)
    ev.add_argument("--workers", type=int, default=2)
    ev.set_defaults(func=cmd_eval)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    a = build_parser().parse_args(argv)
    if a.cmd == "convert" and a.rest and a.rest[0] == "--":
        a.rest = a.rest[1:]
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
