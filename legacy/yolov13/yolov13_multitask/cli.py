"""Command-line entry point: ``python -m yolov13_multitask <subcommand> ...``.

Subcommands: convert | train | val | predict | export.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


# --------------------------------------------------------------------- subcmds
def cmd_convert(args: argparse.Namespace) -> int:
    from yolov13_multitask.data.convert_supervisely import convert

    convert(
        src=args.src,
        dst=args.dst,
        lane_grouping=args.lane_grouping,
        split_tl_by_color=args.split_tl_by_color,
        val_fraction=args.val_fraction,
        seed=args.seed,
        copy_images=not args.symlink,
        limit=args.limit,
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from yolov13_multitask.engine.train import MultiTaskTrainer

    trainer = MultiTaskTrainer(
        data=args.data,
        cfg=args.cfg,
        weights=args.weights,
        model_yaml=args.model_yaml,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
    trainer.train()
    return 0


def cmd_val(args: argparse.Namespace) -> int:
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from yolov13_multitask.data.multitask_dataset import MultiTaskDataset
    from yolov13_multitask.engine.val import MultiTaskValidator

    with open(args.data, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ckpt = torch.load(str(args.weights), map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = MultiTaskDataset(args.data, split="val", imgsz=args.imgsz, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=MultiTaskDataset.collate_fn,
    )
    validator = MultiTaskValidator(
        data=data, device=device,
        da_classes=int(data.get("da_classes", 3)),
        ll_classes=int(data.get("ll_classes", 3)),
    )
    metrics = validator.run(model.to(device).float(), loader)
    print(metrics)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from yolov13_multitask.engine.predict import MultiTaskPredictor

    pred = MultiTaskPredictor(
        weights=args.weights, device=args.device, imgsz=args.imgsz,
        conf_thres=args.conf, iou_thres=args.iou,
    )
    out_dir = Path(args.save_dir or "runs/multitask/predict")
    pred.run(args.source, out_dir, save=args.save)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from yolov13_multitask.export import export_onnx, export_tensorrt

    if args.format == "onnx":
        export_onnx(
            weights=args.weights, output=args.output, imgsz=args.imgsz,
            opset=args.opset, dynamic=args.dynamic, verify=args.verify,
        )
    elif args.format in {"tensorrt", "trt", "engine"}:
        export_tensorrt(
            onnx_path=args.weights, output=args.output, fp16=args.fp16,
            int8=args.int8, calib_dir=args.calib_dir,
        )
    else:
        raise SystemExit(f"unsupported format: {args.format}")
    return 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yolov13_multitask", description="Multi-task YOLOv13 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # convert -------------------------------------------------------
    pc = sub.add_parser("convert", help="Supervisely -> YOLO multi-task dataset")
    pc.add_argument("--src", type=Path, required=True)
    pc.add_argument("--dst", type=Path, required=True)
    pc.add_argument("--lane_grouping", choices=("style", "type"), default="style")
    pc.add_argument("--split_tl_by_color", action="store_true")
    pc.add_argument("--val_fraction", type=float, default=0.1)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--symlink", action="store_true")
    pc.add_argument("--limit", type=int, default=None)
    pc.set_defaults(func=cmd_convert)

    # train ---------------------------------------------------------
    pt = sub.add_parser("train")
    pt.add_argument("--data", type=Path, required=True)
    pt.add_argument("--cfg", type=Path, required=True)
    pt.add_argument("--weights", type=Path, default=None, help="pretrained YOLOv13 weights")
    pt.add_argument("--model_yaml", type=Path, default=None)
    pt.add_argument("--device", type=str, default=None)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--batch", type=int, default=None)
    pt.add_argument("--imgsz", type=int, default=None)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--project", type=str, default=None)
    pt.add_argument("--name", type=str, default=None)
    pt.add_argument("--resume", type=Path, default=None)
    pt.set_defaults(func=cmd_train)

    # val -----------------------------------------------------------
    pv = sub.add_parser("val")
    pv.add_argument("--weights", type=Path, required=True)
    pv.add_argument("--data", type=Path, required=True)
    pv.add_argument("--imgsz", type=int, default=640)
    pv.add_argument("--batch", type=int, default=8)
    pv.add_argument("--workers", type=int, default=2)
    pv.add_argument("--device", type=str, default=None)
    pv.set_defaults(func=cmd_val)

    # predict -------------------------------------------------------
    pp = sub.add_parser("predict")
    pp.add_argument("--weights", type=Path, required=True)
    pp.add_argument("--source", type=str, required=True)
    pp.add_argument("--imgsz", type=int, default=640)
    pp.add_argument("--conf", type=float, default=0.25)
    pp.add_argument("--iou", type=float, default=0.45)
    pp.add_argument("--device", type=str, default=None)
    pp.add_argument("--save", action="store_true", default=True)
    pp.add_argument("--save_dir", type=Path, default=None)
    pp.set_defaults(func=cmd_predict)

    # export --------------------------------------------------------
    pe = sub.add_parser("export")
    pe.add_argument("--weights", type=Path, required=True, help="for ONNX: .pt; for TRT: .onnx")
    pe.add_argument("--output", type=Path, default=None)
    pe.add_argument("--format", choices=("onnx", "tensorrt", "trt", "engine"), default="onnx")
    pe.add_argument("--imgsz", type=int, default=640)
    pe.add_argument("--opset", type=int, default=17)
    pe.add_argument("--dynamic", action="store_true")
    pe.add_argument("--verify", action="store_true", default=True)
    pe.add_argument("--fp16", action="store_true")
    pe.add_argument("--int8", action="store_true")
    pe.add_argument("--calib_dir", type=Path, default=None)
    pe.set_defaults(func=cmd_export)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    _setup_logging()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
