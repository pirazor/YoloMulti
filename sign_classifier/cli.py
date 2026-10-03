"""Sign classifier CLI: ``python -m sign_classifier <subcommand>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence


def cmd_extract(args: argparse.Namespace) -> int:
    from sign_classifier.extract_crops import extract

    extract(
        src=args.src, dst=args.dst, padding=args.padding,
        min_side=args.min_side, negative_per_image=args.negative_per_image, seed=args.seed,
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from sign_classifier.train import train

    train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        lr=args.lr, weight_decay=args.weight_decay, workers=args.workers,
        model_name=args.model, device=args.device, save_dir=args.save_dir,
        label_smoothing=args.label_smoothing,
    )
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    import cv2

    from sign_classifier.infer import SignClassifier

    clf = SignClassifier(weights=args.weights, device=args.device, imgsz=args.imgsz, padding=args.padding)
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    boxes = []
    if args.boxes:
        for line in Path(args.boxes).read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            boxes.append([float(p) for p in parts[:4]])
    results = clf.classify(img, boxes)
    for box, (cls_name, conf) in zip(boxes, results):
        print(f"{box} -> {cls_name} (conf={conf:.4f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sign_classifier", description="Fine-grained traffic-sign classifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract sign crops from Supervisely dataset")
    pe.add_argument("--src", type=Path, required=True)
    pe.add_argument("--dst", type=Path, required=True)
    pe.add_argument("--padding", type=float, default=0.15)
    pe.add_argument("--min_side", type=int, default=16)
    pe.add_argument("--negative_per_image", type=int, default=0)
    pe.add_argument("--seed", type=int, default=0)
    pe.set_defaults(func=cmd_extract)

    pt = sub.add_parser("train")
    pt.add_argument("--data", type=Path, required=True)
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--batch", type=int, default=64)
    pt.add_argument("--imgsz", type=int, default=96)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--model", type=str, default="mobilenetv3_small_100")
    pt.add_argument("--device", type=str, default=None)
    pt.add_argument("--save_dir", type=Path, default=Path("runs/sign_classifier/exp"))
    pt.add_argument("--label_smoothing", type=float, default=0.1)
    pt.set_defaults(func=cmd_train)

    pi = sub.add_parser("infer")
    pi.add_argument("--weights", type=Path, required=True)
    pi.add_argument("--image", type=Path, required=True)
    pi.add_argument("--boxes", type=Path, default=None,
                    help="text file with one xyxy box per line")
    pi.add_argument("--imgsz", type=int, default=96)
    pi.add_argument("--padding", type=float, default=0.15)
    pi.add_argument("--device", type=str, default=None)
    pi.set_defaults(func=cmd_infer)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
