"""Export padded square crops of every Supervisely 'traffic sign' rectangle.

The fine-grained class label is **not** available in the BDD/Supervisely export.
After running this script, the user manually labels crops by moving them into
``train/<class_name>/`` and ``val/<class_name>/`` directories (or by importing
labels from another dataset such as GTSRB or Mapillary Traffic Sign).

A ``manifest.csv`` is written alongside the crops with the bbox + source image,
so a labeling tool can read it back later.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("sign_classifier.extract_crops")


def _square_pad_box(x1: int, y1: int, x2: int, y2: int, pad: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh) * (1.0 + pad)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = side / 2
    nx1 = int(max(0, round(cx - half)))
    ny1 = int(max(0, round(cy - half)))
    nx2 = int(min(w, round(cx + half)))
    ny2 = int(min(h, round(cy + half)))
    return nx1, ny1, nx2, ny2


def _find_image(json_path: Path) -> Optional[Path]:
    stem = json_path.stem
    parent = json_path.parent
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        c = parent / f"{stem}{ext}"
        if c.is_file():
            return c
    for sib in ("img", "images"):
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            c = parent.parent / sib / f"{stem}{ext}"
            if c.is_file():
                return c
    return None


def extract(
    src: Path,
    dst: Path,
    padding: float = 0.15,
    min_side: int = 16,
    negative_per_image: int = 0,
    seed: int = 0,
) -> int:
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    crops_dir = dst / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if negative_per_image > 0:
        (dst / "unknown").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    manifest_rows: List[List[str]] = []
    n_signs = 0
    n_unknowns = 0

    for jp in sorted(src.rglob("*.json")):
        ip = _find_image(jp)
        if ip is None:
            continue
        try:
            with jp.open("r", encoding="utf-8") as f:
                ann = json.load(f)
        except json.JSONDecodeError:
            continue
        size = ann.get("size") or {}
        h, w = int(size.get("height", 0)), int(size.get("width", 0))
        im = cv2.imread(str(ip))
        if im is None:
            continue
        if not (h and w):
            h, w = im.shape[:2]

        sign_boxes_xyxy: List[Tuple[int, int, int, int]] = []
        for obj in ann.get("objects", []):
            title = (obj.get("classTitle") or "").lower()
            if title not in {"traffic sign", "traffic_sign"}:
                continue
            if (obj.get("geometryType") or "") != "rectangle":
                continue
            ext = (obj.get("points") or {}).get("exterior") or []
            if len(ext) != 2:
                continue
            (x1, y1), (x2, y2) = ext[0], ext[1]
            x1, x2 = sorted((int(x1), int(x2)))
            y1, y2 = sorted((int(y1), int(y2)))
            if x2 - x1 < min_side or y2 - y1 < min_side:
                continue
            nx1, ny1, nx2, ny2 = _square_pad_box(x1, y1, x2, y2, padding, w, h)
            crop = im[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                continue
            n_signs += 1
            out_name = f"{ip.stem}_{n_signs:06d}.jpg"
            cv2.imwrite(str(crops_dir / out_name), crop)
            manifest_rows.append([out_name, str(ip), nx1, ny1, nx2, ny2])
            sign_boxes_xyxy.append((nx1, ny1, nx2, ny2))

        # Hard-negative crops
        if negative_per_image > 0 and sign_boxes_xyxy:
            taken = 0
            attempts = 0
            avg_side = int(np.mean([(b[2] - b[0]) for b in sign_boxes_xyxy]))
            avg_side = max(min_side, min(avg_side, min(w, h) // 2))
            while taken < negative_per_image and attempts < negative_per_image * 8:
                attempts += 1
                side = rng.randint(int(avg_side * 0.7), int(avg_side * 1.3))
                cx = rng.randint(side // 2, w - side // 2 - 1)
                cy = rng.randint(side // 2, h - side // 2 - 1)
                nx1, ny1 = cx - side // 2, cy - side // 2
                nx2, ny2 = nx1 + side, ny1 + side
                # reject if it overlaps any sign
                bad = False
                for sx1, sy1, sx2, sy2 in sign_boxes_xyxy:
                    if not (nx2 <= sx1 or sx2 <= nx1 or ny2 <= sy1 or sy2 <= ny1):
                        bad = True
                        break
                if bad:
                    continue
                crop = im[ny1:ny2, nx1:nx2]
                if crop.size == 0:
                    continue
                n_unknowns += 1
                out_name = f"{ip.stem}_neg_{n_unknowns:06d}.jpg"
                cv2.imwrite(str(dst / "unknown" / out_name), crop)
                taken += 1

    with (dst / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop_filename", "source_image", "x1", "y1", "x2", "y2"])
        w.writerows(manifest_rows)
    LOGGER.info("wrote %d sign crops, %d hard-negative 'unknown' crops", n_signs, n_unknowns)
    return n_signs


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Extract traffic-sign crops from Supervisely dataset")
    p.add_argument("--src", type=Path, required=True, help="Supervisely dataset root")
    p.add_argument("--dst", type=Path, required=True, help="Output directory")
    p.add_argument("--padding", type=float, default=0.15)
    p.add_argument("--min_side", type=int, default=16)
    p.add_argument("--negative_per_image", type=int, default=0,
                   help="Number of random hard-negative crops per image (placed in dst/unknown/)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(list(argv) if argv is not None else None)
    extract(args.src, args.dst, padding=args.padding, min_side=args.min_side,
            negative_per_image=args.negative_per_image, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
