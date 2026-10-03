"""Convert a Supervisely-style annotation dump to a YOLO-format multi-task dataset.

Output layout::

    dst/
      images/{train,val}/<stem>.jpg
      labels_det/{train,val}/<stem>.txt          # YOLO box labels
      labels_da/{train,val}/<stem>.png           # uint8 mask (0=bg, 1=direct, 2=alternative)
      labels_ll/{train,val}/<stem>.png           # uint8 mask (0=bg, 1..N=lane classes)
      data.yaml
      lane_classes.json    # only when --lane_grouping=type
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

LOGGER = logging.getLogger("yolov13_multitask.convert")

DEFAULT_DET_CLASSES: Tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "rider",
    "traffic_light",
    "traffic_sign",
)

# Maps Supervisely classTitle -> base class name (without TL color split)
TITLE_TO_DET: Dict[str, str] = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "traffic sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
}

LANE_STYLE_TO_ID: Dict[str, int] = {"solid": 1, "dashed": 2}
LANE_STYLE_NAMES: Tuple[str, ...] = ("background", "solid", "dashed")

DA_NAMES: Tuple[str, ...] = ("background", "direct", "alternative")
DA_VALUE_MAP: Dict[str, int] = {"direct": 1, "alternative": 2}

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ConvertStats:
    images: int = 0
    boxes: int = 0
    da_polys: int = 0
    ll_polys: int = 0
    skipped_degenerate: int = 0
    skipped_unknown_class: Counter = field(default_factory=Counter)
    skipped_tag_parse: int = 0


def _parse_tags(tags: Optional[Sequence[dict]], stats: ConvertStats) -> Dict[str, str]:
    """Tag values in this Supervisely export are stringified Python dicts (single-quoted).

    Returns a flat dict of attributes. On parse failure, increments stats and returns {}.
    """
    out: Dict[str, str] = {}
    if not tags:
        return out
    for t in tags:
        v = t.get("value")
        if v is None:
            name = t.get("name")
            if name:
                out[name] = ""
            continue
        if isinstance(v, dict):
            out.update({str(k): str(vv) for k, vv in v.items()})
            continue
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, dict):
                    out.update({str(k): str(vv) for k, vv in parsed.items()})
                else:
                    name = t.get("name")
                    if name:
                        out[name] = str(parsed)
            except (ValueError, SyntaxError):
                stats.skipped_tag_parse += 1
                name = t.get("name")
                if name:
                    out[name] = v
        else:
            name = t.get("name")
            if name:
                out[name] = str(v)
    return out


def _det_class_name(title: str, attrs: Dict[str, str], split_tl_by_color: bool) -> Optional[str]:
    base = TITLE_TO_DET.get(title.lower())
    if base is None:
        return None
    if split_tl_by_color and base == "traffic_light":
        color = attrs.get("trafficLightColor", "none").lower()
        if color in {"red", "yellow", "green"}:
            return f"traffic_light_{color}"
        return None  # drop traffic lights with no/unknown color when splitting
    return base


def _build_det_classes(split_tl_by_color: bool) -> List[str]:
    classes = list(DEFAULT_DET_CLASSES)
    if split_tl_by_color:
        idx = classes.index("traffic_light")
        classes = (
            classes[:idx]
            + ["traffic_light_red", "traffic_light_yellow", "traffic_light_green"]
            + classes[idx + 1 :]
        )
    return classes


def _find_image(json_path: Path) -> Optional[Path]:
    """Locate the image paired with a Supervisely annotation JSON.

    Tries (in order):
        - sibling file with same stem and a known image extension
        - ../img/<stem>.<ext>  (canonical Supervisely layout)
        - ../images/<stem>.<ext>
    """
    stem = json_path.stem
    parent = json_path.parent
    candidates: List[Path] = []
    for ext in IMAGE_EXTS:
        candidates.append(parent / f"{stem}{ext}")
    for sibling in ("img", "images"):
        for ext in IMAGE_EXTS:
            candidates.append(parent.parent / sibling / f"{stem}{ext}")
    # Also strip a trailing .jpg/.png from the stem if Supervisely embedded it
    if "." in stem:
        bare = stem.rsplit(".", 1)[0]
        for ext in IMAGE_EXTS:
            candidates.append(parent.parent / "img" / f"{bare}{ext}")
            candidates.append(parent / f"{bare}{ext}")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _is_degenerate_polygon(pts: np.ndarray) -> bool:
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return True
    # area test via shoelace
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return area < 1.0


def _line_thickness(image_h: int) -> int:
    return max(2, int(round(4 * image_h / 720)))


def _split_for_path(p: Path, src_root: Path) -> Optional[str]:
    """Return 'train' or 'val' if the Supervisely directory structure indicates a split."""
    try:
        rel = p.relative_to(src_root)
    except ValueError:
        return None
    parts = [s.lower() for s in rel.parts]
    for tok in parts:
        if tok in {"train", "training"}:
            return "train"
        if tok in {"val", "valid", "validation", "test"}:
            return "val"
    return None


def _walk_supervisely(src: Path) -> List[Path]:
    return sorted(src.rglob("*.json"))


def _resolve_lane_class_id(
    attrs: Dict[str, str],
    grouping: str,
    type_to_id: Dict[str, int],
) -> int:
    if grouping == "style":
        style = attrs.get("laneStyle", "").lower()
        return LANE_STYLE_TO_ID.get(style, 0)
    if grouping == "type":
        ltype = attrs.get("laneType", "other").lower().strip()
        if ltype not in type_to_id:
            type_to_id[ltype] = len(type_to_id) + 1
        return type_to_id[ltype]
    raise ValueError(f"unknown lane grouping {grouping!r}")


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1.0 or bh < 1.0:
        return None
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    return cx, cy, bw / w, bh / h


def _process_one(
    json_path: Path,
    img_path: Path,
    out_dirs: Dict[str, Path],
    split: str,
    det_class_to_id: Dict[str, int],
    split_tl_by_color: bool,
    lane_grouping: str,
    type_to_id: Dict[str, int],
    stats: ConvertStats,
    copy_images: bool,
) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    size = ann.get("size") or {}
    h = int(size.get("height", 0))
    w = int(size.get("width", 0))
    if not (h and w):
        # fall back to image read
        im = cv2.imread(str(img_path))
        if im is None:
            LOGGER.warning("cannot read %s; skipping", img_path)
            return
        h, w = im.shape[:2]

    stem = img_path.stem
    det_lines: List[str] = []
    da_mask = np.zeros((h, w), dtype=np.uint8)
    ll_mask = np.zeros((h, w), dtype=np.uint8)
    line_thick = _line_thickness(h)

    for obj in ann.get("objects", []):
        title = (obj.get("classTitle") or "").strip()
        if not title:
            continue
        gtype = (obj.get("geometryType") or "").strip()
        ext = (obj.get("points") or {}).get("exterior") or []
        attrs = _parse_tags(obj.get("tags"), stats)

        if gtype == "rectangle":
            class_name = _det_class_name(title, attrs, split_tl_by_color)
            if class_name is None or class_name not in det_class_to_id:
                if class_name is None and title.lower() not in {"drivable area", "lane"}:
                    stats.skipped_unknown_class[title] += 1
                continue
            if len(ext) != 2:
                continue
            (x1, y1), (x2, y2) = ext[0], ext[1]
            yolo = _xyxy_to_yolo(float(x1), float(y1), float(x2), float(y2), w, h)
            if yolo is None:
                stats.skipped_degenerate += 1
                continue
            cx, cy, bw, bh = yolo
            cls_id = det_class_to_id[class_name]
            det_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats.boxes += 1

        elif title.lower() == "drivable area" and gtype == "polygon":
            pts = np.asarray(ext, dtype=np.int32)
            if _is_degenerate_polygon(pts):
                stats.skipped_degenerate += 1
                continue
            value = DA_VALUE_MAP.get(attrs.get("areaType", "").lower(), 0)
            if value == 0:
                continue
            try:
                cv2.fillPoly(da_mask, [pts.reshape(-1, 1, 2)], value)
                stats.da_polys += 1
            except cv2.error as e:  # pragma: no cover - defensive
                LOGGER.warning("DA fillPoly failed on %s: %s", json_path.name, e)

        elif title.lower() == "lane":
            if gtype == "polygon":
                pts = np.asarray(ext, dtype=np.int32)
                if _is_degenerate_polygon(pts):
                    stats.skipped_degenerate += 1
                    continue
                cls_id = _resolve_lane_class_id(attrs, lane_grouping, type_to_id)
                if cls_id == 0:
                    continue
                cv2.fillPoly(ll_mask, [pts.reshape(-1, 1, 2)], cls_id)
                stats.ll_polys += 1
            elif gtype == "line":
                if len(ext) < 2:
                    stats.skipped_degenerate += 1
                    continue
                pts = np.asarray(ext, dtype=np.int32).reshape(-1, 1, 2)
                cls_id = _resolve_lane_class_id(attrs, lane_grouping, type_to_id)
                if cls_id == 0:
                    continue
                cv2.polylines(ll_mask, [pts], isClosed=False, color=cls_id, thickness=line_thick)
                stats.ll_polys += 1

    # Write outputs ----------------------------------------------------------------
    img_out = out_dirs["images"] / split / f"{stem}.jpg"
    if copy_images or img_path.suffix.lower() != ".jpg":
        # transcode/copy to a uniform jpg
        if img_path.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copyfile(img_path, img_out)
        else:
            im = cv2.imread(str(img_path))
            if im is None:
                LOGGER.warning("cannot read %s; skipping", img_path)
                return
            cv2.imwrite(str(img_out), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        try:
            if img_out.exists() or img_out.is_symlink():
                img_out.unlink()
            img_out.symlink_to(img_path.resolve())
        except OSError:
            shutil.copyfile(img_path, img_out)

    (out_dirs["det"] / split / f"{stem}.txt").write_text("\n".join(det_lines), encoding="utf-8")
    cv2.imwrite(str(out_dirs["da"] / split / f"{stem}.png"), da_mask)
    cv2.imwrite(str(out_dirs["ll"] / split / f"{stem}.png"), ll_mask)
    stats.images += 1


def convert(
    src: Path,
    dst: Path,
    lane_grouping: str = "style",
    split_tl_by_color: bool = False,
    val_fraction: float = 0.1,
    seed: int = 0,
    copy_images: bool = True,
    limit: Optional[int] = None,
) -> ConvertStats:
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    assert lane_grouping in {"style", "type"}, lane_grouping

    out_dirs = {
        "images": dst / "images",
        "det": dst / "labels_det",
        "da": dst / "labels_da",
        "ll": dst / "labels_ll",
    }
    for base in out_dirs.values():
        for s in ("train", "val"):
            (base / s).mkdir(parents=True, exist_ok=True)

    det_classes = _build_det_classes(split_tl_by_color)
    det_class_to_id = {n: i for i, n in enumerate(det_classes)}

    type_to_id: Dict[str, int] = {}  # populated in 'type' grouping

    json_files = _walk_supervisely(src)
    if limit:
        json_files = json_files[:limit]
    LOGGER.info("found %d annotation JSONs under %s", len(json_files), src)

    # Pair JSONs with images and assign splits
    paired: List[Tuple[Path, Path, Optional[str]]] = []
    unpaired = 0
    for jp in json_files:
        ip = _find_image(jp)
        if ip is None:
            unpaired += 1
            continue
        paired.append((jp, ip, _split_for_path(jp, src)))
    if unpaired:
        LOGGER.warning("%d annotations had no matching image", unpaired)

    have_split = any(s is not None for _, _, s in paired)
    if not have_split:
        rng = random.Random(seed)
        all_idx = list(range(len(paired)))
        rng.shuffle(all_idx)
        n_val = max(1, int(round(len(paired) * val_fraction)))
        val_set = set(all_idx[:n_val])
        paired = [
            (jp, ip, "val" if i in val_set else "train") for i, (jp, ip, _) in enumerate(paired)
        ]
    else:
        # default any unknown to train
        paired = [(jp, ip, s or "train") for jp, ip, s in paired]

    stats = ConvertStats()
    for jp, ip, split in paired:
        try:
            _process_one(
                jp,
                ip,
                out_dirs,
                split,  # type: ignore[arg-type]
                det_class_to_id,
                split_tl_by_color,
                lane_grouping,
                type_to_id,
                stats,
                copy_images,
            )
        except Exception as e:  # pragma: no cover
            LOGGER.exception("failed on %s: %s", jp, e)

    # data.yaml --------------------------------------------------------------------
    if lane_grouping == "style":
        ll_names = list(LANE_STYLE_NAMES)
    else:
        # build deterministic ordering: id 0 = background, then 1..N
        ll_names = ["background"] + [
            n for n, _ in sorted(type_to_id.items(), key=lambda kv: kv[1])
        ]
        with (dst / "lane_classes.json").open("w", encoding="utf-8") as f:
            json.dump({**{"background": 0}, **type_to_id}, f, indent=2, sort_keys=True)

    data_yaml = {
        "path": str(dst),
        "train": "images/train",
        "val": "images/val",
        "nc": len(det_classes),
        "names": det_classes,
        "da_classes": len(DA_NAMES),
        "da_names": list(DA_NAMES),
        "ll_classes": len(ll_names),
        "ll_names": ll_names,
    }
    with (dst / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    LOGGER.info(
        "done: %d images, %d boxes, %d DA polys, %d LL polys, %d skipped degenerate, %d unknown-tag-parses",
        stats.images,
        stats.boxes,
        stats.da_polys,
        stats.ll_polys,
        stats.skipped_degenerate,
        stats.skipped_tag_parse,
    )
    if stats.skipped_unknown_class:
        LOGGER.info("unknown classTitles skipped: %s", dict(stats.skipped_unknown_class))
    return stats


def build_parser(p: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(description="Supervisely -> YOLO multi-task converter")
    p.add_argument("--src", type=Path, required=True, help="Supervisely dataset root")
    p.add_argument("--dst", type=Path, required=True, help="Output dataset root (created if missing)")
    p.add_argument(
        "--lane_grouping",
        choices=("style", "type"),
        default="style",
        help="style: 1=solid 2=dashed; type: id per laneType value",
    )
    p.add_argument("--split_tl_by_color", action="store_true")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (faster on large datasets, requires JPG sources)",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N annotations (debug)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
