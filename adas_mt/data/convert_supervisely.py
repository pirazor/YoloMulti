"""Convert a Supervisely-style annotation dump to a YOLO-format multi-task dataset.

Changes vs the legacy converter: lane lines are drawn thicker by default (8 px at 720p,
scaled with image height) and ``--partial_annotation`` skips writing a task mask when the
image carries no object of that task, so the trainer ignores it instead of learning "all
background".


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
import base64
import json
import logging
import random
import shutil
import zlib
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
    skipped_unmapped: int = 0  # DA/lane objects whose attribute value has no class id
    skipped_geometry: Counter = field(default_factory=Counter)  # "<classTitle>/<geometryType>" the converter cannot draw
    size_mismatch: int = 0  # annotations whose `size` differs from the image file (coordinates were scaled)
    renamed_stems: int = 0  # output names disambiguated because two images share a stem


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


def _draw_polygon(mask: np.ndarray, exterior, interior, value: int) -> None:
    """Fill ``exterior`` with ``value`` but leave Supervisely ``interior`` rings (holes) untouched."""
    tmp = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(tmp, [np.asarray(exterior, np.int32).reshape(-1, 1, 2)], 1)
    for ring in interior or []:
        if len(ring) >= 3:
            cv2.fillPoly(tmp, [np.asarray(ring, np.int32).reshape(-1, 1, 2)], 0)
    mask[tmp == 1] = value


def _line_thickness(image_hw: Tuple[int, int], ref_px: float = 8.0) -> int:
    """Lane line thickness: ``ref_px`` for a 1280 px long side, scaled with the long side (min 2 px).

    The trainer scales the LONG side to the training width, so this keeps the drawn lane ~ref_px/2 px wide at 640
    whatever the aspect ratio (scaling with the height alone gave 4:3 sources 1.5x thicker lanes)."""
    return max(2, int(round(ref_px * max(image_hw) / 1280)))


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    """(h, w) from the image header without decoding the pixels; None when unreadable."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.size[1]), int(im.size[0])
    except (OSError, ValueError):
        return None


def _decode_bitmap(obj: dict) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """Supervisely ``bitmap`` geometry: base64 of a (zlib-compressed) PNG plus ``origin`` [x, y] in the image.

    Returns ``(bool mask, (ox, oy))`` or None when the object carries no usable bitmap."""
    bm = obj.get("bitmap") or {}
    data, origin = bm.get("data"), bm.get("origin")
    if not data or not origin or len(origin) != 2:
        return None
    try:
        raw = base64.b64decode(data)
        try:
            raw = zlib.decompress(raw)  # the Supervisely SDK compresses the PNG; some exports do not
        except zlib.error:
            pass
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except (ValueError, cv2.error):
        return None
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = arr[..., 3] if arr.shape[2] == 4 else arr[..., 0]  # RGBA: the alpha plane is the mask
    return arr.astype(bool), (int(origin[0]), int(origin[1]))


def _paste_bitmap(mask: np.ndarray, bm: np.ndarray, origin: Tuple[int, int], value: int, sx: float, sy: float) -> None:
    """Write ``value`` where the bitmap is set, placed at ``origin`` (scaled by sx/sy, clipped to the image)."""
    if sx != 1.0 or sy != 1.0:
        bm = cv2.resize(bm.astype(np.uint8), (max(1, int(round(bm.shape[1] * sx))), max(1, int(round(bm.shape[0] * sy)))),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    ox, oy = int(round(origin[0] * sx)), int(round(origin[1] * sy))
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + bm.shape[1], w), min(oy + bm.shape[0], h)
    if x1 <= x0 or y1 <= y0:
        return
    region = mask[y0:y1, x0:x1]
    region[bm[y0 - oy : y1 - oy, x0 - ox : x1 - ox]] = value


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


def assign_val(
    names: Sequence[str], val_fraction: float, seed: int = 0, group_regex: Optional[str] = None
) -> set:
    """Return the set of indices (into ``names``) that go to the val split.

    ``val_fraction == 0`` -> no val images (the old ``max(1, ...)`` forced one). With
    ``group_regex`` (first capture group = clip/sequence id, e.g. ``^([0-9a-f]{8})-`` for BDD
    names) whole groups are assigned, so consecutive frames of one clip never straddle
    train/val. Without it the split is per image, which leaks near-duplicate frames.
    """
    n = len(names)
    if val_fraction <= 0 or n == 0:
        return set()
    target = max(1, int(round(n * val_fraction))) if n > 1 else 0
    rng = random.Random(seed)
    if group_regex:
        import re

        rx = re.compile(group_regex)
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, nm in enumerate(names):
            m = rx.search(nm)
            groups[m.group(1) if m else nm].append(i)  # no match -> its own group
        keys = sorted(groups)
        rng.shuffle(keys)
        val: set = set()
        for k in keys:
            if len(val) >= target:
                break
            val.update(groups[k])
        return val
    idx = list(range(n))
    rng.shuffle(idx)
    return set(idx[:target])


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
    lane_thickness: float = 8.0,
    partial_annotation: bool = False,
    out_stem: Optional[str] = None,
) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    size = ann.get("size") or {}
    h = int(size.get("height", 0))
    w = int(size.get("width", 0))
    real = _image_size(img_path)  # header only; the JSON `size` is never trusted blindly
    if real is None:
        LOGGER.warning("cannot read %s; skipping", img_path)
        return
    if not (h and w):
        h, w = real
    sx = sy = 1.0
    if (h, w) != real:
        # Supervisely coordinates live in the frame of `size`; an image that was resized afterwards would otherwise
        # get misplaced masks and mis-normalised boxes in silence. Scale into the real frame and say so.
        stats.size_mismatch += 1
        LOGGER.warning("%s: annotation size %dx%d but the image is %dx%d: scaling the coordinates", json_path.name, w, h,
                       real[1], real[0])
        sx, sy = real[1] / w, real[0] / h
        h, w = real

    def scaled(points) -> np.ndarray:
        return np.asarray(points, dtype=np.float64).reshape(-1, 2) * (sx, sy)

    def ipts(points) -> np.ndarray:
        return np.rint(scaled(points)).astype(np.int32)

    stem = out_stem or img_path.stem
    det_lines: List[str] = []
    da_mask = np.zeros((h, w), dtype=np.uint8)
    ll_mask = np.zeros((h, w), dtype=np.uint8)
    line_thick = _line_thickness((h, w), lane_thickness)
    da_seen = ll_seen = False

    for obj in ann.get("objects", []):
        title = (obj.get("classTitle") or "").strip()
        if not title:
            continue
        gtype = (obj.get("geometryType") or "").strip()
        ext = (obj.get("points") or {}).get("exterior") or []
        holes = [ipts(ring) for ring in ((obj.get("points") or {}).get("interior") or [])]
        attrs = _parse_tags(obj.get("tags"), stats)
        is_da, is_lane = title.lower() == "drivable area", title.lower() == "lane"

        if gtype == "rectangle":
            class_name = _det_class_name(title, attrs, split_tl_by_color)
            if class_name is None or class_name not in det_class_to_id:
                if class_name is None and not (is_da or is_lane):
                    stats.skipped_unknown_class[title] += 1
                continue
            if len(ext) != 2:
                continue
            (x1, y1), (x2, y2) = scaled(ext).tolist()
            yolo = _xyxy_to_yolo(float(x1), float(y1), float(x2), float(y2), w, h)
            if yolo is None:
                stats.skipped_degenerate += 1
                continue
            cx, cy, bw, bh = yolo
            cls_id = det_class_to_id[class_name]
            det_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats.boxes += 1

        elif is_da and gtype in {"polygon", "bitmap"}:
            value = DA_VALUE_MAP.get(attrs.get("areaType", "").lower(), 0)
            if value == 0:  # unmapped areaType: do not count the task as annotated for this image
                stats.skipped_unmapped += 1
                continue
            if gtype == "bitmap":
                decoded = _decode_bitmap(obj)
                if decoded is None:
                    stats.skipped_degenerate += 1
                    continue
                _paste_bitmap(da_mask, decoded[0], decoded[1], value, sx, sy)
            else:
                pts = ipts(ext)
                if _is_degenerate_polygon(pts):
                    stats.skipped_degenerate += 1
                    continue
                try:
                    _draw_polygon(da_mask, pts, holes, value)
                except cv2.error as e:  # pragma: no cover - defensive
                    LOGGER.warning("DA fillPoly failed on %s: %s", json_path.name, e)
                    continue
            da_seen = True
            stats.da_polys += 1

        elif is_lane and gtype in {"polygon", "line", "bitmap"}:
            cls_id = _resolve_lane_class_id(attrs, lane_grouping, type_to_id)
            if cls_id == 0:
                stats.skipped_unmapped += 1
                continue
            if gtype == "polygon":
                pts = ipts(ext)
                if _is_degenerate_polygon(pts):
                    stats.skipped_degenerate += 1
                    continue
                _draw_polygon(ll_mask, pts, holes, cls_id)
            elif gtype == "line":
                if len(ext) < 2:
                    stats.skipped_degenerate += 1
                    continue
                cv2.polylines(ll_mask, [ipts(ext).reshape(-1, 1, 2)], isClosed=False, color=cls_id, thickness=line_thick)
            else:
                decoded = _decode_bitmap(obj)
                if decoded is None:
                    stats.skipped_degenerate += 1
                    continue
                _paste_bitmap(ll_mask, decoded[0], decoded[1], cls_id, sx, sy)
            ll_seen = True
            stats.ll_polys += 1

        else:  # a geometry this converter cannot draw (e.g. a 'car' polygon, a DA 'point'): never silently dropped
            stats.skipped_geometry[f"{title}/{gtype or '?'}"] += 1

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
    for key, mask, seen in (("da", da_mask, da_seen), ("ll", ll_mask, ll_seen)):
        png = out_dirs[key] / split / f"{stem}.png"
        if partial_annotation and not seen:
            png.unlink(missing_ok=True)  # missing PNG == task unannotated for this image
        else:
            cv2.imwrite(str(png), mask)
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
    lane_thickness: float = 8.0,
    partial_annotation: bool = False,
    group_regex: Optional[str] = None,
    overwrite: bool = False,
) -> ConvertStats:
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    assert lane_grouping in {"style", "type"}, lane_grouping

    # A second run into the same folder would leave the first run's files behind: images re-split with a
    # different seed end up in BOTH train and val (leakage) and stale masks survive.
    existing = [dst / d for d in ("images", "labels_det", "labels_da", "labels_ll") if (dst / d).exists()]
    if any(f.is_file() for d in existing for f in d.rglob("*")):
        if not overwrite:
            raise FileExistsError(f"{dst} already contains a converted dataset; pass overwrite=True / --overwrite")
        for d in existing:
            shutil.rmtree(d)

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

    # output names must be unique: two source datasets with the same file stem would overwrite each other
    out_stems: Dict[Path, str] = {}
    used: Dict[str, Path] = {}
    renamed = 0
    for _, ip, _ in paired:
        stem = ip.stem
        if stem in used and used[stem] != ip:
            import hashlib

            stem = f"{ip.stem}__{hashlib.sha1(str(ip).encode()).hexdigest()[:8]}"
            renamed += 1
        used[stem] = ip
        out_stems[ip] = stem
    if renamed:
        LOGGER.warning("%d images shared a file stem with another image and were renamed", renamed)

    have_split = any(s is not None for _, _, s in paired)
    if not have_split:
        val_set = assign_val([ip.stem for _, ip, _ in paired], val_fraction, seed, group_regex)
        if val_fraction > 0 and not group_regex:
            LOGGER.warning(
                "random per-image split: consecutive frames of one clip can land in both train and val "
                "(leakage). Pass --group_regex to split by clip id."
            )
        paired = [
            (jp, ip, "val" if i in val_set else "train") for i, (jp, ip, _) in enumerate(paired)
        ]
    else:
        n_unknown = sum(1 for _, _, s in paired if s is None)
        if n_unknown:
            LOGGER.warning("%d files outside train/val folders default to train", n_unknown)
        paired = [(jp, ip, s or "train") for jp, ip, s in paired]
    has_val = any(sp == "val" for _, _, sp in paired)

    stats = ConvertStats()
    stats.renamed_stems = renamed
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
                lane_thickness,
                partial_annotation,
                out_stem=out_stems[ip],
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
        "val": "images/val" if has_val else "images/train",  # no val images -> validate on train (warned)
        "nc": len(det_classes),
        "names": det_classes,
        "da_classes": len(DA_NAMES),
        "da_names": list(DA_NAMES),
        "ll_classes": len(ll_names),
        "ll_names": ll_names,
    }
    with (dst / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    if not has_val:
        LOGGER.warning("no val images: data.yaml 'val' points at images/train (metrics will be optimistic)")
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
    if stats.skipped_geometry:
        LOGGER.warning("objects with a geometry this converter cannot draw were skipped: %s", dict(stats.skipped_geometry))
    if stats.size_mismatch:
        LOGGER.warning("%d annotations had a `size` different from their image file: coordinates were scaled", stats.size_mismatch)
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
    p.add_argument("--val_fraction", type=float, default=0.1, help="0 disables the val split")
    p.add_argument(
        "--group_regex", type=str, default=None,
        help="split whole clips: regex whose 1st group is the clip id, e.g. '^([0-9a-f]{8})-' for BDD names",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (faster on large datasets, requires JPG sources)",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N annotations (debug)")
    p.add_argument("--lane_thickness", type=float, default=8.0, help="lane line width in px at 720p (scaled with height)")
    p.add_argument("--overwrite", action="store_true", help="clear an existing converted dataset in --dst first")
    p.add_argument(
        "--partial_annotation", action="store_true",
        help="do not write a DA/LL mask for images without any object of that task (trainer ignores the task)",
    )
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
        lane_thickness=args.lane_thickness,
        partial_annotation=args.partial_annotation,
        group_regex=args.group_regex,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
