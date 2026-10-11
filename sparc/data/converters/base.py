"""Shared machinery for the converters: output layout, statistics, sample writing, image linking.

Output layout (one directory per source):

    <dst>/manifest.json                 source, licence, splits, counts, the sign names seen, the converter's stats
    <dst>/<split>/samples/<stem>.npz    one Sample per frame
    <dst>/<split>/images/<stem>.jpg     symlink to the original image (``--copy`` copies instead)
    <dst>/<split>/masks/da/<stem>.png   uint8 class ids (0 bg, 1 direct, 2 alternative)
    <dst>/<split>/masks/ll/<stem>.png   uint8 (0 bg, 1 marking)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparc.data.schema import SCHEMA_VERSION, LabelMask, Sample

LOG = logging.getLogger("sparc.convert")


def image_size(path: str | Path) -> Optional[Tuple[int, int]]:
    """(h, w) from the image header without decoding the pixels; None when unreadable."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.size[1]), int(im.size[0])
    except (OSError, ValueError):
        return None


@dataclass
class ConvertStats:
    source: str
    frames: int = 0
    agents: int = 0
    elements: int = 0
    lanes: int = 0
    da_masks: int = 0
    lane_masks: int = 0
    skipped: Counter = field(default_factory=Counter)   # reason -> count (records, not frames)
    skipped_frames: Counter = field(default_factory=Counter)
    per_split: Counter = field(default_factory=Counter)
    seconds: float = 0.0

    def skip(self, reason: str, n: int = 1) -> None:
        self.skipped[reason] += n

    def skip_frame(self, reason: str) -> None:
        self.skipped_frames[reason] += 1

    def summary(self) -> str:
        lines = [f"{self.source}: {self.frames} frames ({dict(self.per_split)}) in {self.seconds:.0f}s: "
                 f"{self.agents} road users, {self.elements} traffic elements, {self.lanes} lanes, "
                 f"{self.da_masks} drivable masks, {self.lane_masks} lane masks"]
        if self.skipped:
            lines.append("  skipped records: " + ", ".join(f"{k}: {v}" for k, v in self.skipped.most_common()))
        if self.skipped_frames:
            lines.append("  skipped frames: " + ", ".join(f"{k}: {v}" for k, v in self.skipped_frames.most_common()))
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {"frames": self.frames, "agents": self.agents, "elements": self.elements, "lanes": self.lanes,
                "da_masks": self.da_masks, "lane_masks": self.lane_masks, "skipped": dict(self.skipped),
                "skipped_frames": dict(self.skipped_frames), "per_split": dict(self.per_split), "seconds": self.seconds}


class Writer:
    def __init__(self, dst: str | Path, source: str, licence: str, copy_images: bool = False, overwrite: bool = False,
                 limit: Optional[int] = None):
        self.dst, self.source, self.licence = Path(dst), source, licence
        self.copy_images, self.limit = copy_images, limit
        self.stats = ConvertStats(source)
        self.signs: set = set()
        self._t0 = time.time()
        if self.dst.exists() and any(self.dst.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.dst} is not empty (use --overwrite)")
            shutil.rmtree(self.dst)
        self.dst.mkdir(parents=True, exist_ok=True)

    @property
    def full(self) -> bool:
        return self.limit is not None and self.stats.frames >= self.limit

    def _link(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if self.copy_images:
            shutil.copyfile(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)

    def add(self, split: str, stem: str, image: str | Path, label_mask: LabelMask, da_mask: Optional[np.ndarray] = None,
            lane_mask: Optional[np.ndarray] = None, hw: Optional[Tuple[int, int]] = None, **fields) -> Optional[Sample]:
        """Write one frame. ``fields`` are :class:`Sample` fields except image / hw / source / licence / masks.

        Returns the Sample, or None when the image is missing (counted as a skipped frame)."""
        image = Path(image)
        if hw is None:
            hw = image_size(image)
            if hw is None:
                self.stats.skip_frame("image unreadable")
                return None
        split_dir = self.dst / split
        img_dst = split_dir / "images" / (stem + image.suffix.lower())
        self._link(image, img_dst)
        rel_img = os.path.relpath(img_dst, split_dir / "samples")
        da_rel = ll_rel = None
        if da_mask is not None:
            p = split_dir / "masks" / "da" / f"{stem}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), np.asarray(da_mask, dtype=np.uint8))
            da_rel = os.path.relpath(p, split_dir / "samples")
            self.stats.da_masks += 1
        if lane_mask is not None:
            p = split_dir / "masks" / "ll" / f"{stem}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), np.asarray(lane_mask, dtype=np.uint8))
            ll_rel = os.path.relpath(p, split_dir / "samples")
            self.stats.lane_masks += 1
        sample = Sample(image=rel_img, hw=hw, source=self.source, licence=self.licence, label_mask=label_mask,
                        da_mask=da_rel, lane_mask=ll_rel, **fields)
        sample.save(split_dir / "samples" / stem)
        self.stats.frames += 1
        self.stats.per_split[split] += 1
        self.stats.agents += len(sample.boxes2d)
        self.stats.elements += len(sample.el_boxes)
        self.stats.lanes += len(sample.lanes)
        self.signs.update(n for n in sample.el_sign.tolist() if n)
        return sample

    def finish(self, extra: Optional[Dict] = None) -> ConvertStats:
        self.stats.seconds = time.time() - self._t0
        manifest = {"format": "sparc-dataset", "schema": SCHEMA_VERSION, "source": self.source, "licence": self.licence,
                    "splits": dict(self.stats.per_split), "signs": sorted(self.signs), "stats": self.stats.to_dict(),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
        (self.dst / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        LOG.info(self.stats.summary())
        return self.stats


def add_common_args(p, default_licence: str = "research") -> None:
    p.add_argument("--dst", required=True, help="output directory (one per source)")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--limit", type=int, default=None, help="stop after this many frames per split (first-run check)")
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--licence", default=default_licence, choices=("bsd", "research", "own"))


def nominal_camera_args(p, hfov: float, height: float, horizon: Optional[float] = None) -> None:
    """Options for sources without calibration: FOV guess, mount height, horizon row (fraction of H)."""
    p.add_argument("--hfov", type=float, default=hfov, help="horizontal field of view guess (deg)")
    p.add_argument("--cam-height", type=float, default=height, help="nominal camera height above the ground (m)")
    p.add_argument("--horizon", type=float, default=horizon, help="horizon row as a fraction of the image height")


def nominal_camera(args, hw: Sequence[int]):
    from sparc.data.schema import ground_cam2ego
    from sparc.data.virtual_camera import nominal_K

    horizon = None if args.horizon is None else float(args.horizon) * (hw[0] - 1)
    return nominal_K(hw, hfov_deg=args.hfov, horizon_row=horizon), ground_cam2ego(args.cam_height), "nominal"
