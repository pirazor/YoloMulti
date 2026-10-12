"""Converted sources -> training samples in the virtual camera, mixed across datasets.

:class:`SparcDataset` reads one converted source (``<root>/<split>/samples/*.npz``), warps the image into the
virtual camera with the augmentation composed into the same affine map (scale, translation, horizontal flip; see
``virtual_camera.py``), transforms the labels consistently (boxes, element attributes and sign names under the
flip, packed drivable / lane-marking masks, lane targets lifted to the ego frame at the forward anchors) and returns
tensors plus the per-sample label mask. :class:`MixedDataset` concatenates sources; :class:`WeightedSourceSampler`
draws them by weight (DDP-aware, one pass over each source's items before repeating).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from adas_mt.data.masks import pack_masks
from sparc.data import virtual_camera as vc
from sparc.data.geometry import lane_anchor_targets, project_ego
from sparc.data.schema import (ATTR_FLIP, DA_CLASSES, LANE_MASK_CLASSES, LANE_ORDER_RANGE, LANE_ORDER_UNLABELLED,
                               LANE_X_ANCHORS, LabelMask, Sample, flip_cam2ego)
from sparc.data.vocab import SignVocab

Y_LIMIT = 38.0  # metres; must stay below sparc.nn.decoder.Y_RANGE
MASK_KEYS = ("agents", "box3d", "track", "elements", "el_attr", "el_sign", "topology", "depth", "da", "lane_geom",
             "lane_3d", "lane_order", "lane_dir", "lane_cont", "lane_type", "lane_mask")
N_ORDER = 2 * LANE_ORDER_RANGE


def order_to_class(order: int) -> int:
    """-3..-1, 1..3 -> 0..5; unlabelled -> -1."""
    if order == LANE_ORDER_UNLABELLED or order == 0 or abs(order) > LANE_ORDER_RANGE:
        return -1
    return order + LANE_ORDER_RANGE if order < 0 else order + LANE_ORDER_RANGE - 1


def class_to_order(c: int) -> int:
    return c - LANE_ORDER_RANGE if c < LANE_ORDER_RANGE else c - LANE_ORDER_RANGE + 1


@dataclass
class AugCfg:
    scale: tuple = (0.7, 1.3)
    translate: float = 0.1       # fraction of the image size
    hflip: float = 0.5
    hsv_h: float = 0.015         # keep traffic-light colours
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    height_jitter: float = 0.0   # metres added to the nominal camera height (sources without calibration)


def augment_hsv(img: np.ndarray, rng: random.Random, h: float, s: float, v: float) -> np.ndarray:
    """Ultralytics-style HSV jitter on an RGB uint8 image."""
    if not (h or s or v):
        return img
    r = np.array([rng.uniform(-1, 1) for _ in range(3)]) * [h, s, v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    out = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)


class SparcDataset(Dataset):
    def __init__(self, root: str | Path, split: str, vocab: SignVocab, camera: vc.VirtualCamera = vc.VirtualCamera(),
                 augment: Optional[AugCfg] = None, anchors: np.ndarray = LANE_X_ANCHORS, seed: int = 0,
                 limit: Optional[int] = None):
        self.root, self.split, self.vocab, self.camera, self.aug, self.anchors = Path(root), split, vocab, camera, augment, anchors
        self.sample_dir = self.root / split / "samples"
        self.files = sorted(self.sample_dir.glob("*.npz"))
        if limit is not None and limit < len(self.files):
            # evenly spaced over the sorted files: a subset that covers every clip / city, not just the first ones
            idx = np.linspace(0, len(self.files) - 1, num=max(1, int(limit))).round().astype(int)
            self.files = [self.files[i] for i in np.unique(idx)]
        if not self.files:
            raise FileNotFoundError(f"no samples in {self.sample_dir}")
        import json

        m = self.root / "manifest.json"
        self.source = json.loads(m.read_text())["source"] if m.exists() else self.root.name
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def sample(self, i: int) -> Sample:
        return Sample.load(self.files[i])

    # ------------------------------------------------------------------ geometry of one item
    def _affine(self, s: Sample, rng: random.Random):
        """(A source->output, flipped, K_out, cam2ego_out)."""
        H, W = self.camera.hw
        A = vc.intrinsics_affine(s.K, self.camera.K)
        T = np.array(s.cam2ego, dtype=np.float64)
        flipped = False
        if self.aug is not None:
            sc = rng.uniform(*self.aug.scale)
            tx = rng.uniform(-1, 1) * self.aug.translate * W
            ty = rng.uniform(-1, 1) * self.aug.translate * H
            A = vc.compose(vc.scale_translate(sc, tx, ty, about=((W - 1) / 2, (H - 1) / 2)), A)
            if rng.random() < self.aug.hflip:
                A = vc.compose(vc.hflip(W), A)
                flipped = True
            if self.aug.height_jitter and s.calib == "nominal":
                T[2, 3] += rng.uniform(-1, 1) * self.aug.height_jitter
        K = vc.apply_to_K(A, s.K)
        if flipped:
            K[0, 0], K[0, 1] = -K[0, 0], 0.0  # the mirrored camera keeps a positive focal length and a mirrored cx
            T = flip_cam2ego(T)
        return A, flipped, K, T

    def __getitem__(self, i: int) -> Dict:
        s = self.sample(i)
        # the worker's (or main process's) torch RNG advances across epochs, also with persistent workers; a seed
        # derived from torch.initial_seed() would give every item the same augmentation in every epoch
        rng = random.Random(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()) ^ (self.seed * 1_000_003 + i * 7919))
        H, W = self.camera.hw
        img = cv2.imread(str(self.sample_dir / s.image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.sample_dir / s.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        A, flipped, K, T = self._affine(s, rng)
        T_lift = flip_cam2ego(T) if flipped else T  # the (possibly height-jittered) mount, unmirrored
        out_img = vc.warp_image(img, A, (H, W))
        if self.aug is not None:
            out_img = augment_hsv(out_img, rng, self.aug.hsv_h, self.aug.hsv_s, self.aug.hsv_v)
        lm = s.label_mask
        # road users
        boxes, keep = vc.warp_boxes(s.boxes2d, A, (H, W))
        boxes, cls = boxes[keep], s.cls[keep]
        box3d = s.box3d[keep].copy()
        if flipped and len(box3d):
            box3d[:, 1] = -box3d[:, 1]   # yaw
            box3d[:, 6] = -box3d[:, 6]   # y_ego
        # elements
        el_boxes, ekeep = vc.warp_boxes(s.el_boxes, A, (H, W))
        el_boxes, el_cls, el_attr = el_boxes[ekeep], s.el_cls[ekeep], s.el_attr[ekeep].copy()
        el_names = s.el_sign[ekeep].tolist()
        if flipped:
            el_attr = np.array([ATTR_FLIP.get(int(a), int(a)) for a in el_attr], dtype=np.int64).reshape(-1)
            el_names = [self.vocab.flip(n) for n in el_names]
        el_sign = np.array([self.vocab.id(n) for n in el_names], dtype=np.int64).reshape(-1)
        # lanes: targets in the ego frame (from the unwarped sample), mirrored under the flip
        L, A_n = len(s.lanes), len(self.anchors)
        lane_y, lane_z = np.full((L, A_n), np.nan, np.float32), np.full((L, A_n), np.nan, np.float32)
        lane_vis = np.zeros((L, A_n), dtype=bool)
        lane_attr = np.full((L, 4), -1, dtype=np.int64)  # order class, dir, cont, type
        lane_weak = np.zeros(L, dtype=bool)
        for j, lane in enumerate(s.lanes):
            y, z, vis = lane_anchor_targets(lane, s.K, T_lift, self.anchors)
            if vis.any():  # the lane must also be inside the output image to count as visible there
                xyz = np.stack([self.anchors, np.nan_to_num(y), np.nan_to_num(z)], 1)
                uv, zc = project_ego(xyz, K if not flipped else _unflip_K(K, W), T_lift)
                inside = (zc > 0.5) & np.isfinite(uv).all(1) & (uv[:, 0] > -0.1 * W) & (uv[:, 0] < 1.1 * W) & \
                         (uv[:, 1] > -0.1 * H) & (uv[:, 1] < 1.1 * H)
                vis = vis & inside
            vis = vis & (np.abs(np.nan_to_num(y)) <= Y_LIMIT)  # the lane head's lateral range
            if flipped:
                y = -y
            lane_y[j], lane_z[j], lane_vis[j] = y, z, vis
            order = -lane.order if (flipped and lane.order != LANE_ORDER_UNLABELLED) else lane.order
            lane_attr[j] = (order_to_class(order), lane.direction, lane.continuity, lane.ltype)
            lane_weak[j] = lane.weak
        keep_l = lane_vis.any(1)
        # masks (packed, then warped with nearest; 255 = ignore outside the source image)
        da = _read_mask(self.sample_dir / s.da_mask) if s.da_mask and lm.da else None
        ll = _read_mask(self.sample_dir / s.lane_mask) if s.lane_mask and lm.lane_mask else None
        packed = pack_masks(da, ll, len(DA_CLASSES), len(LANE_MASK_CLASSES), shape=s.hw)
        packed = vc.warp_mask(packed, A, (H, W))
        return {
            "image": torch.from_numpy(np.ascontiguousarray(out_img.transpose(2, 0, 1))),  # uint8 (3, H, W)
            "K": torch.tensor(K, dtype=torch.float32), "cam2ego": torch.tensor(T, dtype=torch.float32),
            "mask": torch.from_numpy(packed),
            "boxes": torch.tensor(_cxcywh_norm(boxes, W, H), dtype=torch.float32), "cls": torch.tensor(cls, dtype=torch.long),
            "box3d": torch.tensor(box3d, dtype=torch.float32), "track_id": torch.tensor(s.track_id[keep], dtype=torch.long),
            "el_boxes": torch.tensor(_cxcywh_norm(el_boxes, W, H), dtype=torch.float32),
            "el_cls": torch.tensor(el_cls, dtype=torch.long), "el_attr": torch.tensor(el_attr, dtype=torch.long),
            "el_sign": torch.tensor(el_sign, dtype=torch.long),
            "lane_y": torch.tensor(lane_y[keep_l]), "lane_z": torch.tensor(lane_z[keep_l]),
            "lane_vis": torch.tensor(lane_vis[keep_l]), "lane_attr": torch.tensor(lane_attr[keep_l]),
            "lane_weak": torch.tensor(lane_weak[keep_l]),
            "label_mask": {k: bool(getattr(lm, k)) for k in MASK_KEYS},
            "cls_vocab": torch.tensor(lm.vocab_mask()), "el_vocab": torch.tensor(lm.el_vocab_mask()),
            "nominal": s.calib == "nominal", "flipped": flipped,
            "meta": {"source": self.source, "index": i, "path": str(self.files[i]), "A": A, "hw": s.hw,
                     "K_native": s.K, "cam2ego_native": s.cam2ego},
        }


def _unflip_K(K: np.ndarray, W: int) -> np.ndarray:
    K = K.copy()
    K[0, 2] = W - 1 - K[0, 2]
    return K


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        raise ValueError(f"{path}: class-id masks must be single-channel")
    return m.astype(np.uint8)


def _cxcywh_norm(xyxy: np.ndarray, W: int, H: int) -> np.ndarray:
    b = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    cx, cy = (b[:, 0] + b[:, 2]) / 2 / W, (b[:, 1] + b[:, 3]) / 2 / H
    w, h = (b[:, 2] - b[:, 0]) / W, (b[:, 3] - b[:, 1]) / H
    return np.stack([cx, cy, w, h], 1).astype(np.float32)


# --------------------------------------------------------------------------- batching
LIST_KEYS = ("boxes", "cls", "box3d", "track_id", "el_boxes", "el_cls", "el_attr", "el_sign", "lane_y", "lane_z",
             "lane_vis", "lane_attr", "lane_weak")


def collate(items: List[Dict]) -> Dict:
    out = {
        "image": torch.stack([it["image"] for it in items]),
        "K": torch.stack([it["K"] for it in items]), "cam2ego": torch.stack([it["cam2ego"] for it in items]),
        "mask": torch.stack([it["mask"] for it in items]),
        "cls_vocab": torch.stack([it["cls_vocab"] for it in items]), "el_vocab": torch.stack([it["el_vocab"] for it in items]),
        "nominal": torch.tensor([it["nominal"] for it in items]),
        "label_mask": {k: torch.tensor([it["label_mask"][k] for it in items]) for k in MASK_KEYS},
        "meta": [it["meta"] for it in items],
    }
    for k in LIST_KEYS:
        out[k] = [it[k] for it in items]
    return out


class MixedDataset(Dataset):
    def __init__(self, datasets: Sequence[SparcDataset], weights: Optional[Sequence[float]] = None):
        if not datasets:
            raise ValueError("no datasets")
        self.datasets = list(datasets)
        self.weights = [float(w) for w in (weights or [1.0] * len(datasets))]
        if len(self.weights) != len(self.datasets) or any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError("weights must be non-negative, one per dataset, not all zero")
        self.offsets = np.cumsum([0] + [len(d) for d in self.datasets])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, i: int) -> Dict:
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.datasets[k][i - int(self.offsets[k])]

    @property
    def sources(self) -> List[str]:
        return [d.source for d in self.datasets]


class WeightedSourceSampler(Sampler):
    """Each epoch draws ``epoch_size`` items: the source of each item is drawn by weight, the item within the
    source is taken from a per-source shuffled cycle (every item of a source is seen once before any repeats).
    With DDP each rank gets a disjoint, equally long slice."""

    def __init__(self, dataset: MixedDataset, epoch_size: Optional[int] = None, seed: int = 0, rank: int = 0,
                 world_size: int = 1):
        self.ds, self.seed, self.rank, self.world = dataset, seed, rank, world_size
        self.epoch_size = int(epoch_size or len(dataset))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.epoch_size // self.world

    def __iter__(self) -> Iterator[int]:
        g = np.random.default_rng(self.seed + self.epoch)
        p = np.array(self.ds.weights) / sum(self.ds.weights)
        choice = g.choice(len(self.ds.datasets), size=self.epoch_size, p=p)
        cycles = [_Cycle(len(d), g) for d in self.ds.datasets]
        idx = [int(self.ds.offsets[c]) + cycles[c].next() for c in choice]
        per_rank = self.epoch_size // self.world
        return iter(idx[self.rank * per_rank:(self.rank + 1) * per_rank])


class _Cycle:
    def __init__(self, n: int, g):
        self.n, self.g, self.perm, self.pos = n, g, None, n

    def next(self) -> int:
        if self.pos >= self.n:
            self.perm, self.pos = self.g.permutation(self.n), 0
        v = int(self.perm[self.pos])
        self.pos += 1
        return v
