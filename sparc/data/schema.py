"""One training sample, whatever the source dataset (BDD100K, CULane, MTSD, nuScenes front camera, own drives).

Every converter writes the same thing: one ``<stem>.npz`` per frame holding the arrays plus a JSON header (paths,
calibration, provenance and the :class:`LabelMask`). Image, depth and mask files stay next to it as ordinary
image files, referenced by path relative to the ``.npz``.

The label mask is the core of multi-dataset training: a loss term is computed only on samples whose source labels
that term (``label_mask``), and the 2D classification loss only over the classes the source annotates
(``cls_vocab``, Null-BCE); a missing label is never treated as a negative.

Conventions
-----------
* Pixel coordinates are in the **native** image (the converter's output resolution), x right, y down.
* ``K`` is the 3x3 pinhole matrix of ``image``. For stereo samples both eyes are rectified with the same ``K``
  and ``baseline_m`` is the rectified baseline, so ``depth = K[0,0] * baseline_m / disparity``.
* Camera frame: x right, y down, z forward. Ego frame: x forward, y left, z up, origin on the ground below the
  camera (so ``cam2ego`` carries the mounting height, pitch and roll).
* ``box3d`` rows: ``(depth z_cam, yaw, length, width, height, x_ego, y_ego)``, metres and radians; yaw is the
  heading in the ego frame (0 = same direction as ego, counter-clockwise positive). NaN rows = no 3D label.
* Depth files: uint16 PNG in millimetres, 0 = invalid.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 1

DET_CLASSES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light",
               "traffic sign")  # BDD100K detection classes
CLS_TRAFFIC_LIGHT = DET_CLASSES.index("traffic light")
CLS_TRAFFIC_SIGN = DET_CLASSES.index("traffic sign")
TL_COLORS = ("none", "red", "yellow", "green")  # BDD100K trafficLightColor ("NA" -> none)

LANE_ORDER_RANGE = 3  # order relative to ego: -3..-1 left of ego lane, 1..3 right; 0 is unused (a marking is never "the ego lane")
LANE_DIRECTIONS = ("same", "opposing", "boundary")  # travel direction of the lane on the far side of the marking
LANE_CONTINUITY = ("solid", "dashed")
LANE_TYPES = ("single_white", "single_yellow", "double_white", "double_yellow", "curb", "other")
DA_CLASSES = ("background", "direct", "alternative")
BEV_HW = (240, 160)  # 48 m forward x 32 m lateral at 0.2 m
BEV_RES_M = 0.2
LICENCES = ("bsd", "research", "own")
DEPTH_SOURCES = ("none", "stereo", "lidar", "pseudo")

LABEL_KEYS = (
    "det",          # 2D boxes + class over cls_vocab
    "tl",           # traffic-light colour on traffic-light boxes
    "sign",         # fine sign class (MTSD) on traffic-sign boxes
    "box3d",        # 3D state on (some) boxes
    "track",        # track ids consistent within clip_id
    "depth",        # dense depth file
    "da",           # drivable-area mask
    "lane_geom",    # lane polylines (geometry + existence)
    "lane_order", "lane_dir", "lane_cont", "lane_type",  # lane attributes
    "lane_mask",    # rasterised lane-marking mask
    "bev_da", "bev_lane", "bev_agent",  # BEV layers
)


@dataclass
class LabelMask:
    """Which labels the source provides for this frame. Absent == unknown, never "negative"."""

    det: bool = False
    tl: bool = False
    sign: bool = False
    box3d: bool = False
    track: bool = False
    depth: bool = False
    da: bool = False
    lane_geom: bool = False
    lane_order: bool = False
    lane_dir: bool = False
    lane_cont: bool = False
    lane_type: bool = False
    lane_mask: bool = False
    bev_da: bool = False
    bev_lane: bool = False
    bev_agent: bool = False
    # classes the 2D detection labels are exhaustive for (Null-BCE); empty when det is False
    cls_vocab: tuple = ()

    def __post_init__(self) -> None:
        self.cls_vocab = tuple(int(c) for c in self.cls_vocab)
        bad = [c for c in self.cls_vocab if not 0 <= c < len(DET_CLASSES)]
        if bad:
            raise ValueError(f"cls_vocab has ids outside 0..{len(DET_CLASSES) - 1}: {bad}")
        if self.det and not self.cls_vocab:
            raise ValueError("det=True needs a non-empty cls_vocab (the classes this source annotates exhaustively)")
        if not self.det and self.cls_vocab:
            raise ValueError("cls_vocab given but det=False")

    def vocab_mask(self) -> np.ndarray:
        m = np.zeros(len(DET_CLASSES), dtype=bool)
        m[list(self.cls_vocab)] = True
        return m

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["cls_vocab"] = list(self.cls_vocab)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "LabelMask":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown label keys {sorted(unknown)}")
        return cls(**d)


@dataclass
class LaneInstance:
    """One lane marking (or curb) as a polyline in native pixels, ordered bottom (near) to top (far).

    Attributes are -1 when unlabelled. ``weak`` marks attributes derived by rules (lane_rules.py) rather than by a
    person; their loss weight is reduced and they are never used for reported accuracy."""

    points: np.ndarray  # (M, 2) float32, x, y
    order: int = -99    # -3..3 without 0; -99 = unlabelled
    direction: int = -1
    continuity: int = -1
    ltype: int = -1
    weak: bool = False

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)
        if len(self.points) < 2:
            raise ValueError("a lane needs at least 2 points")
        if not np.isfinite(self.points).all():
            raise ValueError("lane points must be finite")
        if self.order != -99 and (self.order == 0 or abs(self.order) > LANE_ORDER_RANGE):
            raise ValueError(f"lane order {self.order} outside -{LANE_ORDER_RANGE}..{LANE_ORDER_RANGE} (0 excluded)")
        for name, n in (("direction", len(LANE_DIRECTIONS)), ("continuity", len(LANE_CONTINUITY)), ("ltype", len(LANE_TYPES))):
            v = getattr(self, name)
            if not -1 <= v < n:
                raise ValueError(f"lane {name}={v} outside -1..{n - 1}")


def _f32(a, shape_tail: Sequence[int]) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).reshape(-1, *shape_tail)


@dataclass
class Sample:
    image: str
    hw: tuple                              # (H, W) of image (and of depth / masks)
    K: np.ndarray                          # (3, 3)
    cam2ego: np.ndarray                    # (4, 4)
    source: str
    licence: str
    label_mask: LabelMask
    image_right: Optional[str] = None      # rectified right eye (same K), stereo sources only
    baseline_m: Optional[float] = None
    depth: Optional[str] = None
    depth_source: str = "none"
    boxes2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    cls: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    tl_color: Optional[np.ndarray] = None  # (N,) -1 = unlabelled
    sign_cls: Optional[np.ndarray] = None  # (N,) -1 = unlabelled
    box3d: Optional[np.ndarray] = None     # (N, 7), NaN rows = unlabelled
    track_id: Optional[np.ndarray] = None  # (N,) -1 = unlabelled
    lanes: List[LaneInstance] = field(default_factory=list)
    da_mask: Optional[str] = None
    lane_mask: Optional[str] = None
    bev: Optional[str] = None              # uint8 (C, 240, 160) .npy: drivable class, lane line, heading bin, agent
    clip_id: Optional[str] = None
    frame_idx: int = 0
    timestamp_ns: Optional[int] = None
    world_from_ego: Optional[np.ndarray] = None  # (4, 4) ego pose (GPS/IMU or dataset odometry)

    def __post_init__(self) -> None:
        n = len(np.asarray(self.boxes2d).reshape(-1, 4))
        self.boxes2d = _f32(self.boxes2d, (4,))
        self.cls = np.asarray(self.cls, dtype=np.int64).reshape(-1)
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.cam2ego = np.asarray(self.cam2ego, dtype=np.float64).reshape(4, 4)
        self.hw = (int(self.hw[0]), int(self.hw[1]))

        def per_box(a, fill, dtype, tail=()):
            if a is None:
                return np.full((n, *tail), fill, dtype=dtype)
            return np.asarray(a, dtype=dtype).reshape(-1, *tail)

        self.tl_color = per_box(self.tl_color, -1, np.int64)
        self.sign_cls = per_box(self.sign_cls, -1, np.int64)
        self.box3d = per_box(self.box3d, np.nan, np.float32, (7,))
        self.track_id = per_box(self.track_id, -1, np.int64)
        if self.world_from_ego is not None:
            self.world_from_ego = np.asarray(self.world_from_ego, dtype=np.float64).reshape(4, 4)
        self.validate()

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        n = len(self.boxes2d)
        for name in ("cls", "tl_color", "sign_cls", "box3d", "track_id"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, boxes2d has {n}")
        if self.licence not in LICENCES:
            raise ValueError(f"licence {self.licence!r} not in {LICENCES}")
        if self.depth_source not in DEPTH_SOURCES:
            raise ValueError(f"depth_source {self.depth_source!r} not in {DEPTH_SOURCES}")
        if not np.isfinite(self.boxes2d).all():
            raise ValueError("boxes2d must be finite")
        if n and ((self.boxes2d[:, 2] <= self.boxes2d[:, 0]) | (self.boxes2d[:, 3] <= self.boxes2d[:, 1])).any():
            raise ValueError("boxes2d must be xyxy with x2 > x1 and y2 > y1")
        if n and ((self.cls < 0) | (self.cls >= len(DET_CLASSES))).any():
            raise ValueError(f"cls ids outside 0..{len(DET_CLASSES) - 1}")
        lm = self.label_mask
        if n and lm.det and not np.isin(self.cls, lm.cls_vocab).all():
            raise ValueError("a box has a class outside the source's cls_vocab")
        if (self.tl_color >= 0).any():
            if (self.cls[self.tl_color >= 0] != CLS_TRAFFIC_LIGHT).any():
                raise ValueError("tl_color set on a box that is not a traffic light")
            if (self.tl_color >= len(TL_COLORS)).any():
                raise ValueError("tl_color id out of range")
        if (self.sign_cls >= 0).any() and (self.cls[self.sign_cls >= 0] != CLS_TRAFFIC_SIGN).any():
            raise ValueError("sign_cls set on a box that is not a traffic sign")
        if self.K[2, 2] != 1 or self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError(f"K is not a pinhole matrix: {self.K.tolist()}")
        if not np.allclose(self.cam2ego[3], (0, 0, 0, 1)):
            raise ValueError("cam2ego last row must be 0 0 0 1")
        if (self.image_right is None) != (self.baseline_m is None):
            raise ValueError("image_right and baseline_m go together (rectified stereo pair)")
        if self.baseline_m is not None and not self.baseline_m > 0:
            raise ValueError("baseline_m must be > 0")
        need = {"depth": self.depth, "da": self.da_mask, "lane_mask": self.lane_mask}
        for key, path in need.items():
            if getattr(lm, key) and path is None:
                raise ValueError(f"label_mask.{key} is set but the {key} file is missing")
        if lm.depth and self.depth_source == "none":
            raise ValueError("label_mask.depth needs a depth_source")
        if (lm.bev_da or lm.bev_lane or lm.bev_agent) and self.bev is None:
            raise ValueError("a BEV label is set but bev is missing")
        if lm.box3d and not np.isfinite(self.box3d).all(axis=1).any():
            raise ValueError("label_mask.box3d set but no box has a 3D label")
        if lm.track and self.clip_id is None:
            raise ValueError("track labels need a clip_id")
        if (lm.lane_geom or lm.lane_order or lm.lane_dir) and not lm.lane_geom:
            raise ValueError("lane attributes need lane_geom")

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        lanes = self.lanes
        lane_pts = np.concatenate([l.points for l in lanes]) if lanes else np.zeros((0, 2), np.float32)
        lane_len = np.array([len(l.points) for l in lanes], dtype=np.int64)
        lane_attr = np.array([[l.order, l.direction, l.continuity, l.ltype, int(l.weak)] for l in lanes],
                             dtype=np.int64).reshape(-1, 5)
        meta = {
            "schema": SCHEMA_VERSION, "image": self.image, "hw": list(self.hw), "source": self.source,
            "licence": self.licence, "label_mask": self.label_mask.to_dict(), "image_right": self.image_right,
            "baseline_m": self.baseline_m, "depth": self.depth, "depth_source": self.depth_source,
            "da_mask": self.da_mask, "lane_mask": self.lane_mask, "bev": self.bev, "clip_id": self.clip_id,
            "frame_idx": int(self.frame_idx), "timestamp_ns": self.timestamp_ns,
        }
        arrays = dict(K=self.K, cam2ego=self.cam2ego, boxes2d=self.boxes2d, cls=self.cls, tl_color=self.tl_color,
                      sign_cls=self.sign_cls, box3d=self.box3d, track_id=self.track_id, lane_pts=lane_pts,
                      lane_len=lane_len, lane_attr=lane_attr)
        if self.world_from_ego is not None:
            arrays["world_from_ego"] = self.world_from_ego
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Sample":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.pop("schema") != SCHEMA_VERSION:
                raise ValueError(f"{path}: schema version mismatch")
            a = {k: z[k] for k in z.files if k != "meta"}
        lanes, start = [], 0
        for ln, attr in zip(a.pop("lane_len"), a.pop("lane_attr")):
            pts = a["lane_pts"][start:start + ln]
            start += ln
            lanes.append(LaneInstance(pts, order=int(attr[0]), direction=int(attr[1]), continuity=int(attr[2]),
                                      ltype=int(attr[3]), weak=bool(attr[4])))
        a.pop("lane_pts")
        meta["label_mask"] = LabelMask.from_dict(meta["label_mask"])
        meta["hw"] = tuple(meta["hw"])
        return cls(**meta, **a, lanes=lanes)

    def resolve(self, rel: Optional[str], root: str | Path) -> Optional[Path]:
        """A path stored in the sample, resolved against the directory of its ``.npz``."""
        return None if rel is None else Path(root) / rel


def ground_cam2ego(height_m: float, pitch_rad: float = 0.0, roll_rad: float = 0.0) -> np.ndarray:
    """Camera-to-ego transform for a forward camera ``height_m`` above the ground.

    ``pitch_rad`` > 0 tilts the camera down; ``roll_rad`` > 0 rotates it about the optical axis. Used for
    2D-only sources (no calibration), where a nominal mount is all there is."""
    # camera axes (x right, y down, z forward) expressed in ego axes (x forward, y left, z up) for a level camera
    base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    cp, sp, cr, sr = np.cos(pitch_rad), np.sin(pitch_rad), np.cos(roll_rad), np.sin(roll_rad)
    pitch = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])   # about camera x: optical axis -> (0, sin, cos) = down
    roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])    # about camera z
    T = np.eye(4)
    T[:3, :3] = base @ pitch @ roll
    T[:3, 3] = (0.0, 0.0, height_m)
    return T
