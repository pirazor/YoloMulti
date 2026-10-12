"""One training sample, whatever the source dataset (BDD100K, CULane, OpenLane, MTSD, ZOD, nuScenes, own drives).

Every converter writes the same thing: one ``<stem>.npz`` per frame holding the arrays plus a JSON header (paths,
calibration, provenance and the :class:`LabelMask`). Image, depth and mask files stay next to it as ordinary
image files, referenced by path relative to the ``.npz``.

The label mask is the core of multi-dataset training: a loss term is computed only on samples whose source labels
that term (``label_mask``), and the classification losses only over the classes the source annotates exhaustively
(``cls_vocab`` for road users, ``el_vocab`` for traffic elements; Null-BCE); a missing label is never a negative.

Conventions
-----------
* Pixel coordinates are in the **native** image (the converter's output resolution), x right, y down, pixel
  centres at integer coordinates.
* ``K`` is the 3x3 pinhole matrix of ``image``; ``calib`` says whether it was measured or is a dataset-level guess
  (``nominal``: BDD100K, CULane, ... have no calibration). For stereo samples both eyes are rectified with the same
  ``K`` and ``baseline_m`` is the rectified baseline, so ``depth = K[0,0] * baseline_m / disparity``.
* Camera frame: x right, y down, z forward. Ego frame: x forward, y left, z up, origin on the ground below the
  camera (``cam2ego`` carries the mounting height, pitch and roll; nominal sources get a nominal mount).
* Road users (``boxes2d``/``cls``, 8 classes) and traffic elements (``el_*``: lights and signs) are separate
  tables; the 10-class BDD100K detection benchmark is their union (:data:`BDD_CLASSES`).
* ``box3d`` rows: ``(depth z_cam, yaw, length, width, height, x_ego, y_ego)``, metres and radians; yaw is the
  heading in the ego frame (0 = same direction as ego, counter-clockwise positive). NaN rows = no 3D label.
* Lanes are polylines in native pixels (``points``) and, when the source has them, in the ego frame
  (``points3d``). Order / direction / continuity / type follow the constants below.
* Depth files: uint16 PNG in millimetres, 0 = invalid.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 2

AGENT_CLASSES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
ELEMENT_CLASSES = ("traffic_light", "traffic_sign")
EL_LIGHT, EL_SIGN = 0, 1
BDD_CLASSES = AGENT_CLASSES + ("traffic light", "traffic sign")  # the 10-class BDD100K detection benchmark
# traffic-element attributes (OpenLane-V2): colour of a light, or the meaning of a light arrow / sign
ELEMENT_ATTRS = ("unknown", "red", "green", "yellow", "go_straight", "turn_left", "turn_right", "no_left_turn",
                 "no_right_turn", "u_turn", "no_u_turn", "slight_left", "slight_right")
ATTR_FLIP = {ELEMENT_ATTRS.index(a): ELEMENT_ATTRS.index(b) for a, b in
             (("turn_left", "turn_right"), ("turn_right", "turn_left"), ("no_left_turn", "no_right_turn"),
              ("no_right_turn", "no_left_turn"), ("slight_left", "slight_right"), ("slight_right", "slight_left"))}

LANE_ORDER_RANGE = 3  # order relative to ego: -3..-1 left of the ego lane, 1..3 right; 0 is unused
LANE_ORDER_UNLABELLED = -99
LANE_DIRECTIONS = ("same", "opposing", "boundary")  # travel direction of the lane on the far side of the marking
LANE_CONTINUITY = ("solid", "dashed")
LANE_TYPES = ("single_white", "single_yellow", "double_white", "double_yellow", "curb", "other")
LANE_X_ANCHORS = np.geomspace(3.0, 60.0, 20)  # forward distances (m) at which a lane is predicted
DA_CLASSES = ("background", "direct", "alternative")
LANE_MASK_CLASSES = ("background", "marking")
BEV_HW = (240, 160)  # 48 m forward x 32 m lateral at 0.2 m
BEV_RES_M = 0.2
LICENCES = ("bsd", "research", "own")
DEPTH_SOURCES = ("none", "stereo", "lidar", "pseudo")
CALIBS = ("measured", "nominal")


@dataclass
class LabelMask:
    """Which labels the source provides for this frame. Absent == unknown, never "negative"."""

    agents: bool = False      # road-user boxes + class over cls_vocab
    box3d: bool = False       # 3D state on (some) road users
    track: bool = False       # track ids consistent within clip_id
    elements: bool = False    # traffic-element boxes + class over el_vocab
    el_attr: bool = False     # light colour / arrow or sign meaning
    el_sign: bool = False     # fine sign class (name)
    topology: bool = False    # element -> governed lane
    depth: bool = False       # dense depth file
    da: bool = False          # drivable-area mask
    lane_geom: bool = False   # lane polylines (geometry + existence)
    lane_3d: bool = False     # lane points in the ego frame (not a ground-plane lift)
    lane_order: bool = False
    lane_dir: bool = False
    lane_cont: bool = False
    lane_type: bool = False
    lane_mask: bool = False   # rasterised lane-marking mask
    bev_da: bool = False
    bev_lane: bool = False
    bev_agent: bool = False
    cls_vocab: tuple = ()     # road-user classes the labels are exhaustive for (Null-BCE); empty when agents is False
    el_vocab: tuple = ()      # element classes the labels are exhaustive for

    def __post_init__(self) -> None:
        self.cls_vocab = tuple(int(c) for c in self.cls_vocab)
        self.el_vocab = tuple(int(c) for c in self.el_vocab)
        bad = [c for c in self.cls_vocab if not 0 <= c < len(AGENT_CLASSES)]
        if bad:
            raise ValueError(f"cls_vocab has ids outside 0..{len(AGENT_CLASSES) - 1}: {bad}")
        bad = [c for c in self.el_vocab if not 0 <= c < len(ELEMENT_CLASSES)]
        if bad:
            raise ValueError(f"el_vocab has ids outside 0..{len(ELEMENT_CLASSES) - 1}: {bad}")
        if self.agents != bool(self.cls_vocab):
            raise ValueError("agents=True needs a non-empty cls_vocab (and cls_vocab needs agents=True)")
        if self.elements != bool(self.el_vocab):
            raise ValueError("elements=True needs a non-empty el_vocab (and el_vocab needs elements=True)")
        if (self.box3d or self.track) and not self.agents:
            raise ValueError("box3d / track labels need agents")
        if (self.el_attr or self.el_sign or self.topology) and not self.elements:
            raise ValueError("el_attr / el_sign / topology need elements")
        if (self.lane_3d or self.lane_order or self.lane_dir or self.lane_cont or self.lane_type or
                self.topology) and not self.lane_geom:
            raise ValueError("lane attributes / lane_3d / topology need lane_geom")

    def vocab_mask(self) -> np.ndarray:
        m = np.zeros(len(AGENT_CLASSES), dtype=bool)
        m[list(self.cls_vocab)] = True
        return m

    def el_vocab_mask(self) -> np.ndarray:
        m = np.zeros(len(ELEMENT_CLASSES), dtype=bool)
        m[list(self.el_vocab)] = True
        return m

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["cls_vocab"], d["el_vocab"] = list(self.cls_vocab), list(self.el_vocab)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "LabelMask":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown label keys {sorted(unknown)}")
        return cls(**d)


@dataclass
class LaneInstance:
    """One lane marking (or curb): a polyline in native pixels, ordered near (bottom) to far (top), plus
    optionally the same polyline in the ego frame (``points3d``, metres; sources with 3D lanes).

    Attributes are -1 (order: -99) when unlabelled. ``weak`` marks attributes derived by rules rather than by a
    person; their loss weight is reduced and they are never used for reported accuracy."""

    points: np.ndarray                      # (M, 2) float32, x, y
    points3d: Optional[np.ndarray] = None   # (M, 3) float32 ego x, y, z (same M as points, or any M >= 2)
    order: int = LANE_ORDER_UNLABELLED
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
        if self.points3d is not None:
            self.points3d = np.asarray(self.points3d, dtype=np.float32).reshape(-1, 3)
            if len(self.points3d) < 2 or not np.isfinite(self.points3d).all():
                raise ValueError("points3d needs >= 2 finite points")
        if self.order != LANE_ORDER_UNLABELLED and (self.order == 0 or abs(self.order) > LANE_ORDER_RANGE):
            raise ValueError(f"lane order {self.order} outside -{LANE_ORDER_RANGE}..{LANE_ORDER_RANGE} (0 excluded)")
        for name, n in (("direction", len(LANE_DIRECTIONS)), ("continuity", len(LANE_CONTINUITY)), ("ltype", len(LANE_TYPES))):
            v = getattr(self, name)
            if not -1 <= v < n:
                raise ValueError(f"lane {name}={v} outside -1..{n - 1}")


def _f32(a, shape_tail: Sequence[int]) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).reshape(-1, *shape_tail)


def _names(a, n: int) -> np.ndarray:
    if a is None:
        return np.full((n,), "", dtype="<U1")
    out = np.asarray(list(a), dtype=str).reshape(-1) if not isinstance(a, np.ndarray) else a.astype(str).reshape(-1)
    return out


@dataclass
class Sample:
    image: str
    hw: tuple                              # (H, W) of image (and of depth / masks)
    K: np.ndarray                          # (3, 3)
    cam2ego: np.ndarray                    # (4, 4)
    source: str
    licence: str
    label_mask: LabelMask
    calib: str = "measured"
    image_right: Optional[str] = None      # rectified right eye (same K), stereo sources only
    baseline_m: Optional[float] = None
    depth: Optional[str] = None
    depth_source: str = "none"
    # road users
    boxes2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    cls: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    box3d: Optional[np.ndarray] = None     # (N, 7), NaN rows = unlabelled
    track_id: Optional[np.ndarray] = None  # (N,) -1 = unlabelled
    # traffic elements
    el_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    el_cls: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    el_attr: Optional[np.ndarray] = None   # (M,) -1 = unlabelled, else index into ELEMENT_ATTRS
    el_sign: Optional[np.ndarray] = None   # (M,) fine sign class *name* ("" = unlabelled); ids come from the manifest vocab
    el_lane: Optional[np.ndarray] = None   # (M,) index into lanes of the governed lane, -1 = unlabelled
    # lanes, masks, BEV
    lanes: List[LaneInstance] = field(default_factory=list)
    da_mask: Optional[str] = None
    lane_mask: Optional[str] = None
    bev: Optional[str] = None              # uint8 (C, 240, 160) .npy: drivable class, lane line, heading bin, agent
    # sequence
    clip_id: Optional[str] = None
    frame_idx: int = 0
    timestamp_ns: Optional[int] = None
    world_from_ego: Optional[np.ndarray] = None  # (4, 4) ego pose (GPS/IMU or dataset odometry)

    def __post_init__(self) -> None:
        self.boxes2d = _f32(self.boxes2d, (4,))
        self.cls = np.asarray(self.cls, dtype=np.int64).reshape(-1)
        self.el_boxes = _f32(self.el_boxes, (4,))
        self.el_cls = np.asarray(self.el_cls, dtype=np.int64).reshape(-1)
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.cam2ego = np.asarray(self.cam2ego, dtype=np.float64).reshape(4, 4)
        self.hw = (int(self.hw[0]), int(self.hw[1]))
        n, m = len(self.boxes2d), len(self.el_boxes)

        def per(a, count, fill, dtype, tail=()):
            if a is None:
                return np.full((count, *tail), fill, dtype=dtype)
            return np.asarray(a, dtype=dtype).reshape(-1, *tail)

        self.box3d = per(self.box3d, n, np.nan, np.float32, (7,))
        self.track_id = per(self.track_id, n, -1, np.int64)
        self.el_attr = per(self.el_attr, m, -1, np.int64)
        self.el_lane = per(self.el_lane, m, -1, np.int64)
        self.el_sign = _names(self.el_sign, m)
        if self.world_from_ego is not None:
            self.world_from_ego = np.asarray(self.world_from_ego, dtype=np.float64).reshape(4, 4)
        self.validate()

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        n, m = len(self.boxes2d), len(self.el_boxes)
        for name, want in (("cls", n), ("box3d", n), ("track_id", n), ("el_cls", m), ("el_attr", m), ("el_sign", m), ("el_lane", m)):
            if len(getattr(self, name)) != want:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {want}")
        if self.licence not in LICENCES:
            raise ValueError(f"licence {self.licence!r} not in {LICENCES}")
        if self.depth_source not in DEPTH_SOURCES:
            raise ValueError(f"depth_source {self.depth_source!r} not in {DEPTH_SOURCES}")
        if self.calib not in CALIBS:
            raise ValueError(f"calib {self.calib!r} not in {CALIBS}")
        for name, b in (("boxes2d", self.boxes2d), ("el_boxes", self.el_boxes)):
            if not np.isfinite(b).all():
                raise ValueError(f"{name} must be finite")
            if len(b) and ((b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])).any():
                raise ValueError(f"{name} must be xyxy with x2 > x1 and y2 > y1")
        if n and ((self.cls < 0) | (self.cls >= len(AGENT_CLASSES))).any():
            raise ValueError(f"cls ids outside 0..{len(AGENT_CLASSES) - 1}")
        if m and ((self.el_cls < 0) | (self.el_cls >= len(ELEMENT_CLASSES))).any():
            raise ValueError(f"el_cls ids outside 0..{len(ELEMENT_CLASSES) - 1}")
        lm = self.label_mask
        if n and lm.agents and not np.isin(self.cls, lm.cls_vocab).all():
            raise ValueError("a road-user box has a class outside the source's cls_vocab")
        if m and lm.elements and not np.isin(self.el_cls, lm.el_vocab).all():
            raise ValueError("an element box has a class outside the source's el_vocab")
        if m and ((self.el_attr < -1) | (self.el_attr >= len(ELEMENT_ATTRS))).any():
            raise ValueError("el_attr id out of range")
        if m and (self.el_sign != "").any() and (self.el_cls[self.el_sign != ""] != EL_SIGN).any():
            raise ValueError("a sign name is set on a box that is not a traffic sign")
        if m and ((self.el_lane < -1) | (self.el_lane >= len(self.lanes))).any():
            raise ValueError("el_lane points outside lanes")
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
        finite = np.isfinite(self.box3d)
        if (finite.any(axis=1) & ~finite.all(axis=1)).any():
            raise ValueError("a box3d row must be all finite (labelled) or all NaN (unlabelled)")
        if lm.box3d and not finite.all(axis=1).any():
            raise ValueError("label_mask.box3d set but no box has a 3D label")
        if lm.track and self.clip_id is None:
            raise ValueError("track labels need a clip_id")
        if lm.lane_3d and self.lanes and not all(l.points3d is not None for l in self.lanes):
            raise ValueError("label_mask.lane_3d set but a lane has no points3d")

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        lanes = self.lanes
        lane_pts = np.concatenate([l.points for l in lanes]) if lanes else np.zeros((0, 2), np.float32)
        lane_len = np.array([len(l.points) for l in lanes], dtype=np.int64)
        lane_xyz = np.concatenate([l.points3d for l in lanes if l.points3d is not None]) if any(
            l.points3d is not None for l in lanes) else np.zeros((0, 3), np.float32)
        lane_len3d = np.array([0 if l.points3d is None else len(l.points3d) for l in lanes], dtype=np.int64)
        lane_attr = np.array([[l.order, l.direction, l.continuity, l.ltype, int(l.weak)] for l in lanes],
                             dtype=np.int64).reshape(-1, 5)
        meta = {
            "schema": SCHEMA_VERSION, "image": self.image, "hw": list(self.hw), "source": self.source,
            "licence": self.licence, "label_mask": self.label_mask.to_dict(), "calib": self.calib,
            "image_right": self.image_right, "baseline_m": self.baseline_m, "depth": self.depth,
            "depth_source": self.depth_source, "da_mask": self.da_mask, "lane_mask": self.lane_mask, "bev": self.bev,
            "clip_id": self.clip_id, "frame_idx": int(self.frame_idx), "timestamp_ns": self.timestamp_ns,
        }
        arrays = dict(K=self.K, cam2ego=self.cam2ego, boxes2d=self.boxes2d, cls=self.cls, box3d=self.box3d,
                      track_id=self.track_id, el_boxes=self.el_boxes, el_cls=self.el_cls, el_attr=self.el_attr,
                      el_sign=self.el_sign, el_lane=self.el_lane, lane_pts=lane_pts, lane_len=lane_len,
                      lane_xyz=lane_xyz, lane_len3d=lane_len3d, lane_attr=lane_attr)
        if self.world_from_ego is not None:
            arrays["world_from_ego"] = self.world_from_ego
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Sample":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.pop("schema") != SCHEMA_VERSION:
                raise ValueError(f"{path}: schema version mismatch (file {meta.get('schema')}, code {SCHEMA_VERSION})")
            a = {k: z[k] for k in z.files if k != "meta"}
        lanes, s2, s3 = [], 0, 0
        for ln, ln3, attr in zip(a.pop("lane_len"), a.pop("lane_len3d"), a.pop("lane_attr")):
            pts = a["lane_pts"][s2:s2 + ln]
            s2 += ln
            xyz = a["lane_xyz"][s3:s3 + ln3] if ln3 else None
            s3 += ln3
            lanes.append(LaneInstance(pts, xyz, order=int(attr[0]), direction=int(attr[1]), continuity=int(attr[2]),
                                      ltype=int(attr[3]), weak=bool(attr[4])))
        a.pop("lane_pts")
        a.pop("lane_xyz")
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


def flip_cam2ego(T: np.ndarray) -> np.ndarray:
    """``cam2ego`` of the horizontally mirrored scene (camera x -> -x, ego y -> -y)."""
    M_cam = np.diag([-1.0, 1.0, 1.0, 1.0])
    M_ego = np.diag([1.0, -1.0, 1.0, 1.0])
    return M_ego @ np.asarray(T, dtype=np.float64) @ M_cam
