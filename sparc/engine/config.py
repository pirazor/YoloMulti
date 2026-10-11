"""Training configuration (``sparc/cfg/*.yaml``) and the data manifest (sources, weights, splits)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _strict(cls, d: Dict[str, Any], where: str):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} (valid: {sorted(names)})")
    return cls(**d)


@dataclass
class ModelCfg:
    backbone: str = "vit_t16_dinov3"
    pretrained: bool = True
    imgsz: List[int] = field(default_factory=lambda: [384, 640])
    hfov: float = 60.0
    decoder_layers: int = 3
    n_agents: int = 200
    n_elements: int = 50
    n_lanes: int = 24
    enc_depth: int = 1
    bev: bool = False
    lane_aux: bool = True
    attn_mode: str = "grid_sample"


@dataclass
class AugYaml:
    scale: List[float] = field(default_factory=lambda: [0.7, 1.3])
    translate: float = 0.1
    hflip: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    height_jitter: float = 0.0


@dataclass
class DistillCfg:
    enabled: bool = False
    teacher: str = "dinov3_b"
    checkpoint: Optional[str] = None
    weight: float = 1.0
    affinity: float = 1.0
    dtype: str = "auto"


@dataclass
class TrainCfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    aug: AugYaml = field(default_factory=AugYaml)
    distill: DistillCfg = field(default_factory=DistillCfg)
    gains: Dict[str, float] = field(default_factory=dict)      # LossGains overrides
    fitness: Dict[str, float] = field(default_factory=lambda: {"agents_mAP50-95": 0.4, "da_mIoU": 0.2, "lane_F1": 0.2,
                                                               "bdd10_mAP50-95": 0.2})
    epochs: int = 30
    batch: int = 16
    workers: int = 8
    lr: float = 2e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    warmup_iters: int = 1000
    lr_min_ratio: float = 0.01
    grad_clip: float = 0.1
    amp: str = "bf16"              # bf16 | fp16 | none
    ema: float = 0.9998
    epoch_size: Optional[int] = None  # items per epoch (default: the sum of the source sizes)
    seed: int = 0
    log_every: int = 50
    val_every: int = 1             # validate every N epochs (0 = only after the last epoch); the last epoch always validates
    val_batch: Optional[int] = None
    min_score: float = 0.01

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainCfg":
        d = dict(d or {})
        out = _strict(cls, {k: v for k, v in d.items() if k not in ("model", "aug", "distill")}, "train cfg")
        out.model = _strict(ModelCfg, d.get("model") or {}, "model")
        out.aug = _strict(AugYaml, d.get("aug") or {}, "aug")
        out.distill = _strict(DistillCfg, d.get("distill") or {}, "distill")
        return out

    @classmethod
    def load(cls, path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> "TrainCfg":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        for k, v in (overrides or {}).items():
            if v is None:
                continue
            if "." in k:
                a, b = k.split(".", 1)
                d.setdefault(a, {})[b] = v
            else:
                d[k] = v
        return cls.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceCfg:
    root: str
    weight: float = 1.0
    split: Optional[str] = None   # defaults: train for training sources, val for validation sources
    limit: Optional[int] = None   # N evenly spaced samples (quick runs, fast in-training validation)


@dataclass
class DataCfg:
    train: List[SourceCfg]
    val: List[SourceCfg]
    signs: Optional[str] = None   # saved vocabulary; default: union of the manifests

    @classmethod
    def load(cls, path: str | Path) -> "DataCfg":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        base = Path(path).parent

        def src(e, split):
            e = dict(e) if isinstance(e, dict) else {"root": str(e)}
            root = Path(e["root"])
            e["root"] = str(root if root.is_absolute() or root.exists() else (base / root))
            e.setdefault("split", split)
            return _strict(SourceCfg, e, "data source")
        return cls(train=[src(e, "train") for e in d.get("train", [])], val=[src(e, "val") for e in d.get("val", [])],
                   signs=d.get("signs"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
