"""Multi-task settings that are NOT valid Ultralytics arguments.

Ultralytics' ``get_cfg`` rejects unknown keys, and under DDP the trainer is re-created in every worker from
``vars(args)``, so these settings cannot travel through ``args``. They live in a small YAML (``mt.yaml``,
copied into the run directory) that workers find through the ``ADAS_MT_CFG`` environment variable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

ENV_VAR = "ADAS_MT_CFG"


@dataclass
class DistillCfg:
    enabled: bool = False
    teacher: str = "dinov3_b"  # alias from adas_mt.distill.ALIASES, or any timm model name
    teacher_pretrained: bool = True  # False -> random init (tests only)
    teacher_ckpt: Optional[str] = None  # local weights when the hub is unreachable
    teacher_dtype: str = "auto"  # auto (bf16 on native-bf16 GPUs, else fp16) | float32 | bfloat16 | float16
    teacher_scale: float = 1.0
    weight: float = 1.0
    weight_end: float = 0.1
    w_cos: float = 1.0
    w_aff: float = 1.0
    every: int = 1
    n_aff_tokens: int = 256


@dataclass
class MTConfig:
    imgsz: Sequence[int] = (384, 640)  # (h, w); both multiples of 32
    scale: str = "s"  # yolo26 scale when `model` is a yaml without one
    loss_gains: Dict[str, float] = field(default_factory=lambda: {"da": 1.0, "ll": 1.0})
    fitness: Dict[str, float] = field(default_factory=lambda: {"det": 0.5, "da": 0.25, "ll": 0.25})
    head_lr_mult: float = 3.0  # LR multiplier for the freshly initialised da_head / ll_head / kd_proj
    distill: DistillCfg = field(default_factory=DistillCfg)

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MTConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown mt config keys: {sorted(unknown)} (valid: {sorted(known)})")
        dist = d.pop("distill", None) or {}
        if not isinstance(dist, dict):
            raise ValueError(f"mt.distill must be a mapping (enabled: true, teacher: ...), got {dist!r}")
        bad = set(dist) - {f.name for f in fields(DistillCfg)}
        if bad:
            raise ValueError(f"unknown mt.distill keys: {sorted(bad)}")
        cfg = cls(**d, distill=DistillCfg(**dist))
        try:
            cfg.imgsz = tuple(int(x) for x in cfg.imgsz)
        except TypeError:
            raise ValueError(f"imgsz must be a list [h, w], got {cfg.imgsz!r}") from None
        if len(cfg.imgsz) != 2 or any(x <= 0 or x % 32 for x in cfg.imgsz):
            raise ValueError(f"imgsz must be (h, w), both positive multiples of 32, got {cfg.imgsz}")
        for name, allowed, have in (("loss_gains", {"da", "ll"}, cfg.loss_gains), ("fitness", {"det", "da", "ll"}, cfg.fitness)):
            bad = set(have) - allowed
            if bad:
                raise ValueError(f"unknown mt.{name} keys {sorted(bad)} (valid: {sorted(allowed)})")
        cfg.loss_gains = {"da": 1.0, "ll": 1.0, **cfg.loss_gains}
        cfg.fitness = {"det": 0.5, "da": 0.25, "ll": 0.25, **cfg.fitness}
        if not cfg.head_lr_mult > 0:
            raise ValueError(f"head_lr_mult must be > 0, got {cfg.head_lr_mult}")
        if cfg.distill.teacher_dtype not in {"auto", "float32", "bfloat16", "float16"}:
            raise ValueError(f"mt.distill.teacher_dtype {cfg.distill.teacher_dtype!r} invalid")
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "MTConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["imgsz"] = list(self.imgsz)
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    @staticmethod
    def find_run_cfg(path: "str | Path | bool | None") -> Optional[Path]:
        """``<run>/mt.yaml`` for a checkpoint ``<run>/weights/<x>.pt`` (None for ``resume=True`` or no file)."""
        if not isinstance(path, (str, Path)) or not str(path):
            return None
        near = Path(path).resolve().parent.parent / "mt.yaml"
        return near if near.is_file() else None

    @classmethod
    def resolve(cls, mt: "MTConfig | Dict | str | Path | None" = None, resume: "str | Path | bool | None" = None) -> "MTConfig":
        """Explicit config > ``ADAS_MT_CFG`` (DDP workers) > ``mt.yaml`` of the run being resumed > defaults."""
        if isinstance(mt, MTConfig):
            return mt
        if isinstance(mt, dict):
            return cls.from_dict(mt)
        if mt:
            return cls.load(mt)
        env = os.environ.get(ENV_VAR)
        if env:
            if not Path(env).is_file():
                raise FileNotFoundError(f"{ENV_VAR}={env} does not exist (stale environment variable?)")
            return cls.load(env)
        near = cls.find_run_cfg(resume)
        return cls.load(near) if near else cls()


@contextmanager
def env_cfg(path: str | Path):
    """Expose ``path`` to DDP worker processes spawned inside the block, then restore the environment."""
    old = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = old
