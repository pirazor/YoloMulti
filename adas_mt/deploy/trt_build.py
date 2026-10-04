"""Build a TensorRT engine from the exported ONNX **on the target device** (engines are not portable across
TensorRT versions or GPU architectures).

Works with TensorRT 8.6 (JetPack 6.0/6.1) and 10.x (JetPack 6.2: 10.3, JetPack 7.2: 10.16). Only API that exists in
both is used: ``set_memory_pool_limit``, ``parse_from_file``, ``build_serialized_network`` and, for INT8,
``IInt8EntropyCalibrator2``. The one difference handled here is the network-creation flag (``EXPLICIT_BATCH`` is
required before 10 and the default after).

    python -m adas_mt trt-build --onnx best.onnx --precision fp16
    python -m adas_mt trt-build --onnx best.onnx --precision int8 --calib data/dataset.yaml
"""

from __future__ import annotations

import json
import logging
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from .meta import meta_path, read_meta

LOG = logging.getLogger("adas_mt.trt_build")

PRECISIONS = ("fp32", "fp16", "int8")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# which components stay in FP16 when building INT8 (their ONNX node-name substrings come from the export metadata)
KEEP_FP16_PRESETS = {"none": (), "seg": ("da", "ll"), "heads": ("det", "da", "ll")}


class TrtBuildError(RuntimeError):
    pass


def _major(trt) -> int:
    m = re.match(r"(\d+)", str(getattr(trt, "__version__", "0")))
    return int(m.group(1)) if m else 0


def _import_trt():
    try:
        import tensorrt as trt
    except ImportError as e:  # pragma: no cover - exercised on the device
        raise TrtBuildError(
            "the `tensorrt` python package is not installed. On a Jetson it ships with JetPack "
            "(`sudo apt install python3-libnvinfer-dev`, then make it visible to your venv: --system-site-packages)."
        ) from e
    return trt


# --------------------------------------------------------------------------- calibration data
def calibration_images(source: str | Path, n: int = 512, seed: int = 0) -> List[Path]:
    """``n`` images for INT8 calibration from a directory or a ``data.yaml`` (its train split).

    Picks evenly across the sorted list (video frames are strongly correlated, so a random block would be one
    scene) and then shuffles. Use training images, covering the conditions you deploy in (night, rain, glare)."""
    import yaml

    src = Path(source)
    if src.suffix in {".yaml", ".yml"}:
        d = yaml.safe_load(src.read_text(encoding="utf-8"))
        root = Path(d.get("path", src.parent))
        root = root if root.is_absolute() else (src.parent / root)
        train = d["train"]
        folders = [root / t for t in (train if isinstance(train, list) else [train])]
    else:
        folders = [src]
    files = sorted(p for f in folders for p in f.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise TrtBuildError(f"no calibration images found in {[str(f) for f in folders]}")
    if len(files) > n:
        idx = np.linspace(0, len(files) - 1, n).round().astype(int)
        files = [files[i] for i in idx]
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    return files


def make_calibrator(trt, images: Sequence[Path], net_hw: Sequence[int], batch: int, cache_file: Path | None,
                    device: str = "cuda"):
    """``IInt8EntropyCalibrator2`` that feeds letterboxed frames (the deployment preprocessing) from torch buffers."""
    import cv2
    import torch

    from .preprocess import preprocess

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.images, self.i, self.batch, self.buf = list(images), 0, int(batch), None

        def get_batch_size(self) -> int:
            return self.batch

        def get_batch(self, names):  # noqa: ARG002
            if self.i + self.batch > len(self.images):
                return None
            arrs = []
            for p in self.images[self.i : self.i + self.batch]:
                img = cv2.imread(str(p))
                if img is None:
                    raise TrtBuildError(f"cannot read calibration image {p}")
                arrs.append(preprocess(img, net_hw)[0])
            self.i += self.batch
            self.buf = torch.from_numpy(np.stack(arrs)).to(device).contiguous()  # keep alive until the next call
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            if cache_file is not None and Path(cache_file).is_file():
                LOG.info("using calibration cache %s", cache_file)
                return Path(cache_file).read_bytes()
            return None

        def write_calibration_cache(self, cache) -> None:
            if cache_file is not None:
                Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
                Path(cache_file).write_bytes(bytes(cache))

    return Calibrator()


# --------------------------------------------------------------------------- build
@dataclass
class BuildResult:
    engine: Path
    meta_file: Path
    seconds: float
    info: dict = field(default_factory=dict)


def _input_shape(network) -> tuple[str, tuple]:
    t = network.get_input(0)
    return t.name, tuple(t.shape)


def _pin_fp16(trt, network, config, substrings: Sequence[str]) -> int:
    """Force the floating-point layers whose name contains one of ``substrings`` to FP16.

    Layers with an integer output (TopK indices, Shape, Gather, Cast ...) and constants are left alone: forcing
    a float type on them is invalid."""
    float_types = (trt.float32, trt.float16)
    constant = getattr(getattr(trt, "LayerType", None), "CONSTANT", None)
    n = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if not any(s in layer.name for s in substrings) or (constant is not None and layer.type == constant):
            continue
        if not all(layer.get_output(j).dtype in float_types for j in range(layer.num_outputs)):
            continue
        layer.precision = trt.float16
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float16)
        n += 1
    if n:
        flag = getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None) or getattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS")
        config.set_flag(flag)
    return n


def build_engine(
    onnx: str | Path,
    engine: str | Path | None = None,
    precision: str = "fp16",
    workspace_mb: int = 1024,
    calib: str | Path | None = None,
    calib_n: int = 512,
    calib_cache: str | Path | None = None,
    keep_fp16: str | Sequence[str] = "heads",
    timing_cache: str | Path | None = None,
    opt_batch: int | None = None,
    max_batch: int | None = None,
    optimization_level: int | None = None,
    verbose: bool = False,
    trt_module: Any = None,
    calib_device: str = "cuda",
) -> BuildResult:
    """Build ``engine`` (default: next to the ONNX) and write its metadata (``<engine>.json``).

    ``keep_fp16``: with ``precision='int8'``, components kept in FP16 for accuracy: ``heads`` (detection + both
    segmentation heads), ``seg`` or ``none``; or an explicit list of ONNX node-name substrings.
    ``opt_batch`` / ``max_batch`` only apply to an ONNX exported with ``--dynamic`` (default 1 / 4)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    onnx = Path(onnx)
    if not onnx.is_file():
        raise FileNotFoundError(onnx)
    engine = Path(engine) if engine else onnx.with_suffix(".engine")
    meta = read_meta(onnx)
    trt = trt_module or _import_trt()
    major = _major(trt)
    if meta.get("seg_dtype") == "uint8" and major < 10:
        raise TrtBuildError("this ONNX has uint8 segmentation outputs, which need TensorRT >= 10 "
                            f"(found {trt.__version__}); export with --seg-dtype int32")
    if precision == "int8" and not calib:
        raise TrtBuildError("--precision int8 needs --calib <image dir | data.yaml> (calibration images)")

    t0 = time.time()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0 if major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise TrtBuildError(f"TensorRT could not parse {onnx}:\n  " + "\n  ".join(errs)
                            + "\nIf it names DepthToSpace, re-export with --decompose-pixel-shuffle.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) << 20)
    if optimization_level is not None and hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(optimization_level)

    if precision in ("fp16", "int8"):
        if not builder.platform_has_fast_fp16:
            LOG.warning("this GPU reports no fast FP16; the engine will still build")
        config.set_flag(trt.BuilderFlag.FP16)  # int8 builds fall back to FP16, not FP32, for unsupported layers

    # dynamic batch -> an optimisation profile (static ONNX needs none)
    in_name, in_shape = _input_shape(network)
    cal_batch = int(in_shape[0]) if in_shape[0] > 0 else int(opt_batch or 1)
    profile, op, mx = None, None, None
    if in_shape[0] < 0:
        h, w = int(in_shape[2]), int(in_shape[3])
        mn, op = 1, int(opt_batch or 1)
        mx = int(max_batch or max(4, op))
        profile = builder.create_optimization_profile()
        profile.set_shape(in_name, (mn, 3, h, w), (op, 3, h, w), (mx, 3, h, w))
        config.add_optimization_profile(profile)

    pinned = 0
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        net_hw = tuple(meta["imgsz"])
        images = calibration_images(calib, calib_n)
        if len(images) < cal_batch:
            raise TrtBuildError(f"only {len(images)} calibration images for a calibration batch of {cal_batch}")
        # a calibration cache is only reused when asked for: scales are keyed by tensor name, so a stale one
        # (same name, different weights) would silently give a wrong INT8 engine
        calibrator = make_calibrator(trt, images, net_hw, cal_batch, Path(calib_cache) if calib_cache else None, calib_device)
        if profile is not None and hasattr(config, "set_calibration_profile"):
            config.set_calibration_profile(profile)
        config.int8_calibrator = calibrator
        subs = KEEP_FP16_PRESETS[keep_fp16] if isinstance(keep_fp16, str) and keep_fp16 in KEEP_FP16_PRESETS else keep_fp16
        if isinstance(subs, str):
            subs = [subs]
        subs = [meta.get("layers", {}).get(s, s) for s in subs]  # component alias -> node-name substring
        pinned = _pin_fp16(trt, network, config, subs) if subs else 0
        LOG.info("INT8: %d calibration images, %d layers pinned to FP16 (%s)", len(images), pinned, list(subs))

    if timing_cache:
        blob = Path(timing_cache).read_bytes() if Path(timing_cache).is_file() else b""
        if not config.set_timing_cache(config.create_timing_cache(blob), False):
            LOG.warning("timing cache %s does not match this TensorRT/GPU; ignoring it", timing_cache)

    LOG.info("building %s engine from %s with TensorRT %s (this takes minutes on a Jetson)...", precision, onnx, trt.__version__)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise TrtBuildError("TensorRT returned no engine; re-run with --verbose to see the failing layer")
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(bytes(serialized))
    if timing_cache:
        Path(timing_cache).write_bytes(bytes(config.get_timing_cache().serialize()))

    info = {
        "precision": precision, "tensorrt": str(trt.__version__), "workspace_mb": int(workspace_mb), "onnx": onnx.name,
        "int8_fp16_layers": pinned, "dynamic_batch": profile is not None,
        "opt_batch": op if profile is not None else None, "max_batch": mx if profile is not None else None,
        "host": platform.node(), "machine": platform.machine(), "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_file = meta_path(engine)
    meta_file.write_text(json.dumps({**meta, "engine": info}, indent=2), encoding="utf-8")
    secs = time.time() - t0
    LOG.info("wrote %s (%.1f MB) in %.0f s", engine, engine.stat().st_size / 1e6, secs)
    return BuildResult(engine, meta_file, secs, info)


def trtexec_command(onnx: str | Path, engine: str | Path, precision: str = "fp16", workspace_mb: int = 1024) -> str:
    """The equivalent ``trtexec`` line (for reproducing a build or timing an engine without the python API)."""
    flag = {"fp32": "", "fp16": " --fp16", "int8": " --fp16 --int8"}[precision]
    return (f"trtexec --onnx={onnx} --saveEngine={engine}{flag} --memPoolSize=workspace:{workspace_mb}M "
            f"--useCudaGraph --noDataTransfers --warmUp=500 --iterations=1000 --avgRuns=100")
