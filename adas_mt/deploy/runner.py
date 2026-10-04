"""Inference runner for the exported network: ONNX Runtime (development / parity) and TensorRT (the Jetson).

Depends only on numpy + cv2 (+ onnxruntime or tensorrt and torch for the buffers); it does not import
ultralytics, so a Jetson can run it with just the engine and the sidecar JSON.

    from adas_mt.deploy.runner import Pipeline
    pipe = Pipeline("best.engine", conf=0.25)
    res = pipe(frame_bgr)          # res.boxes / res.scores / res.classes in original pixels,
                                   # res.da / res.ll uint8 class maps at the frame's resolution
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .meta import read_meta
from .preprocess import LetterboxInfo, letterbox_torch, preprocess

LOG = logging.getLogger("adas_mt.runner")

ENGINE_SUFFIXES = {".engine", ".plan", ".trt"}
DA_COLORS = {1: (60, 200, 60), 2: (230, 150, 40)}  # BGR: direct = green, alternative = blue
LL_COLORS = {1: (40, 40, 240), 2: (30, 220, 240)}  # solid = red, dashed = yellow


# --------------------------------------------------------------------------- backends
class OrtBackend:
    """ONNX Runtime: CPU on a dev machine, CUDA where onnxruntime-gpu is installed. Not the production path."""

    def __init__(self, path: str | Path, providers: Optional[Sequence[str]] = None):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        providers = list(providers) if providers else [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(str(path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_names = [o.name for o in self.session.get_outputs()]
        b = inp.shape[0]
        self.batch: Optional[int] = b if isinstance(b, int) else None  # None = dynamic
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        self.providers = self.session.get_providers()

    def infer(self, x) -> Dict[str, np.ndarray]:
        if hasattr(x, "detach"):  # torch tensor
            x = x.detach().cpu().numpy()
        x = np.ascontiguousarray(x, dtype=np.float32)
        return dict(zip(self.output_names, self.session.run(self.output_names, {self.input_name: x})))

    def close(self) -> None:
        self.session = None


_TRT_TO_TORCH = {"FLOAT": "float32", "HALF": "float16", "INT32": "int32", "INT64": "int64", "INT8": "int8", "UINT8": "uint8",
                 "BOOL": "bool", "BF16": "bfloat16"}


class TrtBackend:
    """TensorRT engine executed with the tensor-name API (``set_tensor_address`` + ``execute_async_v3``), which exists
    in TensorRT 8.6 and 10.x. Buffers are torch tensors (the JetPack torch wheel is the usual CUDA allocator there)
    so the stream is shared with any torch pre/post-processing."""

    def __init__(self, path: str | Path, device: str = "cuda", trt_module: Any = None):
        import torch

        trt = trt_module
        if trt is None:
            try:
                import tensorrt as trt  # type: ignore[no-redef]
            except ImportError as e:  # pragma: no cover - device only
                raise RuntimeError("`tensorrt` is not importable; on a Jetson create the venv with --system-site-packages") from e
        self.torch, self.trt, self.device = torch, trt, torch.device(device)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {path}: engines only load on the TensorRT version and GPU they were built on")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode_in = trt.TensorIOMode.INPUT
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == mode_in)
        self.output_names = [n for n in names if self.engine.get_tensor_mode(n) != mode_in]
        shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.dynamic = shape[0] < 0
        if self.dynamic:
            _, _, mx = self.engine.get_tensor_profile_shape(self.input_name, 0)
            self.max_batch = int(mx[0])
            self.batch = None
        else:
            self.max_batch = self.batch = int(shape[0])
        self.input_hw = (int(shape[2]), int(shape[3]))
        self._bufs: Dict[str, Any] = {}
        in_shape = (self.max_batch, 3, *self.input_hw)
        self._bufs[self.input_name] = self._alloc(self.input_name, in_shape)
        if self.dynamic:
            self.context.set_input_shape(self.input_name, in_shape)  # lets the context report output shapes
        for n in self.output_names:
            oshape = tuple(self.context.get_tensor_shape(n)) if self.dynamic else tuple(self.engine.get_tensor_shape(n))
            self._bufs[n] = self._alloc(n, oshape)
        for n, b in self._bufs.items():
            self.context.set_tensor_address(n, int(b.data_ptr()))

    def _alloc(self, name: str, shape: Tuple[int, ...]):
        dt = getattr(self.torch, _TRT_TO_TORCH[self.engine.get_tensor_dtype(name).name])
        return self.torch.empty(tuple(int(s) for s in shape), dtype=dt, device=self.device)

    def infer_device(self, x) -> Dict[str, Any]:
        """Run the engine; returns the outputs as torch tensors on the device (views of the persistent buffers: copy
        them before the next call)."""
        torch = self.torch
        x = torch.as_tensor(x, device=self.device)
        b = int(x.shape[0])
        if b > self.max_batch or (not self.dynamic and b != self.batch):
            raise ValueError(f"engine takes batch {self.batch if not self.dynamic else f'1..{self.max_batch}'}, got {b}")
        inbuf = self._bufs[self.input_name]
        inbuf[:b].copy_(x)
        if self.dynamic:
            self.context.set_input_shape(self.input_name, (b, 3, *self.input_hw))
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            ok = self.context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
        else:  # fake engines in tests
            ok = self.context.execute_async_v3(0)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        return {n: self._bufs[n][:b] for n in self.output_names}

    def infer(self, x) -> Dict[str, np.ndarray]:
        # copy=True: the outputs are views of persistent buffers, so the host arrays must never alias them
        # (a plain .cpu() is a no-op on a CPU device and the next call would overwrite earlier results)
        return {k: v.detach().to("cpu", copy=True).numpy() for k, v in self.infer_device(x).items()}

    def close(self) -> None:
        self._bufs.clear()
        self.context = self.engine = None


def open_backend(model: str | Path, backend: str = "auto", device: str = "cuda", providers=None, trt_module: Any = None):
    suffix = Path(model).suffix.lower()
    if backend == "auto":
        backend = "trt" if suffix in ENGINE_SUFFIXES else "ort"
    if backend == "ort":
        return OrtBackend(model, providers)
    if backend == "trt":
        return TrtBackend(model, device, trt_module)
    raise ValueError(f"backend must be auto|ort|trt, got {backend!r}")


# --------------------------------------------------------------------------- pipeline
@dataclass
class Result:
    boxes: np.ndarray  # (N, 4) x1 y1 x2 y2 in original frame pixels
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) int
    da: np.ndarray  # drivable-area class map, uint8 (frame resolution unless masks_at='network')
    ll: np.ndarray  # lane class map
    info: LetterboxInfo
    timings: Dict[str, float] = field(default_factory=dict)  # ms: pre, infer, post, total

    def __len__(self) -> int:
        return len(self.scores)


class Pipeline:
    """frame (BGR uint8) -> :class:`Result`, with the preprocessing identical to validation."""

    def __init__(self, model: str | Path, backend: str = "auto", conf: float = 0.25, device: str = "cuda",
                 gpu_preprocess: bool = False, masks_at: str = "original", providers=None, trt_module: Any = None):
        if masks_at not in ("original", "network"):
            raise ValueError("masks_at must be 'original' or 'network'")
        self.model = Path(model)
        self.meta = read_meta(self.model)
        self.backend = open_backend(self.model, backend, device, providers, trt_module)
        self.net_hw: Tuple[int, int] = tuple(self.meta["imgsz"])  # type: ignore[assignment]
        if tuple(self.backend.input_hw) != self.net_hw:
            raise ValueError(f"{self.model} has input {tuple(self.backend.input_hw)} but its metadata says {self.net_hw}")
        self.conf, self.masks_at, self.gpu_preprocess, self.device = float(conf), masks_at, bool(gpu_preprocess), device
        self.names: List[str] = list(self.meta.get("names", []))
        self.da_names, self.ll_names = list(self.meta.get("da_names", [])), list(self.meta.get("ll_names", []))
        if self.gpu_preprocess:
            import torch

            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("gpu_preprocess needs a CUDA device")

    # ---- stages
    def _pre(self, frame: np.ndarray):
        if self.gpu_preprocess:
            import torch

            t = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
            return letterbox_torch(t, self.net_hw)
        arr, info = preprocess(frame, self.net_hw)
        return arr[None], info

    def _post(self, out: Dict[str, np.ndarray], info: LetterboxInfo) -> Result:
        det = np.asarray(out["det"])[0]
        det = det[det[:, 4] >= self.conf]
        boxes = info.boxes_to_original(det[:, :4]) if len(det) else np.zeros((0, 4), np.float32)
        ok = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)  # drop boxes clipped away entirely
        boxes, det = boxes[ok], det[ok]
        maps = []
        for k in ("da", "ll"):
            m = np.asarray(out[k])[0]
            if m.ndim == 3:  # logits export (C, H, W)
                m = m.argmax(0)
            m = m.astype(np.uint8, copy=False)
            maps.append(info.mask_to_original(m) if self.masks_at == "original" else self._crop(m, info))
        return Result(boxes.astype(np.float32), det[:, 4].astype(np.float32), det[:, 5].astype(np.int64), maps[0], maps[1], info)

    @staticmethod
    def _crop(m: np.ndarray, info: LetterboxInfo) -> np.ndarray:
        w, h = info.unpad_wh
        return np.ascontiguousarray(m[info.top : info.top + h, info.left : info.left + w])

    def __call__(self, frame: np.ndarray) -> Result:
        t0 = time.perf_counter()
        x, info = self._pre(frame)
        t1 = time.perf_counter()
        out = self.backend.infer(x)
        t2 = time.perf_counter()
        res = self._post(out, info)
        t3 = time.perf_counter()
        res.timings = {"pre": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3}
        return res

    def close(self) -> None:
        self.backend.close()


# --------------------------------------------------------------------------- visualisation
def class_palette(n: int) -> List[Tuple[int, int, int]]:
    """Distinct BGR colours (golden-ratio hue steps)."""
    out = []
    for i in range(n):
        hsv = np.uint8([[[int((i * 0.618034 % 1.0) * 179), 200, 255]]])
        out.append(tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))
    return out


def overlay(frame: np.ndarray, res: Result, names: Sequence[str] = (), alpha: float = 0.45, boxes: bool = True) -> np.ndarray:
    """Tinted drivable area + opaque lane classes + boxes on a copy of ``frame``."""
    out = frame.copy()
    h, w = out.shape[:2]
    da, ll = res.da, res.ll
    if da.shape != (h, w):
        da = cv2.resize(da, (w, h), interpolation=cv2.INTER_NEAREST)
    if ll.shape != (h, w):
        ll = cv2.resize(ll, (w, h), interpolation=cv2.INTER_NEAREST)
    layer = np.zeros_like(out)
    for c, col in DA_COLORS.items():
        layer[da == c] = col
    m = da > 0
    if m.any():
        out[m] = cv2.addWeighted(out, 1 - alpha, layer, alpha, 0)[m]
    for c, col in LL_COLORS.items():
        out[ll == c] = col
    if boxes and len(res):
        pal = class_palette(max(int(res.classes.max()) + 1, len(names), 1))
        for (x1, y1, x2, y2), s, c in zip(res.boxes, res.scores, res.classes):
            col = pal[int(c) % len(pal)]
            p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
            cv2.rectangle(out, p1, p2, col, 2)
            label = f"{names[c] if c < len(names) else c} {s:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y0 = max(p1[1], th + bl)
            cv2.rectangle(out, (p1[0], y0 - th - bl), (p1[0] + tw, y0), col, -1)
            cv2.putText(out, label, (p1[0], y0 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- sources
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_frames(source: str | int | Path) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield ``(name, bgr_frame)`` from an image, a directory of images, a video file, a camera index
    (``0``) or a GStreamer pipeline string (contains ``!``, e.g. a CSI camera on a Jetson)."""
    src = str(source)
    if src.isdigit() or "!" in src:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src, cv2.CAP_ANY if src.isdigit() else cv2.CAP_GSTREAMER)
        yield from _capture(cap, "camera" if src.isdigit() else "gst", src)
        return
    p = Path(src)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"no images in {p}")
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                raise OSError(f"cannot read {f}")
            yield f.name, img
    elif p.suffix.lower() in VID_EXT:
        yield from _capture(cv2.VideoCapture(str(p)), p.stem, src)
    elif p.is_file():
        img = cv2.imread(str(p))
        if img is None:
            raise OSError(f"cannot read {p}")
        yield p.name, img
    else:
        raise FileNotFoundError(src)


def _capture(cap, stem: str, src: str) -> Iterator[Tuple[str, np.ndarray]]:
    if not cap.isOpened():
        raise OSError(f"cannot open video source {src}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"{stem}_{i:06d}", frame
            i += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------- statistics
def summarize(ms: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(ms, dtype=np.float64)
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "p50", "p90", "p99", "min", "max")}
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "min": float(a.min()), "max": float(a.max())}


def jetson_info() -> Dict[str, str]:
    """Best-effort device description for benchmark reports (power mode changes latency by 2x)."""
    info: Dict[str, str] = {}
    try:
        info["model"] = Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except OSError:
        pass
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        if lines:
            info["nvpmodel"] = " ".join(lines[:2])
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import tensorrt as trt  # noqa: F401

        info["tensorrt"] = str(trt.__version__)
    except ImportError:
        pass
    return info


def bench(pipe: Pipeline, frame: Optional[np.ndarray] = None, n: int = 300, warmup: int = 50) -> Dict[str, Any]:
    """Time ``pipe`` on one frame: per-stage latency distribution (ms) and FPS.

    The frame defaults to a 1280x720 synthetic one. The network cost does not depend on content, but the postprocessing
    cost scales with the number of detections, so pass a real frame (``--source``) for end-to-end numbers."""
    if frame is None:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    for _ in range(warmup):
        pipe(frame)
    stages: Dict[str, List[float]] = {"pre": [], "infer": [], "post": [], "total": []}
    for _ in range(n):
        t = pipe(frame).timings
        for k in stages:
            stages[k].append(t[k])
    stats = {k: summarize(v) for k, v in stages.items()}
    return {
        "frame_hw": [int(frame.shape[0]), int(frame.shape[1])], "n": n, "warmup": warmup, "stages_ms": stats,
        "fps_mean": 1000.0 / stats["total"]["mean"], "fps_p99": 1000.0 / stats["total"]["p99"],
        "backend": type(pipe.backend).__name__, "gpu_preprocess": pipe.gpu_preprocess, "device": jetson_info(),
    }


def format_bench(r: Dict[str, Any]) -> str:
    lines = [f"{r['backend']}  frame {r['frame_hw'][1]}x{r['frame_hw'][0]}  n={r['n']}  gpu_preprocess={r['gpu_preprocess']}"]
    if r["device"]:
        lines.append("device: " + "; ".join(f"{k}={v}" for k, v in r["device"].items()))
    lines.append(f"{'stage':>8} {'mean':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}   (ms)")
    for k, s in r["stages_ms"].items():
        lines.append(f"{k:>8} {s['mean']:8.2f} {s['p50']:8.2f} {s['p90']:8.2f} {s['p99']:8.2f} {s['max']:8.2f}")
    lines.append(f"FPS: mean {r['fps_mean']:.1f}   worst-1% {r['fps_p99']:.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- predict
def predict(model: str | Path, source: str | int | Path, out_dir: str | Path | None = None, conf: float = 0.25,
            backend: str = "auto", device: str = "cuda", gpu_preprocess: bool = False, save_masks: bool = False,
            show: bool = False, max_frames: int | None = None, fps: float | None = None) -> Dict[str, Any]:
    """Run on an image / directory / video / camera; write overlays (images, or one mp4 for video sources)."""
    pipe = Pipeline(model, backend, conf, device, gpu_preprocess)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    is_video = str(source).isdigit() or "!" in str(source) or Path(str(source)).suffix.lower() in VID_EXT
    writer = None
    totals: List[float] = []
    n = 0
    try:
        for name, frame in iter_frames(source):
            res = pipe(frame)
            totals.append(res.timings["total"])
            n += 1
            vis = overlay(frame, res, pipe.names) if (out or show) else None
            if out and vis is not None:
                if is_video:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(str(out / "predictions.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (w, h))
                    writer.write(vis)
                else:
                    cv2.imwrite(str(out / f"{Path(name).stem}.jpg"), vis)
                if save_masks:
                    cv2.imwrite(str(out / f"{Path(name).stem}_da.png"), res.da)
                    cv2.imwrite(str(out / f"{Path(name).stem}_ll.png"), res.ll)
            if show:
                cv2.imshow("adas_mt", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and n >= max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        pipe.close()
    stats = summarize(totals[1:] if len(totals) > 1 else totals)  # the first frame pays for lazy initialisation
    LOG.info("%d frame(s); total latency mean %.1f ms, p50 %.1f ms, p99 %.1f ms", n, stats["mean"], stats["p50"], stats["p99"])
    return {"frames": n, "latency_ms": stats, "out_dir": str(out) if out else None}
