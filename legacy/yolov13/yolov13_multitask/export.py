"""Export multi-task YOLOv13 to ONNX (and optionally TensorRT FP16/INT8).

ONNX is exported with three named outputs: ``det``, ``da``, ``ll``.

The model is briefly wrapped in a ``nn.Module`` that returns a tuple from
``forward`` (ONNX dislikes dict outputs and we set ``export_mode=True`` on
the model, but the wrapper guards against future changes).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

from yolov13_multitask.nn.multitask_model import YOLOv13MultiTask


class _ExportWrapper(nn.Module):
    def __init__(self, model: YOLOv13MultiTask):
        super().__init__()
        self.model = model
        self.model.export_mode = True
        # Detect head export-mode so we get a tensor instead of (y, x) tuple.
        self.model.model[-1].export = True

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if isinstance(out, dict):
            return out["det"], out["da"], out["ll"]
        return out


def _load_model(weights: Path) -> YOLOv13MultiTask:
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
    else:
        model = ckpt
    if not isinstance(model, YOLOv13MultiTask):
        raise TypeError("checkpoint does not contain a YOLOv13MultiTask")
    return model.float().eval()


def export_onnx(
    weights: Path,
    output: Optional[Path] = None,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = False,
    verify: bool = True,
) -> Path:
    weights = Path(weights)
    output = Path(output) if output else weights.with_suffix(".onnx")
    model = _load_model(weights)
    wrapped = _ExportWrapper(model).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "images": {0: "batch"},
            "det": {0: "batch"},
            "da": {0: "batch"},
            "ll": {0: "batch"},
        }
    LOGGER.info("exporting ONNX -> %s", output)
    torch.onnx.export(
        wrapped,
        dummy,
        str(output),
        input_names=["images"],
        output_names=["det", "da", "ll"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    if verify:
        try:
            import onnx
            onnx.checker.check_model(str(output))
        except Exception as e:  # pragma: no cover
            LOGGER.warning("onnx checker failed: %s", e)

        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            ort_outs = sess.run(None, {"images": dummy.numpy()})
            with torch.no_grad():
                torch_outs = wrapped(dummy)
            for name, ort_o, t_o in zip(("det", "da", "ll"), ort_outs, torch_outs):
                t_np = t_o.detach().cpu().numpy()
                diff = float(np.max(np.abs(ort_o - t_np)))
                LOGGER.info("ORT vs torch [%s] max-abs-diff = %.3e", name, diff)
                if diff > 1e-3:
                    LOGGER.warning("ONNX output '%s' diverges from PyTorch (>1e-3): %.3e", name, diff)
        except ImportError:
            LOGGER.info("onnxruntime not available, skipping numerical-equivalence check")
        except Exception as e:  # pragma: no cover
            LOGGER.warning("ORT verification failed: %s", e)
    return output


# --------------------------------------------------------------- TensorRT path
class _DirCalibrator:
    """Image-folder INT8 calibrator used by the TRT builder.

    Produces fixed-shape float32 batches in NCHW with values in [0, 1].
    """

    def __init__(self, image_dir: Path, batch_size: int, imgsz: int, cache_path: Path):
        import cv2

        self.cv2 = cv2
        self.batch_size = int(batch_size)
        self.imgsz = int(imgsz)
        self.cache_path = Path(cache_path)
        self.files = [
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not self.files:
            raise FileNotFoundError(f"no images in {image_dir} for calibration")
        self.idx = 0

    def next_batch(self) -> Optional[np.ndarray]:
        if self.idx + self.batch_size > len(self.files):
            return None
        batch = []
        for p in self.files[self.idx : self.idx + self.batch_size]:
            im = self.cv2.imread(str(p))
            if im is None:
                continue
            im = self.cv2.cvtColor(im, self.cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]
            r = min(self.imgsz / h, self.imgsz / w)
            im = self.cv2.resize(im, (int(round(w * r)), int(round(h * r))))
            canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
            canvas[: im.shape[0], : im.shape[1]] = im
            batch.append(canvas.astype(np.float32) / 255.0)
        self.idx += self.batch_size
        if not batch:
            return None
        return np.transpose(np.stack(batch, 0), (0, 3, 1, 2)).astype(np.float32)


def export_tensorrt(
    onnx_path: Path,
    output: Optional[Path] = None,
    fp16: bool = False,
    int8: bool = False,
    calib_dir: Optional[Path] = None,
    imgsz: int = 640,
    workspace_gb: int = 4,
) -> Path:
    """Build a TensorRT engine from an ONNX file.

    Requires ``tensorrt`` to be installed (NVIDIA-supplied wheel).
    """
    onnx_path = Path(onnx_path)
    output = Path(output) if output else onnx_path.with_suffix(".engine")
    try:
        import tensorrt as trt  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("tensorrt is required for engine export") from e

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                LOGGER.error(parser.get_error(i))
            raise RuntimeError("failed to parse ONNX")

    config = builder.create_builder_config()
    config.max_workspace_size = int(workspace_gb) * (1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not calib_dir:
            raise ValueError("--calib_dir is required when --int8 is set")
        config.set_flag(trt.BuilderFlag.INT8)
        cache = output.with_suffix(".cache")

        class _Calib(trt.IInt8MinMaxCalibrator):  # pragma: no cover - requires TRT
            def __init__(self_inner):
                trt.IInt8MinMaxCalibrator.__init__(self_inner)
                self_inner.helper = _DirCalibrator(calib_dir, 1, imgsz, cache)
                self_inner._buf = None

            def get_batch_size(self_inner):
                return 1

            def get_batch(self_inner, names):
                import cuda.cudart as cudart  # type: ignore
                arr = self_inner.helper.next_batch()
                if arr is None:
                    return None
                self_inner._buf = arr
                return [int(arr.ctypes.data)]

            def read_calibration_cache(self_inner):
                if cache.exists():
                    return cache.read_bytes()
                return None

            def write_calibration_cache(self_inner, cache_bytes):
                cache.write_bytes(cache_bytes)

        config.int8_calibrator = _Calib()

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    output.write_bytes(serialized)
    LOGGER.info("TensorRT engine -> %s (fp16=%s, int8=%s)", output, fp16, int8)
    return output
