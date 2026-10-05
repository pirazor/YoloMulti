"""Evaluate an exported model (ONNX or TensorRT engine) with the training-time validator.

The same dataset pipeline, the same metrics (mAP, DA mIoU, lane IoU) and the same fitness as ``adas_mt val`` on the
``.pt``, so the difference between the two is exactly what export and quantisation cost::

    python -m adas_mt val  --weights runs/mt/exp/weights/best.pt  --data data.yaml
    python -m adas_mt eval --model   runs/mt/exp/weights/best.onnx   --data data.yaml
    python -m adas_mt eval --model   runs/mt/exp/weights/best.engine --data data.yaml --device 0
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np

from .meta import read_meta
from .postprocess import decode_det
from .runner import ENGINE_SUFFIXES, open_backend


class _ExportedModel(SimpleNamespace):
    """The attributes ``MultiTaskValidator.init_metrics`` reads from a model."""


def run_batched(backend, x) -> Dict[str, np.ndarray]:
    """Run ``x`` (B, 3, H, W) through a backend whose batch may be fixed (engines are usually built for batch 1).

    A short last chunk is padded to a static batch size (repeating its last image) and the padding sliced off."""
    import torch

    n = int(x.shape[0])
    static = bool(backend.batch)
    step = backend.batch or getattr(backend, "max_batch", None) or n
    outs = []
    for i in range(0, n, step):
        chunk = x[i : i + step]
        m = int(chunk.shape[0])
        if static and m < step:
            chunk = torch.cat([chunk, chunk[-1:].expand(step - m, -1, -1, -1)])
        outs.append({k: v[:m] for k, v in backend.infer(chunk).items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def evaluate(model: str | Path, data: str | Path, batch: int = 16, device: str | None = None, split: str = "val",
             backend: str = "auto", conf: float = 0.001, workers: int = 2, trt_module: Any = None,
             providers=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns ``(stats, speed)``: validator statistics (same keys as ``adas_mt val``) and
    ``{'infer_ms_per_img': ..., 'images': ...}`` measured around the backend call only."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import LOGGER

    from adas_mt.engine.config import MTConfig
    from adas_mt.engine.val import MultiTaskValidator

    meta = read_meta(model)
    mt = MTConfig.from_dict({"imgsz": meta["imgsz"]})
    if device is None:  # an engine can only run on the GPU; anything else defaults to the CPU
        device = "0" if Path(model).suffix.lower() in ENGINE_SUFFIXES else "cpu"
    dev = torch.device("cpu" if str(device) in {"", "cpu"} else (f"cuda:{device}" if str(device).isdigit() else str(device)))
    be = open_backend(model, backend, str(dev), providers, trt_module)
    args = get_cfg(overrides={
        "model": str(model), "data": str(data), "batch": batch, "imgsz": max(mt.imgsz), "split": split, "nms": False,
        "plots": False, "conf": conf, "mode": "val", "workers": workers, "device": str(dev),
        "project": str(Path.cwd() / "runs" / "mt"), "name": "eval", "exist_ok": True, "max_det": int(meta["max_det"]),
    })
    v = MultiTaskValidator(args=args, mt=mt)
    v.training, v.device, v.stride = False, dev, 32
    v.data = check_det_dataset(str(data), split=split)
    for key, have in (("nc", meta["nc"]), ("da_classes", meta["da_classes"]), ("ll_classes", meta["ll_classes"])):
        if int(v.data[key]) != int(have):
            raise ValueError(f"data.yaml {key}={v.data[key]} but {model} was exported with {have}")
    v.names = {i: n for i, n in enumerate(meta["names"])}
    v.dataloader = v.get_dataloader(v.data.get(split), batch)
    adapter = _ExportedModel(format="exported", names=v.names, end2end=True, da_classes=meta["da_classes"],
                             ll_classes=meta["ll_classes"], da_names=meta["da_names"], ll_names=meta["ll_names"])
    v.init_metrics(adapter)

    n_img, t_infer = 0, 0.0
    for batch_ in v.dataloader:
        batch_ = v.preprocess(batch_)
        t0 = time.perf_counter()
        out = run_batched(be, batch_["img"])
        t_infer += time.perf_counter() - t0
        out["det"] = decode_det(out["det"], meta)  # host top-k for --det-head raw exports
        n_img += int(batch_["img"].shape[0])
        preds = v.postprocess({k: torch.from_numpy(np.ascontiguousarray(out[k])).to(dev) for k in ("det", "da", "ll")})
        v.update_metrics(preds, batch_)
    v.gather_stats()
    stats = v.get_stats()
    v.finalize_metrics()
    v.print_results()
    be.close()
    speed = {"infer_ms_per_img": t_infer / max(n_img, 1) * 1e3, "images": float(n_img)}
    LOGGER.info(f"backend inference: {speed['infer_ms_per_img']:.2f} ms/img over {n_img} images "
                f"(batch {be.batch or 'dynamic'}; excludes data loading and postprocessing)")
    return stats, speed
