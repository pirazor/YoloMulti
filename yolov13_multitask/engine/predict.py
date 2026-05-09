"""Multi-task predictor: detection + drivable-area + lane segmentation overlay.

Self-contained (does not subclass BasePredictor) because we need fine control
over the dual-mask postprocess and overlay rendering. Supports image / image-
directory / video sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER, ops

from yolov13_multitask.nn.multitask_model import YOLOv13MultiTask
from yolov13_multitask.utils.visualize import render_multitask


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _letterbox(
    img: np.ndarray, new_shape: int, color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize + pad to a square, return (image, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    canvas[:nh, :nw] = resized  # top-left padding to match albumentations val transform
    return canvas, r, (0, 0)


class MultiTaskPredictor:
    def __init__(
        self,
        weights: str | Path,
        device: Optional[str] = None,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        names: Optional[List[str]] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.imgsz = int(imgsz)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model = ckpt["model"]
        else:
            self.model = ckpt
        if not isinstance(self.model, YOLOv13MultiTask):
            raise TypeError(
                "Loaded weights do not contain a YOLOv13MultiTask model. "
                "Pass a checkpoint produced by yolov13_multitask training."
            )
        self.model = self.model.float().to(self.device).eval()
        self.names = names or list(getattr(self.model, "names", {i: str(i) for i in range(self.model.nc)}).values())

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict_image(self, image_bgr: np.ndarray):
        h0, w0 = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        canvas, scale, _ = _letterbox(rgb, self.imgsz)
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        preds = self.model(tensor)
        det = preds["det"] if isinstance(preds, dict) else preds[0]
        da_logits = preds["da"] if isinstance(preds, dict) else preds[1]
        ll_logits = preds["ll"] if isinstance(preds, dict) else preds[2]

        decoded = det[0] if isinstance(det, (tuple, list)) else det
        nms_out = ops.non_max_suppression(
            decoded, self.conf_thres, self.iou_thres, max_det=self.max_det
        )[0]

        # Map back to original image
        boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        cls = np.zeros((0,), dtype=np.int64)
        conf = np.zeros((0,), dtype=np.float32)
        if nms_out is not None and nms_out.numel():
            np_pred = nms_out.cpu().numpy()
            boxes_xyxy = np_pred[:, :4] / scale
            boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, w0)
            boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, h0)
            conf = np_pred[:, 4]
            cls = np_pred[:, 5].astype(np.int64)

        # Resize masks back to original
        nh = int(round(h0 * scale))
        nw = int(round(w0 * scale))
        da_arg = da_logits.argmax(dim=1)[0]  # (H, W) on imgsz x imgsz
        ll_arg = ll_logits.argmax(dim=1)[0]
        da_arg = da_arg[:nh, :nw]
        ll_arg = ll_arg[:nh, :nw]
        da_mask = cv2.resize(da_arg.cpu().numpy().astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_arg.cpu().numpy().astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)

        return {
            "boxes_xyxy": boxes_xyxy,
            "cls": cls,
            "conf": conf,
            "da_mask": da_mask,
            "ll_mask": ll_mask,
            "rgb": rgb,
        }

    def render(self, result: dict) -> np.ndarray:
        return render_multitask(
            result["rgb"], da_mask=result["da_mask"], ll_mask=result["ll_mask"],
            boxes_xyxy=result["boxes_xyxy"], cls=result["cls"], conf=result["conf"],
            names=self.names,
        )

    # ------------------------------------------------------------------ sources
    def run(self, source: str | Path, save_dir: Path, save: bool = True) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        src = Path(source)
        if src.is_dir():
            files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            for p in files:
                im = cv2.imread(str(p))
                if im is None:
                    continue
                r = self.predict_image(im)
                out = cv2.cvtColor(self.render(r), cv2.COLOR_RGB2BGR)
                if save:
                    cv2.imwrite(str(save_dir / p.name), out)
        elif src.suffix.lower() in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise RuntimeError(f"could not open video {src}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = None
            if save:
                writer = cv2.VideoWriter(
                    str(save_dir / f"{src.stem}_overlay.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
                )
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                r = self.predict_image(frame)
                out = cv2.cvtColor(self.render(r), cv2.COLOR_RGB2BGR)
                if writer is not None:
                    writer.write(out)
            cap.release()
            if writer is not None:
                writer.release()
        else:
            im = cv2.imread(str(src))
            if im is None:
                raise FileNotFoundError(src)
            r = self.predict_image(im)
            out = cv2.cvtColor(self.render(r), cv2.COLOR_RGB2BGR)
            if save:
                cv2.imwrite(str(save_dir / src.name), out)
        LOGGER.info("predictions saved under %s", save_dir)
