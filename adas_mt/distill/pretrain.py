"""Stage A: distillation-only pretraining of backbone + neck on UNLABELLED driving images.

Every frame you have (not just the annotated ones) teaches the student the foundation model's dense
features; Stage B (normal multi-task training) then starts from this checkpoint via
``build_model(scale, weights=<last.pt>)``.

    python -m adas_mt.distill.pretrain --images /data/frames --scale s --weights yolo26s.pt \
        --teacher dinov3_s_plus --epochs 20 --batch 32 --imgsz 384 640
"""

from __future__ import annotations

import argparse
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from adas_mt.distill.loss import Distiller
from adas_mt.distill.teacher import FrozenTeacher

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnlabeledImages(Dataset):
    """Images -> random (h, w) views: scale jitter, random placement, flip, brightness/contrast. uint8 RGB CHW."""

    def __init__(self, root: str | Path, hw: Sequence[int] = (384, 640), augment: bool = True):
        self.files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not self.files:
            raise FileNotFoundError(f"no images under {root}")
        self.hw, self.augment = (int(hw[0]), int(hw[1])), augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.files[i])
        h, w = self.hw
        h0, w0 = img.shape[:2]
        r = min(h / h0, w / w0) * (random.uniform(0.75, 1.35) if self.augment else 1.0)
        nh, nw = max(2, round(h0 * r)), max(2, round(w0 * r))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h, w, 3), 114, np.uint8)
        # placement: centred when validating, random when augmenting; crop when the view is larger
        oy = (h - nh) // 2 if not self.augment else random.randint(min(0, h - nh), max(0, h - nh))
        ox = (w - nw) // 2 if not self.augment else random.randint(min(0, w - nw), max(0, w - nw))
        y1, x1 = max(oy, 0), max(ox, 0)
        y2, x2 = min(oy + nh, h), min(ox + nw, w)
        canvas[y1:y2, x1:x2] = img[y1 - oy : y2 - oy, x1 - ox : x2 - ox]
        if self.augment:
            if random.random() < 0.5:
                canvas = canvas[:, ::-1]
            a, b = random.uniform(0.8, 1.2), random.uniform(-20, 20)  # contrast, brightness
            canvas = np.clip(canvas.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1)))  # BGR->RGB, CHW


def pretrain(
    model,
    teacher: FrozenTeacher,
    images: str | Path,
    imgsz: Sequence[int] = (384, 640),
    epochs: int = 20,
    batch: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    workers: int = 4,
    device: str | torch.device | None = None,
    save_dir: str | Path = "runs/distill/exp",
    amp: bool = True,
    w_cos: float = 1.0,
    w_aff: float = 1.0,
    every: int = 1,
    seed: int = 0,
    log=print,
) -> Path:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed), random.seed(seed), np.random.seed(seed)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    distiller = Distiller(model, teacher, w_cos=w_cos, w_aff=w_aff, weight=1.0, weight_end=1.0, every=every)

    ds = UnlabeledImages(images, imgsz, augment=True)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=len(ds) >= batch,
                    pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    params = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("model.") or n.startswith("kd_proj"))
              and not n.startswith(f"model.{len(model.model) - 1}.")]  # backbone + neck + projector only
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    total = epochs * len(dl)
    warm = min(100, max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / max(total, 1))))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    last = save_dir / "last.pt"
    for ep in range(epochs):
        t0, run_cos, run_loss, n = time.time(), 0.0, 0.0, 0
        for img in dl:
            img = img.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                _, p3, p4 = model.features(img.float() / 255.0)
                loss, items = distiller.loss_from_feats(p3, p4, img)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_cos += float(items["kd_cos"])
            run_loss += float(items["kd_loss"])
            n += 1
        log(f"epoch {ep + 1}/{epochs}  kd_loss {run_loss / max(n, 1):.4f}  cos_sim {run_cos / max(n, 1):.4f}  {time.time() - t0:.0f}s")
        export = deepcopy(model).cpu().float().strip_training_only()  # clean checkpoint: no projector, no aux
        torch.save({"model": export, "epoch": ep, "kd_cos": run_cos / max(n, 1), "teacher": teacher.name}, last)
    return last


def main(argv=None) -> int:
    from adas_mt.nn import MultiTaskModel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of (unlabelled) frames, searched recursively")
    ap.add_argument("--scale", default="s", choices=list("nsmlx"))
    ap.add_argument("--weights", default=None, help="pretrained YOLO26 checkpoint to start from")
    ap.add_argument("--nc", type=int, default=9)
    ap.add_argument("--teacher", default="dinov3_s_plus")
    ap.add_argument("--teacher_ckpt", default=None, help="local teacher weights (.pth/.safetensors) instead of timm download")
    ap.add_argument("--teacher_scale", type=float, default=1.0, help="teacher input scale (0.5 = 4x cheaper, coarser targets)")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--every", type=int, default=1, help="distil every k-th step (teacher is the cost)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save_dir", default="runs/distill/exp")
    a = ap.parse_args(argv)

    from adas_mt.nn import build_model

    model = build_model(a.scale, nc=a.nc, weights=a.weights)
    teacher = FrozenTeacher(a.teacher, pretrained=a.teacher_ckpt is None, checkpoint=a.teacher_ckpt, input_scale=a.teacher_scale)
    pretrain(model, teacher, a.images, a.imgsz, a.epochs, a.batch, a.lr, a.workers, a.device, a.save_dir, every=a.every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
