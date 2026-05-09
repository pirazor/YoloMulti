"""Self-contained training loop for ``YOLOv13MultiTask``.

We deliberately do **not** inherit from ``ultralytics.engine.trainer.BaseTrainer``
because that class is heavily entangled with ``build_yolo_dataset`` and the v8
label-cache pipeline -- both of which assume a single mask channel rasterized
from polygon segments. Reimplementing a trim training loop is shorter and keeps
the multi-task plumbing (joint loss, dual masks, dual-task validation) explicit.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import ModelEMA, init_seeds, one_cycle, select_device

from yolov13_multitask.data.multitask_dataset import MultiTaskDataset
from yolov13_multitask.engine.val import MultiTaskValidator
from yolov13_multitask.nn.multitask_model import YOLOv13MultiTask, DEFAULT_V13_CFG


class MultiTaskTrainer:
    """Joint training of detection + drivable-area + lane segmentation."""

    LOSS_NAMES = ("box_loss", "cls_loss", "dfl_loss", "da_loss", "ll_loss")

    def __init__(
        self,
        data: str | Path,
        cfg: str | Path,
        weights: Optional[str | Path] = None,
        model_yaml: Optional[str | Path] = None,
        device: Optional[str] = None,
        epochs: Optional[int] = None,
        batch: Optional[int] = None,
        imgsz: Optional[int] = None,
        workers: int = 4,
        project: Optional[str] = None,
        name: Optional[str] = None,
        resume: Optional[str | Path] = None,
    ) -> None:
        self.cfg_path = Path(cfg)
        with self.cfg_path.open("r", encoding="utf-8") as f:
            self.hyp = yaml.safe_load(f) or {}
        # CLI overrides
        if epochs is not None:
            self.hyp["epochs"] = epochs
        if batch is not None:
            self.hyp["batch"] = batch
        if imgsz is not None:
            self.hyp["imgsz"] = imgsz
        if project:
            self.hyp["project"] = project
        if name:
            self.hyp["name"] = name

        self.data_yaml = Path(data).resolve()
        with self.data_yaml.open("r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}

        self.device = select_device(device)
        init_seeds(int(self.hyp.get("seed", 0)))
        self.amp = bool(self.hyp.get("amp", True)) and self.device.type == "cuda"

        # Output dir --------------------------------------------------------
        project = Path(self.hyp.get("project", "runs/multitask"))
        run = self.hyp.get("name", "exp")
        self.save_dir = self._unique_dir(project / run, exist_ok=bool(self.hyp.get("exist_ok", False)))
        self.wdir = self.save_dir / "weights"
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last = self.wdir / "last.pt"
        self.best = self.wdir / "best.pt"
        self.csv_path = self.save_dir / "results.csv"

        # Model -------------------------------------------------------------
        model_cfg = str(model_yaml) if model_yaml else DEFAULT_V13_CFG
        self.model = YOLOv13MultiTask(
            cfg=model_cfg,
            nc=int(self.data["nc"]),
            da_classes=int(self.data.get("da_classes", 3)),
            ll_classes=int(self.data.get("ll_classes", 3)),
            decoder_mid_ch=int(self.hyp.get("decoder_mid_ch", 128)),
            verbose=False,
        ).to(self.device)
        # Detection-loss internal gains expected by v8DetectionLoss
        self.model.args = SimpleNamespace(
            box=float(self.hyp.get("box", 7.5)),
            cls=float(self.hyp.get("cls", 0.5)),
            dfl=float(self.hyp.get("dfl", 1.5)),
            loss_weights=self.hyp.get("loss_weights", {"det": 1.0, "da": 1.0, "ll": 1.0}),
        )
        self.model.use_uncertainty_weighting = bool(self.hyp.get("use_uncertainty_weighting", False))

        if weights:
            ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
            self.model.load(ckpt)

        # Dataloaders -------------------------------------------------------
        bs = int(self.hyp.get("batch", 16))
        imgsz = int(self.hyp.get("imgsz", 640))
        self.train_ds = MultiTaskDataset(
            self.data_yaml, split="train", imgsz=imgsz, augment=True, hyp=self.hyp,
            mosaic_prob=float(self.hyp.get("mosaic", 0.5)),
        )
        self.val_ds = MultiTaskDataset(
            self.data_yaml, split="val", imgsz=imgsz, augment=False, hyp=self.hyp,
        )
        self.train_loader = DataLoader(
            self.train_ds, batch_size=bs, shuffle=True, num_workers=workers,
            collate_fn=MultiTaskDataset.collate_fn, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=bs, shuffle=False, num_workers=max(1, workers // 2),
            collate_fn=MultiTaskDataset.collate_fn, pin_memory=True,
        )

        # Optimizer + sched -------------------------------------------------
        self.optimizer = self._build_optimizer()
        self.epochs = int(self.hyp.get("epochs", 100))
        lrf = float(self.hyp.get("lrf", 0.01))
        self.lf = one_cycle(1.0, lrf, self.epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lf)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.ema = ModelEMA(self.model)

        self.freeze_backbone_epochs = int(self.hyp.get("freeze_backbone_epochs", 0))
        self.best_fitness = -math.inf

        if resume:
            self._resume(Path(resume))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unique_dir(p: Path, exist_ok: bool) -> Path:
        if not p.exists() or exist_ok:
            p.mkdir(parents=True, exist_ok=True)
            return p
        i = 2
        while True:
            cand = Path(f"{p}{i}")
            if not cand.exists():
                cand.mkdir(parents=True, exist_ok=True)
                return cand
            i += 1

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, nodecay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias"):
                nodecay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": float(self.hyp.get("weight_decay", 5e-4))},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        opt_name = str(self.hyp.get("optimizer", "AdamW")).lower()
        lr0 = float(self.hyp.get("lr0", 1e-3))
        if opt_name == "adamw":
            return torch.optim.AdamW(param_groups, lr=lr0, betas=(0.9, 0.999))
        if opt_name == "adam":
            return torch.optim.Adam(param_groups, lr=lr0, betas=(0.9, 0.999))
        return torch.optim.SGD(param_groups, lr=lr0, momentum=float(self.hyp.get("momentum", 0.937)))

    def _set_freeze(self, freeze: bool) -> None:
        for n, p in self.model.named_parameters():
            if "da_decoder" in n or "ll_decoder" in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(not freeze)

    # ----------------------------------------------------------------- main loop
    def train(self) -> None:
        nb = len(self.train_loader)
        warmup_iters = max(1, int(self.hyp.get("warmup_epochs", 3)) * nb)
        last_opt_step = -1
        self._init_csv()

        for epoch in range(self.epochs):
            self.model.train()
            self._set_freeze(epoch < self.freeze_backbone_epochs)
            ep_loss = torch.zeros(len(self.LOSS_NAMES), device=self.device)
            t0 = time.time()
            for i, batch in enumerate(self.train_loader):
                ni = i + nb * epoch
                # Warmup ------------------------------------------------------
                if ni <= warmup_iters:
                    xi = [0, warmup_iters]
                    for j, pg in enumerate(self.optimizer.param_groups):
                        pg["lr"] = np.interp(ni, xi, [0.0 if j == 0 else float(self.hyp.get("warmup_bias_lr", 0.1)),
                                                       pg.get("initial_lr", float(self.hyp.get("lr0", 1e-3))) * self.lf(epoch)])

                batch = self._to_device(batch)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    loss, items = self.model.loss(batch)
                self.scaler.scale(loss).backward()
                # Step optimizer every iter (no grad accumulation by default)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.ema.update(self.model)
                last_opt_step = ni

                ep_loss = (ep_loss * i + items.detach()) / (i + 1)
                if i % 50 == 0:
                    LOGGER.info(
                        "ep %d/%d it %d/%d lr %.2e | %s",
                        epoch + 1, self.epochs, i, nb, self.optimizer.param_groups[0]["lr"],
                        " ".join(f"{n}={v:.4f}" for n, v in zip(self.LOSS_NAMES, ep_loss.tolist())),
                    )

            self.scheduler.step()
            t1 = time.time()
            LOGGER.info("epoch %d done in %.1fs", epoch + 1, t1 - t0)

            # Validation ---------------------------------------------------
            metrics = self._validate()
            fitness = self._fitness(metrics)
            self._log_row(epoch, ep_loss, metrics, fitness)
            self._save_ckpt(epoch, fitness)

    # ------------------------------------------------------------------ val
    def _validate(self) -> Dict:
        validator = MultiTaskValidator(
            data=self.data,
            device=self.device,
            ll_classes=self.model.ll_classes,
            da_classes=self.model.da_classes,
        )
        return validator.run(self.ema.ema, self.val_loader)

    def _fitness(self, m: Dict) -> float:
        return 0.5 * float(m.get("map50_95", 0.0)) + 0.25 * float(m.get("mIoU_da", 0.0)) + 0.25 * float(m.get("mIoU_ll", 0.0))

    # ---------------------------------------------------------------- IO helpers
    def _to_device(self, batch: Dict) -> Dict:
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["da_mask"] = batch["da_mask"].to(self.device, non_blocking=True)
        batch["ll_mask"] = batch["ll_mask"].to(self.device, non_blocking=True)
        batch["bboxes"] = batch["bboxes"].to(self.device, non_blocking=True)
        batch["cls"] = batch["cls"].to(self.device, non_blocking=True)
        batch["batch_idx"] = batch["batch_idx"].to(self.device, non_blocking=True)
        return batch

    def _init_csv(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["epoch", *self.LOSS_NAMES, "map50", "map50_95", "mIoU_da", "mIoU_ll", "fitness"]
            )

    def _log_row(self, epoch: int, ep_loss: torch.Tensor, m: Dict, fitness: float) -> None:
        with self.csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [epoch + 1, *(round(float(x), 5) for x in ep_loss.tolist()),
                 round(float(m.get("map50", 0.0)), 5), round(float(m.get("map50_95", 0.0)), 5),
                 round(float(m.get("mIoU_da", 0.0)), 5), round(float(m.get("mIoU_ll", 0.0)), 5),
                 round(fitness, 5)]
            )

    def _save_ckpt(self, epoch: int, fitness: float) -> None:
        ckpt = {
            "epoch": epoch,
            "model": deepcopy(self.ema.ema).half(),
            "optimizer": self.optimizer.state_dict(),
            "fitness": fitness,
            "data_yaml": str(self.data_yaml),
            "hyp": self.hyp,
        }
        torch.save(ckpt, self.last)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            torch.save(ckpt, self.best)

    def _resume(self, path: Path) -> None:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        self.model.load(ck)
        self.best_fitness = float(ck.get("fitness", -math.inf))
        LOGGER.info("resumed from %s (fitness %.4f)", path, self.best_fitness)
