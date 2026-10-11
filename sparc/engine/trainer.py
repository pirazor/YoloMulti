"""Lean trainer for SPARC (stage A onward): multi-source sampling, AMP, EMA, DDP, resume, per-source metrics.

    python -m sparc train --cfg sparc/cfg/stage_a.yaml --data sparc/cfg/data_stage_a.yaml --out runs/sparc/stage_a
    torchrun --nproc_per_node 4 -m sparc train ...        (DDP: the sampler splits each epoch across ranks)

Writes ``<out>/signs.json`` (the fine-sign vocabulary), ``run.yaml`` (effective config), ``results.csv`` (one row
per epoch with every per-source metric), ``weights/last.pt`` / ``best.pt`` (model, EMA, optimizer, scaler, schedule,
epoch, vocabulary). ``best.pt`` is chosen by ``fitness``: a weighted mean of the named metrics averaged over the
validation sources that report them.
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from adas_mt.data.masks import unpack_masks
from sparc.data.dataset import AugCfg, MixedDataset, SparcDataset, WeightedSourceSampler, collate
from sparc.data.schema import DA_CLASSES, LANE_MASK_CLASSES, Sample
from sparc.data.virtual_camera import VirtualCamera
from sparc.data.vocab import SignVocab
from sparc.engine.config import DataCfg, TrainCfg
from sparc.engine.evaluators import SourceEvaluator
from sparc.engine.postprocess import postprocess
from sparc.nn.criterion import Criterion, LossGains, sign_prefix_ids
from sparc.nn.decoder import DecoderCfg
from sparc.nn.model import SparcCfg, SparcNet

LOG = logging.getLogger("sparc.train")


class ModelEMA:
    """Exponential moving average with the Ultralytics decay ramp (short warm-up, full decay later)."""

    def __init__(self, model: nn.Module, decay: float = 0.9998, tau: int = 2000):
        self.ema = copy.deepcopy(_unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = _unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def build_model(cfg: TrainCfg, n_sign: int) -> SparcNet:
    m = cfg.model
    dec = DecoderCfg(layers=m.decoder_layers, n_agents=m.n_agents, n_elements=m.n_elements, n_lanes=m.n_lanes,
                     n_sign=max(1, n_sign), attn_mode=m.attn_mode)
    return SparcNet(SparcCfg(backbone=m.backbone, imgsz=tuple(m.imgsz), pretrained=m.pretrained, decoder=dec,
                             enc_depth=m.enc_depth, bev=m.bev, stereo=False, lane_aux=m.lane_aux))


def fitness_of(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted mean of each named metric averaged over the sources that report it (missing metrics are skipped)."""
    total, wsum = 0.0, 0.0
    for key, w in weights.items():
        vals = [v for k, v in metrics.items() if k.endswith("/" + key)]
        if vals:
            total += w * float(np.mean(vals))
            wsum += w
    return total / wsum if wsum else 0.0


class Trainer:
    def __init__(self, cfg: TrainCfg, data: DataCfg, out: str | Path, device: Optional[str] = None,
                 resume: Optional[str | Path] = None):
        self.cfg, self.data, self.out = cfg, data, Path(out)
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.distributed = self.world > 1
        if self.distributed:
            torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
            local = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(local)
            self.device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.main = self.rank == 0
        if self.main:
            (self.out / "weights").mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed + self.rank)
        np.random.seed(cfg.seed + self.rank)
        self.camera = VirtualCamera(tuple(cfg.model.imgsz), cfg.model.hfov)
        # vocabulary
        roots = [s.root for s in data.train] + [s.root for s in data.val]
        self.vocab = SignVocab.load(data.signs) if data.signs else SignVocab.from_manifests(roots)
        if self.main:
            self.vocab.save(self.out / "signs.json")
            (self.out / "run.yaml").write_text(yaml.safe_dump({"train": cfg.to_dict(), "data": data.to_dict()}, sort_keys=False))
        # data
        aug = AugCfg(**{f.name: (tuple(getattr(cfg.aug, f.name)) if f.name == "scale" else getattr(cfg.aug, f.name)) for f in fields(cfg.aug)})
        self.train_sets = [SparcDataset(s.root, s.split or "train", self.vocab, self.camera, aug, seed=cfg.seed, limit=s.limit)
                           for s in data.train]
        self.mixed = MixedDataset(self.train_sets, [s.weight for s in data.train])
        self.sampler = WeightedSourceSampler(self.mixed, cfg.epoch_size, cfg.seed, self.rank, self.world)
        self.train_loader = DataLoader(self.mixed, batch_size=cfg.batch, sampler=self.sampler, num_workers=cfg.workers,
                                       collate_fn=collate, pin_memory=self.device.type == "cuda", drop_last=True,
                                       persistent_workers=cfg.workers > 0)
        self.val_sets = [SparcDataset(s.root, s.split or "val", self.vocab, self.camera, None, limit=s.limit) for s in data.val]
        # model, loss, optimiser
        self.model = build_model(cfg, len(self.vocab)).to(self.device)
        gains = LossGains(**cfg.gains)
        self.criterion = Criterion(gains, sign_prefix_ids(self.vocab.names)).to(self.device)
        self.distill = None
        if cfg.distill.enabled:
            from sparc.distill import DistillHook

            self.distill = DistillHook(cfg.distill.teacher, cfg.distill.checkpoint, c_student=256, weight=cfg.distill.weight,
                                       affinity=cfg.distill.affinity, dtype=cfg.distill.dtype).to(self.device)
        self.optimizer = self._optimizer()
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = cfg.epochs * self.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp)
        if self.amp_dtype is not None and self.device.type != "cuda":
            self.amp_dtype = None  # autocast on CPU is slow and bf16-only; keep fp32
        self.scaler = torch.amp.GradScaler(enabled=self.amp_dtype == torch.float16)
        self.ema = ModelEMA(self.model, cfg.ema)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                                                             find_unused_parameters=True)
        self.epoch, self.step, self.best = 0, 0, -1.0
        self.history: List[Dict[str, float]] = []
        if resume:
            self.load(resume)

    # ------------------------------------------------------------------ pieces
    def _optimizer(self):
        cfg = self.cfg
        bb, rest, no_decay = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            elif name.startswith("backbone."):
                bb.append(p)
            else:
                rest.append(p)
        groups = [{"params": rest, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                  {"params": bb, "lr": cfg.lr * cfg.backbone_lr_mult, "weight_decay": cfg.weight_decay},
                  {"params": no_decay, "lr": cfg.lr, "weight_decay": 0.0}]
        if self.distill is not None:
            groups.append({"params": list(self.distill.proj.parameters()), "lr": cfg.lr, "weight_decay": cfg.weight_decay})
        for g in groups:
            g["initial_lr"] = g["lr"]
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999))

    def _lr_lambda(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.warmup_iters:
            return (step + 1) / cfg.warmup_iters
        t = (step - cfg.warmup_iters) / max(1, self.total_steps - cfg.warmup_iters)
        return cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def _forward(self, batch: Dict):
        image = batch["image"].to(self.device, non_blocking=True).float() / 255.0
        K, T = batch["K"].to(self.device), batch["cam2ego"].to(self.device)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            out = self.model(image, K=K, cam2ego=T)
        return image, out

    # ------------------------------------------------------------------ training
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.sampler.set_epoch(self.epoch)
        sums: Dict[str, float] = {}
        n = 0
        t0 = time.time()
        for i, batch in enumerate(self.train_loader):
            image, out = self._forward(batch)
            out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}
            if "aux" in out:
                out["aux"] = [{k: v.float() for k, v in a.items()} for a in out["aux"]]
            loss, items = self.criterion(out, batch)
            if self.distill is not None:
                kd = self.distill(image, out["enc_feats"][1].float(), self.step / max(1, self.total_steps))
                loss = loss + kd["kd"]
                items["kd"] = float(kd["kd"])
                items["kd_cos"] = float(kd["kd_cos"])
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            params = [p for g in self.optimizer.param_groups for p in g["params"]]
            gn = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema.update(self.model)
            self.step += 1
            n += 1
            for k, v in items.items():
                sums[k] = sums.get(k, 0.0) + v
            if self.main and (i + 1) % self.cfg.log_every == 0:
                LOG.info("epoch %d  it %d/%d  loss %.3f  (cls %.3f l1 %.3f giou %.3f lane_y %.3f da %.3f)  gn %.2f  lr %.2e  %.1f it/s",
                         self.epoch, i + 1, self.steps_per_epoch, items["total"], items["cls"], items["l1"], items["giou"],
                         items["lane_y"], items.get("da_ce", 0.0), float(gn), self.optimizer.param_groups[0]["lr"],
                         (i + 1) / (time.time() - t0))
        return {f"train/{k}": v / max(n, 1) for k, v in sums.items()}

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, model: Optional[nn.Module] = None) -> Dict[str, float]:
        model = (model or self.ema.ema).eval()
        results: Dict[str, float] = {}
        bs = self.cfg.val_batch or self.cfg.batch
        for ds in self.val_sets:
            ev = SourceEvaluator(ds.source, self.device)
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=self.cfg.workers, collate_fn=collate)
            for batch in loader:
                image = batch["image"].to(self.device).float() / 255.0
                with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    out = model(image, K=batch["K"].to(self.device), cam2ego=batch["cam2ego"].to(self.device))
                out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}
                batch["vocab"] = self.vocab
                preds = postprocess(out, batch, self.cfg.min_score)
                samples = [Sample.load(m["path"]) for m in batch["meta"]]
                da_t, _ = unpack_masks(batch["mask"].to(self.device), len(DA_CLASSES), len(LANE_MASK_CLASSES))
                has_da = batch["label_mask"]["da"].any()
                ev.update(preds, samples, batch, out["da_logits"] if has_da else None, da_t if has_da else None)
            results.update(ev.results())
        results["fitness"] = fitness_of(results, self.cfg.fitness)
        return results

    # ------------------------------------------------------------------ loop
    def fit(self) -> Dict[str, float]:
        cfg = self.cfg
        last = {}
        while self.epoch < cfg.epochs:
            t0 = time.time()
            train_items = self.train_epoch()
            self.epoch += 1
            row = {"epoch": self.epoch, **train_items}
            if self.main:
                val = self.validate()
                row.update(val)
                row["time_s"] = time.time() - t0
                self.history.append(row)
                self._write_results(row)
                improved = val["fitness"] > self.best
                self.best = max(self.best, val["fitness"])
                self.save(self.out / "weights" / "last.pt")
                if improved:
                    self.save(self.out / "weights" / "best.pt")
                LOG.info("epoch %d/%d  train loss %.3f  fitness %.4f (best %.4f)  %s", self.epoch, cfg.epochs,
                         train_items.get("train/total", 0.0), val["fitness"], self.best,
                         "  ".join(f"{k}={v:.3f}" for k, v in sorted(val.items()) if "/" in k and ("mAP50-95" in k or k.endswith("mIoU") or k.endswith("lane_F1"))))
                last = row
            if self.distributed:
                torch.distributed.barrier()
        return last

    def _write_results(self, row: Dict[str, float]) -> None:
        path = self.out / "results.csv"
        keys = sorted(set(k for r in self.history for k in r), key=lambda k: (k != "epoch", k))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.history:
                w.writerow({k: r.get(k, "") for k in keys})

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: Path) -> None:
        state = {"model": _unwrap(self.model).state_dict(), "ema": self.ema.ema.state_dict(), "ema_updates": self.ema.updates,
                 "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                 "epoch": self.epoch, "step": self.step, "best": self.best, "history": self.history,
                 "cfg": self.cfg.to_dict(), "data": self.data.to_dict(), "vocab": self.vocab.names, "aliases": self.vocab.aliases}
        if self.distill is not None:
            state["distill_proj"] = self.distill.proj.state_dict()
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck["vocab"] != self.vocab.names:
            raise ValueError("the checkpoint's sign vocabulary differs from the data manifest's; pass --signs <run>/signs.json")
        _unwrap(self.model).load_state_dict(ck["model"])
        self.ema.ema.load_state_dict(ck["ema"])
        self.ema.updates = ck["ema_updates"]
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        if self.distill is not None and "distill_proj" in ck:
            self.distill.proj.load_state_dict(ck["distill_proj"])
        self.epoch, self.step, self.best, self.history = ck["epoch"], ck["step"], ck["best"], ck.get("history", [])
        LOG.info("resumed %s at epoch %d (step %d, best %.4f)", path, self.epoch, self.step, self.best)


def load_for_eval(weights: str | Path, device: str = "cpu", ema: bool = True):
    """(model, vocab, cfg) from a checkpoint, for validation / export."""
    ck = torch.load(weights, map_location=device, weights_only=False)
    cfg = TrainCfg.from_dict(ck["cfg"])
    vocab = SignVocab(ck["vocab"], ck.get("aliases"))
    model = build_model(cfg, len(vocab))
    model.load_state_dict(ck["ema"] if ema else ck["model"])
    return model.to(device).eval(), vocab, cfg
