"""Trainer: Ultralytics ``DetectionTrainer`` specialised for the multi-task model.

Everything the stock trainer does (AMP, EMA, warmup, close_mosaic, DDP, resume, plots, checkpoints) is kept.
What is added (each item is a requirement found by running a prototype against ultralytics 8.4.171, see
``docs/training_requirements.md``):

* packed (DA+LL) mask dataset at a fixed ``(h, w)`` for train AND val (``rect=False``);
* the model is built with ``kd_dim`` so the distillation projector exists before optimizer/EMA/DDP;
* new heads (``da_head``, ``ll_head``, ``kd_proj``) get a boosted LR whatever the optimizer is;
* masks are resized with the image under ``multi_scale``;
* multi-task settings travel via ``mt.yaml`` / ``ADAS_MT_CFG`` (extra keys are invalid Ultralytics args and
  DDP workers are re-created from ``vars(args)``);
* validation uses the NMS-free head (``nms=False``) on the deployed geometry.
"""

from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import LOCAL_RANK, LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from adas_mt.data.dataset import MultiTaskDataset
from adas_mt.nn.model import MultiTaskModel, check_matches_data

from .config import MTConfig, env_cfg
from .val import MultiTaskValidator

NEW_MODULES = ("da_head", "ll_head", "kd_proj")  # freshly initialised: higher LR than the pretrained trunk


class MultiTaskTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides: Dict[str, Any] | None = None, _callbacks=None, mt=None):
        overrides = dict(overrides or {})
        explicit = mt or overrides.pop("mt_cfg", None)
        resume = overrides.get("resume")
        resuming = bool(resume)
        run_cfg = MTConfig.find_run_cfg(resume)
        if run_cfg is not None:
            # The run's own mt.yaml is authoritative on resume: the optimizer state, EMA and checkpoint contents
            # depend on it (e.g. toggling distillation changes the parameter groups and the resume crashes).
            if explicit is not None:
                LOGGER.warning(f"resuming: ignoring the given multi-task config in favour of {run_cfg}")
            self.mt = MTConfig.load(run_cfg)
        else:
            if resuming and resume is not True:  # resume=True is resolved to a path by check_resume (handled below)
                LOGGER.warning("resuming but no <run>/mt.yaml was found: using the given/default multi-task config; "
                               "a mismatch with the original run (distillation, imgsz) will break the resume")
            self.mt = MTConfig.resolve(explicit, resume=resume)
        if overrides.get("imgsz") not in (None, max(self.mt.imgsz)):
            LOGGER.warning(f"ignoring imgsz={overrides['imgsz']}: the geometry is mt.imgsz={tuple(self.mt.imgsz)}")
        overrides["imgsz"] = max(self.mt.imgsz)  # Ultralytics wants an int (multi_scale range, autobatch, BN check)
        overrides.setdefault("nms", False)  # validate the NMS-free head that ships; None would validate o2m + NMS
        if not resuming:
            # Ultralytics prefixes a relative `project` with runs/<task>/ (runs/detect/runs/mt/exp): make it absolute
            overrides["project"] = str(Path(overrides["project"]).resolve()) if overrides.get("project") else str(
                Path.cwd() / "runs" / "mt")
            overrides.setdefault("name", "exp")
        self.teacher = None
        super().__init__(cfg, overrides, _callbacks)
        if resuming and run_cfg is None and resume is True:
            # Python-API `resume=True`: check_resume resolved it to the latest last.pt; that run's mt.yaml is as
            # authoritative as for an explicit path (distillation / geometry decide the parameter groups).
            run_cfg = MTConfig.find_run_cfg(self.args.resume)
            if run_cfg is not None:
                if explicit is not None:
                    LOGGER.warning(f"resuming: ignoring the given multi-task config in favour of {run_cfg}")
                self.mt = MTConfig.load(run_cfg)
                self.args.imgsz = max(self.mt.imgsz)
            else:
                LOGGER.warning(f"resuming {self.args.resume} but no <run>/mt.yaml was found next to it: using the "
                               "given/default multi-task config")
        if self.args.nms is not False:
            LOGGER.warning("nms is not False: validation will use the one-to-many head + NMS, not the deployed head")
        if isinstance(self.args.batch, (int, float)) and self.args.batch < 1:
            # Ultralytics' autobatch profiles a square imgsz x imgsz input, cannot measure the backward pass of a model
            # whose forward returns a dict (profile_ops swallows the error) and ignores the distillation teacher: the
            # estimate is meaningless and a wrong batch is only auto-reduced 3 times in epoch 0.
            raise ValueError(f"batch={self.args.batch}: autobatch is not supported for the multi-task model, set batch explicitly")
        if self.args.compile:
            LOGGER.warning("compile=True is not tested with the multi-task model")
        if RANK in {-1, 0} and (run_cfg is None or not (self.save_dir / "mt.yaml").exists()):
            self.mt.save(self.save_dir / "mt.yaml")  # never rewrite the file of a resumed run

    def train(self):
        # only the launching process hands the config to workers; workers (self.ddp False) just train
        return self._train_ddp() if self.ddp else super().train()

    def _train_ddp(self):
        """DDP workers are re-created from ``vars(args)`` and would lose ``self.mt``: hand them the config through
        ADAS_MT_CFG. The file must live OUTSIDE ``save_dir``: Ultralytics deletes the run directory right before
        spawning the workers (unless resuming), and each worker re-writes ``<save_dir>/mt.yaml`` itself."""
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="adas_mt_"))
        try:
            self.mt.save(tmp / "mt.yaml")
            with env_cfg(tmp / "mt.yaml"):
                return super().train()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ model
    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        scale = (cfg.get("scale") if isinstance(cfg, dict) else guess_model_scale(cfg or "")) or self.mt.scale
        if scale != self.mt.scale:
            LOGGER.info(f"model scale {scale!r} comes from the checkpoint/yaml (mt.scale was {self.mt.scale!r})")
            self.mt.scale = scale  # record what was really trained
            if RANK in {-1, 0} and not MTConfig.find_run_cfg(self.args.resume):
                self.mt.save(self.save_dir / "mt.yaml")
        d = self.mt.distill
        kd_dim = None
        if d.enabled:
            from adas_mt.distill import FrozenTeacher

            with torch_distributed_zero_first(LOCAL_RANK):  # rank 0 downloads the weights, the others hit the cache
                self.teacher = FrozenTeacher(
                    d.teacher, pretrained=d.teacher_pretrained, checkpoint=d.teacher_ckpt,
                    input_scale=d.teacher_scale, dtype=d.teacher_dtype,
                )
            kd_dim = self.teacher.dim  # projector must exist before the optimizer / EMA / DDP are built
        model = MultiTaskModel(
            f"yolo26{scale}.yaml",
            ch=self.data.get("channels", 3),
            nc=self.data["nc"],
            da_classes=int(self.data["da_classes"]),
            ll_classes=int(self.data["ll_classes"]),
            verbose=verbose and RANK == -1,
            kd_dim=kd_dim,
        )
        if weights is not None:
            model.load(weights)
        if self.teacher is not None:
            from adas_mt.distill import Distiller

            kw = dict(w_cos=d.w_cos, w_aff=d.w_aff, weight=d.weight, weight_end=d.weight_end, every=d.every,
                      n_aff_tokens=d.n_aff_tokens)
            model.set_distiller_factory(lambda m: Distiller(m, self.teacher, **kw))
        return model

    def setup_model(self):
        ckpt = super().setup_model()
        # a Stage-A checkpoint carries the trained projector next to the (stripped) model
        if isinstance(ckpt, dict) and ckpt.get("kd_proj") is not None and hasattr(self.model, "_restore_kd"):
            self.model._restore_kd({"state": ckpt["kd_proj"], "dim": int(ckpt["kd_dim"])})
        return ckpt

    def set_model_attributes(self):
        super().set_model_attributes()  # nc, names, args
        model = self.model
        check_matches_data(model, self.data)
        model.da_names = list(self.data.get("da_names", model.da_names))
        model.ll_names = list(self.data.get("ll_names", model.ll_names))
        model.loss_gains = dict(self.mt.loss_gains)

    # ------------------------------------------------------------------- data
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        train = mode == "train"
        return MultiTaskDataset(
            img_path=img_path,
            data=self.data,
            imgsz=self.mt.imgsz,
            augment=train,
            hyp=self.args,
            batch_size=batch,
            cache=self.args.cache or False,
            rect=False,  # val at exactly the deployed (h, w); Ultralytics would pass rect=True for val
            stride=gs,
            pad=0.0,
            single_cls=self.args.single_cls or False,
            classes=self.args.classes,
            fraction=self.args.fraction if train else 1.0,
            prefix=colorstr(f"{mode}: "),
        )

    def preprocess_batch(self, batch: dict) -> dict:
        mask = batch.get("semantic_mask")
        batch = super().preprocess_batch(batch)  # to device, /255, optional multi-scale resize of img only
        if mask is not None and batch["img"].shape[-2:] != batch["semantic_mask"].shape[-2:]:
            m = batch["semantic_mask"]
            batch["semantic_mask"] = F.interpolate(m[:, None].float(), size=batch["img"].shape[-2:], mode="nearest")[
                :, 0
            ].to(m.dtype)  # nearest keeps the packed codes intact
        return batch

    # -------------------------------------------------------------- optimizer
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        opt = super().build_optimizer(model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)
        mult = float(self.mt.head_lr_mult)
        if mult == 1.0:
            return opt
        new_ids = {id(p) for n, p in unwrap_model(model).named_parameters() if n.startswith(NEW_MODULES)}
        groups = list(opt.param_groups)
        n_boosted = 0
        for g in groups:
            moved = [p for p in g["params"] if id(p) in new_ids]
            if not moved:
                continue
            g["params"] = [p for p in g["params"] if id(p) not in new_ids]
            boosted = {k: v for k, v in g.items() if k != "params"}
            boosted["lr"] = g["lr"] * mult
            boosted["new_head"] = True  # keep `param_group` unchanged: warmup / weight-decay rescale key on it
            opt.add_param_group({"params": moved, **boosted})
            n_boosted += len(moved)
        opt.param_groups[:] = [g for g in opt.param_groups if g["params"]]  # drop groups that became empty
        if RANK in {-1, 0}:
            LOGGER.info(f"{colorstr('optimizer:')} {n_boosted} new-head tensors at {mult}x LR (da_head, ll_head, kd_proj)")
        return opt

    def _setup_train(self):
        super()._setup_train()
        # Ultralytics builds the initial `metrics` dict (and so the results.csv header) from DetMetrics.keys only.
        # A run that skips validation in some epochs would write a short header and later long rows, which breaks
        # the CSV reader and results.png; pre-seed the segmentation columns in their final order.
        if RANK in {-1, 0} and isinstance(self.metrics, dict):
            for k in self.validator.seg_keys:
                self.metrics.setdefault(k, 0.0)

    # --------------------------------------------------------------- validator
    def get_validator(self):
        m = unwrap_model(self.model)
        return MultiTaskValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks, mt=self.mt,
            seg_names=(m.da_names, m.ll_names, m.da_classes, m.ll_classes),  # declares the results.csv columns up front
        )

