"""End-to-end smoke tests of the Ultralytics trainer subclass on the colour-coded synthetic set (CPU)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import yaml

from adas_mt.engine import MTConfig, MultiTaskTrainer

from .conftest import make_split

HW = (96, 160)  # tiny, still a multiple of 32


def _root(tmp_path: Path) -> Path:
    make_split(tmp_path, "train", 8, seed=0)
    make_split(tmp_path, "val", 4, seed=1)
    data = {"path": str(tmp_path), "train": "images/train", "val": "images/val", "nc": 2, "names": ["red", "green"],
            "da_classes": 3, "da_names": ["background", "direct", "alternative"],
            "ll_classes": 3, "ll_names": ["background", "solid", "dashed"]}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(data))
    return tmp_path


def _overrides(root: Path, **kw):
    base = dict(model="yolo26n.yaml", data=str(root / "data.yaml"), epochs=2, batch=4, workers=0, device="cpu", amp=False,
                plots=False, close_mosaic=1, optimizer="AdamW", lr0=0.002, project=str(root / "runs"), name="exp",
                exist_ok=True, warmup_epochs=0, pretrained=False, cache=False, seed=0, val=True)
    base.update(kw)
    return base


def test_trains_validates_and_saves_with_distillation(tmp_path):
    root = _root(tmp_path)
    mt = MTConfig.from_dict({"imgsz": list(HW), "scale": "n", "distill": {
        "enabled": True, "teacher": "vit_tiny_patch16_224", "teacher_pretrained": False, "weight": 1.0}})
    t = MultiTaskTrainer(overrides=_overrides(root), mt=mt)
    t.train()
    # distillation ran: kd items logged, projector trained (in optimizer groups) and saved
    assert "kd_loss" in t.loss_names and "da_loss" in t.loss_names and "ll_loss" in t.loss_names
    ck = torch.load(t.best, weights_only=False)  # final_eval stripped it: the EMA weights are under "model"
    assert any(k.startswith("kd_proj") for k in ck["model"].state_dict()), "projector missing from the checkpoint"
    assert isinstance(ck["model"].da_names, list) and ck["model"].ll_names[1] == "solid"
    # the optimizer really contains the new modules at a boosted LR
    new = [g for g in t.optimizer.param_groups if str(g.get("param_group", "")).endswith("_new")]
    base = [g for g in t.optimizer.param_groups if not any(g is n for n in new)]
    assert new and base
    assert all(abs(g["initial_lr"] - 3.0 * base[0]["initial_lr"]) < 1e-9 for g in new)  # boosted vs the trunk groups
    assert all(g["initial_lr"] == base[0]["initial_lr"] for g in base)
    ids = {id(p) for g in new for p in g["params"]}
    assert {id(p) for n, p in t.model.named_parameters() if n.startswith(("da_head", "ll_head", "kd_proj"))} == ids
    # validation produced all three task metrics
    for k in ("metrics/mAP50-95(B)", "metrics/da_mIoU", "metrics/ll_IoU_fg", "metrics/da_IoU_fg"):
        assert k in t.metrics, (k, list(t.metrics))
    assert (t.save_dir / "mt.yaml").is_file() and (t.save_dir / "results.csv").is_file()
    assert os.environ.get("ADAS_MT_CFG") is None  # env handoff restored after train()


def test_multi_scale_resizes_masks_with_images(tmp_path):
    root = _root(tmp_path)
    t = MultiTaskTrainer(overrides=_overrides(root, multi_scale=0.5), mt=MTConfig.from_dict({"imgsz": list(HW), "scale": "n"}))
    t.model = __import__("adas_mt.nn", fromlist=["build_model"]).build_model("n", nc=2)
    ds = t.build_dataset(str(root / "images" / "train"), "train", 4)
    batch = ds.collate_fn([ds[i] for i in range(4)])
    t.device = torch.device("cpu")
    t.stride = 32
    shapes = set()
    for _ in range(12):
        b = t.preprocess_batch({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()})
        assert b["img"].shape[-2:] == b["semantic_mask"].shape[-2:], "mask not resized with the image"
        assert b["img"].shape[-2] % 32 == 0 and b["img"].shape[-1] % 32 == 0
        assert set(b["semantic_mask"].unique().tolist()) <= set(batch["semantic_mask"].unique().tolist())  # codes intact
        shapes.add(tuple(b["img"].shape[-2:]))
    assert len(shapes) > 1  # multi-scale really varied the size


def test_val_dataset_is_the_deployed_geometry_not_rect(tmp_path):
    root = _root(tmp_path)
    t = MultiTaskTrainer(overrides=_overrides(root), mt=MTConfig.from_dict({"imgsz": list(HW), "scale": "n"}))
    t.model = __import__("adas_mt.nn", fromlist=["build_model"]).build_model("n", nc=2)
    ds = t.build_dataset(str(root / "images" / "val"), "val", 4)
    s = ds[0]
    assert s["img"].shape == (3, *HW) and s["semantic_mask"].shape == HW and not ds.rect


def test_mt_config_validation_and_resolution(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="multiples of 32"):
        MTConfig.from_dict({"imgsz": [100, 200]})
    with pytest.raises(ValueError, match="unknown mt config keys"):
        MTConfig.from_dict({"da_gain": 2.0})  # the typo-class that Ultralytics' get_cfg would reject elsewhere
    p = MTConfig.from_dict({"imgsz": [96, 160], "head_lr_mult": 5.0}).save(tmp_path / "run" / "mt.yaml")
    monkeypatch.setenv("ADAS_MT_CFG", str(p))
    assert MTConfig.resolve().head_lr_mult == 5.0  # what a DDP worker sees
    monkeypatch.delenv("ADAS_MT_CFG")
    assert MTConfig.resolve().head_lr_mult == 3.0
    assert MTConfig.resolve(resume=tmp_path / "run" / "weights" / "last.pt").head_lr_mult == 5.0  # resume finds mt.yaml


def test_ddp_worker_rebuilds_the_trainer_with_mt_settings(tmp_path):
    """Mimic what a DDP worker does (ultralytics.utils.dist.generate_ddp_file): rebuild from vars(args) only."""
    import shutil

    from ultralytics.utils import DEFAULT_CFG_DICT
    from ultralytics.utils.dist import generate_ddp_file
    from ultralytics.utils.patches import torch_load

    root = _root(tmp_path)
    mt = MTConfig.from_dict({"imgsz": list(HW), "scale": "n", "head_lr_mult": 7.0})
    t = MultiTaskTrainer(overrides=_overrides(root), mt=mt)
    t.world_size = 2  # as with device="0,1"
    seen = {}

    def fake_base_train(self):
        """What Ultralytics does: wipe save_dir, write the worker script, run workers (parent waits for them)."""
        shutil.rmtree(self.save_dir)
        seen["env"] = os.environ["ADAS_MT_CFG"]
        script = generate_ddp_file(self)
        assert Path(seen["env"]).is_file(), "hand-off file was deleted together with save_dir"
        state = torch_load(script.replace(".py", ".pt"), map_location="cpu")  # the worker's first step
        assert state["trainer"] is MultiTaskTrainer
        cfg = DEFAULT_CFG_DICT.copy()
        cfg.update(save_dir="")
        # the worker is a new process: only the inherited env var and vars(args) reach it
        seen["worker"] = state["trainer"](cfg=cfg, overrides=state["args"], _callbacks=state["callbacks"])

    from ultralytics.models.yolo.detect import DetectionTrainer

    orig = DetectionTrainer.train
    DetectionTrainer.train = fake_base_train
    try:
        t.train()
    finally:
        DetectionTrainer.train = orig
    worker = seen["worker"]
    assert worker.mt.head_lr_mult == 7.0 and tuple(worker.mt.imgsz) == HW and worker.args.nms is False
    assert os.environ.get("ADAS_MT_CFG") is None and not Path(seen["env"]).exists()  # parent cleaned up afterwards


def test_resume_keeps_schedule_distillation_and_mt_cfg(tmp_path):
    root = _root(tmp_path)
    mt = {"imgsz": list(HW), "scale": "n", "distill": {"enabled": True, "teacher": "vit_tiny_patch16_224",
                                                         "teacher_pretrained": False}}
    t = MultiTaskTrainer(overrides=_overrides(root, epochs=3), mt=MTConfig.from_dict(mt))

    def crash(trainer):  # after epoch 0's last.pt is written, like a killed job (a clean stop would strip last.pt)
        raise RuntimeError("simulated crash")

    t.add_callback("on_model_save", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        t.train()
    last = t.wdir / "last.pt"
    assert torch.load(last, weights_only=False)["epoch"] == 0
    r = MultiTaskTrainer(overrides=dict(resume=str(last), exist_ok=True, device="cpu", workers=0))
    assert r.mt.distill.enabled and r.mt.distill.teacher == "vit_tiny_patch16_224"  # mt.yaml next to the checkpoint
    r.train()
    crit = r.model.criterion
    assert crit.distiller is not None, "distillation dropped after resume"
    assert crit.det.updates >= 3, f"E2ELoss schedule restarted (updates={crit.det.updates})"
    assert r.epoch + 1 == 3


def test_cli_train_and_val_end_to_end(tmp_path):
    from adas_mt.cli import main

    root = _root(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"train": dict(imgsz=160, plots=False, amp=False, warmup_epochs=0, close_mosaic=1,
                                                  pretrained=False, exist_ok=True),
                                   "mt": {"imgsz": list(HW), "scale": "n"}}))
    rc = main(["train", "--data", str(root / "data.yaml"), "--model", "yolo26n.yaml", "--cfg", str(cfg), "--epochs", "1",
               "--batch", "4", "--workers", "0", "--device", "cpu", "--project", str(root / "runs"), "--name", "cli"])
    assert rc == 0
    best = root / "runs" / "cli" / "weights" / "best.pt"
    assert best.is_file() and (root / "runs" / "cli" / "mt.yaml").is_file()
    assert main(["val", "--weights", str(best), "--data", str(root / "data.yaml"), "--cfg", str(cfg), "--batch", "4",
                 "--device", "cpu"]) == 0
    assert main(["profile", "--scale", "n", "--imgsz", "96", "160"]) == 0


@pytest.mark.skipif(not os.environ.get("ADAS_SLOW_TESTS"), reason="set ADAS_SLOW_TESTS=1 (about 2 minutes on CPU)")
def test_trainer_actually_learns_all_three_tasks(tmp_path):
    """From random init, through the real trainer + validator, on the colour-coded toy set (val == train)."""
    root = _root(tmp_path)
    data = yaml.safe_load((root / "data.yaml").read_text())
    data["val"] = "images/train"
    (root / "data.yaml").write_text(yaml.safe_dump(data))
    t = MultiTaskTrainer(
        overrides=_overrides(root, epochs=100, close_mosaic=100, lr0=0.004, warmup_epochs=1, mosaic=0.0, hsv_h=0.0,
                             hsv_s=0.0, hsv_v=0.0, fliplr=0.0, translate=0.0, scale=0.0, patience=1000),
        mt=MTConfig.from_dict({"imgsz": [64, 96], "scale": "n"}),
    )
    t.train()
    m = t.metrics
    assert m["metrics/mAP50(B)"] > 0.9, m
    assert m["metrics/da_IoU_fg"] > 0.6 and m["metrics/ll_IoU_fg"] > 0.6 and m["metrics/ll_recall_fg"] > 0.6, m


def test_starts_from_a_stage_a_checkpoint_with_its_projector(tmp_path, synth_root):
    """setup_model path: model=<Stage-A last.pt>; the trained projector must reach the new model."""
    from adas_mt.distill import FrozenTeacher
    from adas_mt.distill.pretrain import pretrain
    from adas_mt.nn import build_model

    torch.manual_seed(0)
    teacher = FrozenTeacher("vit_tiny_patch16_224", pretrained=False)
    last = pretrain(build_model("n", nc=2), teacher, synth_root / "images" / "train", HW, epochs=2, batch=4, workers=0,
                    device="cpu", save_dir=tmp_path / "A", amp=False, log=lambda *_: None)
    ck = torch.load(last, weights_only=False)
    root = _root(tmp_path / "ds")
    mt = {"imgsz": list(HW), "scale": "n", "distill": {"enabled": True, "teacher": "vit_tiny_patch16_224", "teacher_pretrained": False}}
    t = MultiTaskTrainer(overrides=_overrides(root, model=str(last), pretrained=True), mt=MTConfig.from_dict(mt))
    t.model = str(last)
    t.setup_model()
    assert t.model.transfer_ratio > 0.99 and t.model.yaml["scale"] == "n"
    assert all(torch.allclose(t.model.kd_proj.state_dict()[k].float(), v.float()) for k, v in ck["kd_proj"].items())


def test_plots_enabled_run(tmp_path):
    root = _root(tmp_path)
    t = MultiTaskTrainer(overrides=_overrides(root, epochs=1, plots=True, close_mosaic=0),
                         mt=MTConfig.from_dict({"imgsz": list(HW), "scale": "n"}))
    t.train()
    assert any(t.save_dir.glob("train_batch*.jpg")) and any(t.save_dir.glob("val_batch*.jpg"))


@pytest.mark.skipif(not os.environ.get("YOLO26S_PT"), reason="set YOLO26S_PT=/path/to/yolo26s.pt")
def test_trains_from_the_official_checkpoint(tmp_path):
    root = _root(tmp_path)
    t = MultiTaskTrainer(overrides=_overrides(root, model=os.environ["YOLO26S_PT"], epochs=1, pretrained=True, close_mosaic=0),
                         mt=MTConfig.from_dict({"imgsz": list(HW)}))  # scale comes from the checkpoint, not mt.scale
    t.train()
    assert t.model.yaml["scale"] == "s" and t.model.transfer_ratio > 0.99


@pytest.mark.parametrize("opt", ["SGD", "MuSGD"])
def test_boosted_head_groups_with_other_optimizers(tmp_path, opt):
    """The new-head LR boost re-groups parameters after Ultralytics built the optimizer; check SGD and MuSGD too."""
    root = _root(tmp_path)
    t = MultiTaskTrainer(overrides=_overrides(root, epochs=1, optimizer=opt, lr0=0.01, close_mosaic=0),
                         mt=MTConfig.from_dict({"imgsz": list(HW), "scale": "n", "head_lr_mult": 4.0}))
    t.train()
    groups = t.optimizer.param_groups
    new = [g for g in groups if str(g.get("param_group", "")).endswith("_new")]
    assert new and all(g["params"] for g in groups)  # no empty groups left behind
    assert all(abs(g["initial_lr"] - 4.0 * min(x["initial_lr"] for x in groups)) < 1e-9 for g in new)
    head_ids = {id(p) for n, p in t.model.named_parameters() if n.startswith(("da_head", "ll_head"))}
    assert head_ids <= {id(p) for g in new for p in g["params"]}
    assert t.metrics and all(v == v for v in t.metrics.values())  # no NaN metric after a step with the regrouped optimizer
