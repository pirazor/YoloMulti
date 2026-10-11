"""Trainer on synthetic data (CPU): the loop runs end to end, losses fall, metrics and checkpoints are produced,
resume continues; the opt-in learning test (ADAS_SLOW_TESTS=1) checks that the model actually learns the scenes."""

import csv
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from sparc.cli import main
from sparc.data.synthetic import make_dataset
from sparc.engine.config import DataCfg, TrainCfg
from sparc.engine.evaluators import DetEvaluator, LaneEvaluator, match_tp
from sparc.engine.trainer import Trainer, fitness_of, load_for_eval

CFG = Path(__file__).parents[2] / "sparc" / "cfg" / "smoke.yaml"


def _manifest(tmp, root, limit=None):
    m = tmp / "data.yaml"
    entry = {"root": str(root)}
    if limit:
        entry["limit"] = limit
    m.write_text(yaml.safe_dump({"train": [{**entry, "weight": 1.0}], "val": [entry]}))
    return m


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    make_dataset(root, n_train=8, n_val=2, seed=11, partial=False)
    return root


def test_config_loading_is_strict(tmp_path):
    cfg = TrainCfg.load(CFG, {"epochs": 2, "model.backbone": "yolo26s"})
    assert cfg.epochs == 2 and cfg.model.backbone == "yolo26s" and cfg.aug.scale == [0.9, 1.1]
    bad = tmp_path / "bad.yaml"
    bad.write_text("epochs: 1\nlearning_rate: 3\n")
    with pytest.raises(ValueError, match="unknown keys"):
        TrainCfg.load(bad)
    bad.write_text("model: {backbones: x}\n")
    with pytest.raises(ValueError, match="unknown keys"):
        TrainCfg.load(bad)


def test_evaluator_primitives():
    pb = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60.0]]).numpy()
    gb = torch.tensor([[0, 0, 10, 10], [21, 21, 31, 31.0]]).numpy()
    tp = match_tp(pb, torch.tensor([0, 0, 1]).numpy(), gb, torch.tensor([0, 0]).numpy())
    assert tp[0].all() and tp[1, 0] and not tp[1, -1] and not tp[2].any()
    ev = DetEvaluator(["a", "b"])
    ev.update(pb, [0.9, 0.8, 0.7], [0, 0, 1], gb, [0, 0])
    r = ev.results("x_")
    assert r["x_mAP50"] > 0.99 and 0 < r["x_mAP50-95"] < 1  # Ultralytics interpolates the PR curve (101 points)
    assert fitness_of({"s1/agents_mAP50-95": 0.5, "s2/agents_mAP50-95": 0.7, "s1/lane_F1": 0.9}, {"agents_mAP50-95": 1, "lane_F1": 1, "da_mIoU": 1}) == pytest.approx((0.6 + 0.9) / 2)


def test_smoke_training_runs_resumes_and_evaluates(syn, tmp_path):
    out = tmp_path / "run"
    cfg = TrainCfg.load(CFG, {"epochs": 2, "epoch_size": 8, "batch": 4, "workers": 0, "log_every": 1})
    data = DataCfg.load(_manifest(tmp_path, syn))
    tr = Trainer(cfg, data, out, device="cpu")
    assert len(tr.vocab) == 5 and tr.steps_per_epoch == 2
    last = tr.fit()
    assert tr.epoch == 2 and tr.step == 4
    assert (out / "weights" / "last.pt").exists() and (out / "weights" / "best.pt").exists() and (out / "signs.json").exists()
    rows = list(csv.DictReader(open(out / "results.csv")))
    assert len(rows) == 2 and "synthetic/agents_mAP50-95" in rows[0] and "synthetic/lane_F1" in rows[0] and "synthetic/da_mIoU" in rows[0]
    assert "synthetic/light_color_acc" in last or "synthetic/sign_top1" in last or "synthetic/el_recall50" in last
    assert all(k in last for k in ("train/total", "train/cls", "train/lane_y", "train/da_ce", "fitness"))
    # resume: two more epochs continue the step count and keep the history
    cfg2 = TrainCfg.load(CFG, {"epochs": 3, "epoch_size": 8, "batch": 4, "workers": 0, "log_every": 1})
    tr2 = Trainer(cfg2, data, out, device="cpu", resume=out / "weights" / "last.pt")
    assert tr2.epoch == 2 and tr2.step == 4 and len(tr2.history) == 2
    tr2.fit()
    assert tr2.epoch == 3 and tr2.step == 6 and len(list(csv.DictReader(open(out / "results.csv")))) == 3
    # the checkpoint loads for evaluation and the val CLI runs
    model, vocab, c = load_for_eval(out / "weights" / "best.pt")
    assert vocab.names == tr.vocab.names and c.model.backbone == "yolo26s"
    assert main(["val", "--weights", str(out / "weights" / "best.pt"), "--data", str(tmp_path / "data.yaml"),
                 "--out", str(tmp_path / "valrun"), "--device", "cpu", "--workers", "0"]) == 0
    res = json.loads((tmp_path / "valrun" / "val.json").read_text())
    assert "fitness" in res and "synthetic/lane_F1" in res


def test_train_cli_and_synth_cli(tmp_path):
    root = tmp_path / "syn"
    assert main(["synth", "--", "--dst", str(root), "--n-train", "4", "--n-val", "2", "--no-partial"]) == 0
    m = _manifest(tmp_path, root)
    assert main(["train", "--cfg", str(CFG), "--data", str(m), "--out", str(tmp_path / "run"), "--epochs", "1",
                 "--epoch-size", "4", "--batch", "2", "--workers", "0", "--device", "cpu"]) == 0
    assert (tmp_path / "run" / "results.csv").exists()


@pytest.mark.skipif(not os.environ.get("ADAS_SLOW_TESTS"), reason="set ADAS_SLOW_TESTS=1 (several minutes on CPU)")
def test_learning_on_synthetic_scenes(tmp_path):
    root = tmp_path / "syn"
    make_dataset(root, n_train=48, n_val=8, seed=5, partial=False)
    cfg = TrainCfg.load(CFG, {"epochs": 25, "epoch_size": 48, "batch": 8, "workers": 2, "log_every": 3, "lr": 4e-4,
                              "warmup_iters": 30})
    tr = Trainer(cfg, DataCfg.load(_manifest(tmp_path, root)), tmp_path / "run", device="cpu")
    last = tr.fit()
    first = tr.history[0]
    assert last["train/total"] < 0.5 * first["train/total"]
    assert last["synthetic/da_mIoU"] > 0.5, last
    assert last["synthetic/lane_F1"] > 0.5, last
    assert last["synthetic/agents_mAP50"] > 0.3, last
