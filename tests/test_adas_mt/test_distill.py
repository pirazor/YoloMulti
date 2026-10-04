from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from adas_mt.distill import Distiller, FrozenTeacher, distill_terms
from adas_mt.nn import build_model
from adas_mt.nn.loss import MultiTaskLoss

from .test_loss import _batch

TINY = "vit_tiny_patch16_224"  # random-init stand-in teacher (HF weights are not downloadable here)


@pytest.fixture(scope="module")
def tiny_teacher():
    torch.manual_seed(0)
    return FrozenTeacher(TINY, pretrained=False)


@pytest.mark.parametrize(
    "name,scale,hw,grid",
    [(TINY, 1.0, (96, 160), (6, 10)), (TINY, 0.5, (96, 160), (3, 5)),
     ("vit_small_patch14_dinov2", 1.0, (112, 168), (8, 12)),
     ("vit_small_patch16_dinov3", 1.0, (96, 160), (6, 10))],  # real DINOv3 arch (RoPE, register tokens)
)
def test_teacher_token_grid(name, scale, hw, grid):
    t = FrozenTeacher(name, pretrained=False, input_scale=scale)
    tokens, g = t(torch.rand(2, 3, *hw))
    assert g == grid and tokens.shape == (2, grid[0] * grid[1], t.dim) and tokens.dtype == torch.float32


def test_teacher_is_frozen_and_deterministic(tiny_teacher):
    assert not tiny_teacher.training and all(not p.requires_grad for p in tiny_teacher.parameters())
    tiny_teacher.train()
    assert not tiny_teacher.training  # train() cannot unfreeze it
    x = torch.rand(1, 3, 96, 160)
    assert torch.equal(tiny_teacher(x)[0], tiny_teacher(x)[0])


def test_distill_terms_extremes():
    t = torch.randn(2, 40, 16)
    cos, aff = distill_terms(t, t)
    assert cos.abs() < 1e-5 and aff.abs() < 1e-6
    cos, _ = distill_terms(-t, t)
    assert abs(cos.item() - 2.0) < 1e-5


def test_lambda_schedule(tiny_teacher):
    m = build_model("n", nc=2)
    d = Distiller(m, tiny_teacher, weight=1.0, weight_end=0.1)
    d.set_progress(0.0)
    assert math.isclose(d.lam, 1.0)
    d.set_progress(0.5)
    assert math.isclose(d.lam, 0.55)
    d.set_progress(1.0)
    assert math.isclose(d.lam, 0.1)


def test_gradients_reach_student_not_teacher(tiny_teacher):
    m = build_model("n", nc=2).train()
    d = Distiller(m, tiny_teacher)
    before = {k: v.clone() for k, v in tiny_teacher.state_dict().items()}
    x = torch.rand(2, 3, 128, 224)
    _, p3, p4 = m.features(x)
    loss, items = d.loss_from_feats(p3, p4, x)
    assert torch.isfinite(loss) and 0.0 < items["kd_loss"].item() < 4.0
    loss.backward()
    for name, mod in (("backbone", m.model[0]), ("neck", m.model[16]), ("projector", m.kd_proj)):
        assert sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None) > 0, name
    assert all(p.grad is None for p in tiny_teacher.parameters())
    torch.optim.SGD(m.parameters(), lr=0.1).step()
    assert all(torch.equal(v, tiny_teacher.state_dict()[k]) for k, v in before.items())


def test_every_k_skips_teacher(tiny_teacher):
    m = build_model("n", nc=2).train()
    d = Distiller(m, tiny_teacher, every=3)
    calls = []
    orig = d.teacher.forward
    d.teacher.forward = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    x = torch.rand(1, 3, 96, 160)
    _, p3, p4 = m.features(x)
    vals = [d.loss_from_feats(p3, p4, x)[0].item() for _ in range(6)]
    d.teacher.forward = orig
    assert len(calls) == 2 and [v != 0.0 for v in vals] == [True, False, False, True, False, False]


def test_integrates_with_multitask_loss(tiny_teacher):
    m = build_model("n", nc=2)
    m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    m.train()
    crit = MultiTaskLoss(m, distiller=Distiller(m, tiny_teacher))
    b = _batch()
    loss, items = crit(m(b["img"]), b)
    assert loss.shape == (6,) and torch.isfinite(loss).all() and "kd_loss" in items and "kd_cos" in items
    loss.sum().backward()
    assert m.kd_proj.net[0].weight.grad.abs().sum() > 0
    assert m.model[0].conv.weight.grad.abs().sum() > 0


def test_strip_training_only_keeps_inference_identical(tiny_teacher):
    m = build_model("n", nc=2)
    Distiller(m, tiny_teacher)
    m.eval()
    x = torch.rand(1, 3, 96, 160)
    with torch.no_grad():
        ref = m(x)
    m.strip_training_only()
    assert not any(k.startswith("kd_proj") or "da_head.aux" in k for k in m.state_dict())
    with torch.no_grad():
        out = m(x)
    assert torch.equal(ref["da"], out["da"]) and torch.equal(ref["ll"], out["ll"])
    m.train()
    assert "da_aux" not in m(x)  # training forward still works without the aux head


def test_stage_a_pretrain_improves_alignment_and_checkpoint_loads(tmp_path, synth_root, tiny_teacher):
    from adas_mt.distill.pretrain import pretrain

    torch.manual_seed(0)
    hw = (96, 160)
    model = build_model("n", nc=2)
    logs: list = []
    last = pretrain(model, tiny_teacher, synth_root / "images" / "train", hw, epochs=14, batch=4, lr=3e-3,
                    workers=0, device="cpu", save_dir=tmp_path / "stageA", amp=False, log=logs.append)
    cos = [float(l.split("cos_sim")[1].split()[0]) for l in logs]
    assert cos[-1] > cos[0] + 0.05, cos  # student features align better with the teacher
    ck = torch.load(last, weights_only=False)
    assert ck["teacher"] == TINY
    reloaded = build_model("n", nc=2, weights=last)  # Stage B starts from this checkpoint
    assert reloaded.transfer_ratio > 0.99
    assert not any(k.startswith("kd_proj") for k in ck["model"].state_dict())
