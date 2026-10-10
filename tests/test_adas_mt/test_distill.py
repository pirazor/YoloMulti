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
     ("vit_small_patch16_dinov3", 1.0, (96, 160), (6, 10)),  # real DINOv3 arch (RoPE, register tokens)
     ("vit_small_patch16_dinov3", 0.5, (384, 640), (12, 20))],  # production resolution, half-res teacher
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


def test_distill_terms_extremes_and_scale():
    torch.manual_seed(0)
    t = torch.randn(2, 40, 16) + 2.0 * torch.randn(2, 1, 16)  # correlated tokens, like real dense features
    cos, aff = distill_terms(t, t)
    assert cos.abs() < 1e-5 and aff.abs() < 1e-6
    cos, _ = distill_terms(-t, t)
    assert abs(cos.item() - 2.0) < 1e-5
    # the affinity term must be O(1) for an unrelated student (a raw MSE of cosines is ~0.02 and would vanish)
    _, aff_rand = distill_terms(torch.randn_like(t), t)
    _, aff_close = distill_terms(t + 0.1 * torch.randn_like(t), t)
    assert 0.3 < aff_rand.item() < 10 and aff_close.item() < 0.1 * aff_rand.item()


def test_lambda_schedule(tiny_teacher):
    m = build_model("n", nc=2, kd_dim=tiny_teacher.dim)
    d = Distiller(m, tiny_teacher, weight=1.0, weight_end=0.1)
    d.set_progress(0.0)
    assert math.isclose(d.lam, 1.0)
    d.set_progress(0.5)
    assert math.isclose(d.lam, 0.55)
    d.set_progress(1.0)
    assert math.isclose(d.lam, 0.1)


def test_gradients_reach_student_not_teacher(tiny_teacher):
    m = build_model("n", nc=2, kd_dim=tiny_teacher.dim).train()
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
    m = build_model("n", nc=2, kd_dim=tiny_teacher.dim).train()
    d = Distiller(m, tiny_teacher, every=3)
    calls = []
    orig = d.teacher.forward
    d.teacher.forward = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    x = torch.rand(1, 3, 96, 160)
    _, p3, p4 = m.features(x)
    vals = [d.loss_from_feats(p3, p4, x)[0].item() for _ in range(6)]
    d.teacher.forward = orig
    assert len(calls) == 2 and [v != 0.0 for v in vals] == [True, False, False, True, False, False]


def test_skipped_steps_keep_projector_in_graph(tiny_teacher):
    """DDP raises on parameters that get no gradient; a skipped (every>1) step must still touch kd_proj."""
    m = build_model("n", nc=2, kd_dim=tiny_teacher.dim)
    m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    m.train()
    crit = MultiTaskLoss(m, distiller=Distiller(m, tiny_teacher, every=2))
    b = _batch()
    for step in range(2):
        m.zero_grad(set_to_none=True)
        loss, _ = crit(m(b["img"]), b)
        loss.sum().backward()
        assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == [], f"step {step}"


def test_resume_restores_distiller_and_schedule(tiny_teacher):
    """Ultralytics resume does: criterion = model.init_criterion(); criterion.updates = k; criterion.update()."""
    import copy

    m = build_model("n", nc=2, kd_dim=tiny_teacher.dim)
    m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    m.set_distiller_factory(lambda mm: Distiller(mm, tiny_teacher, weight=1.0, weight_end=0.1))
    crit = m.init_criterion()
    assert crit.distiller is not None
    crit.updates = 4
    crit.update()
    assert crit.det.updates == 5  # inner E2ELoss schedule restored, not reset to 1
    assert math.isclose(crit.distiller.progress, 5 / 9)
    clone = copy.deepcopy(m)  # EMA / checkpoints must never drag the teacher along
    assert not hasattr(clone, "_distiller_factory") and not hasattr(clone, "criterion")


def test_integrates_with_multitask_loss(tiny_teacher):
    m = build_model("n", nc=2, kd_dim=tiny_teacher.dim)
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
    m = build_model("n", nc=2, kd_dim=tiny_teacher.dim)
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
    # ... and from the trained projector, not a fresh one (a fresh one would undo the Stage-A alignment)
    proj = reloaded.attach_kd_projector(tiny_teacher.dim)
    assert all(torch.equal(proj.state_dict()[k], v) for k, v in ck["kd_proj"].items())
    other = build_model("n", nc=2, weights=last).attach_kd_projector(128)  # different teacher width -> fresh
    assert other.net[-1].out_channels == 128


def test_projector_exists_before_optimizer_and_ema(tiny_teacher):
    """The trap: a kd_proj created lazily at the first loss call is in no optimizer, EMA or DDP wrapper."""
    from ultralytics.utils.torch_utils import ModelEMA

    m = build_model("n", nc=2, kd_dim=tiny_teacher.dim)
    names = {n for n, _ in m.named_parameters()}
    assert any(n.startswith("kd_proj") for n in names)
    assert any(k.startswith("kd_proj") for k in ModelEMA(m).ema.state_dict())
    late = build_model("n", nc=2)  # no kd_dim -> must fail loudly instead of registering a projector too late
    with pytest.raises(RuntimeError, match="kd_proj"):
        Distiller(late, tiny_teacher)


def test_stage_a_projector_restored_at_construction(tmp_path, synth_root, tiny_teacher):
    from adas_mt.distill.pretrain import pretrain

    last = pretrain(build_model("n", nc=2), tiny_teacher, synth_root / "images" / "train", (96, 160), epochs=1,
                    batch=4, workers=0, device="cpu", save_dir=tmp_path, amp=False, log=lambda *_: None)
    ck = torch.load(last, weights_only=False)
    b = build_model("n", nc=2, weights=last, kd_dim=tiny_teacher.dim)  # trainer path: kd_proj built, then restored
    assert all(torch.equal(b.kd_proj.state_dict()[k], v) for k, v in ck["kd_proj"].items())


def test_local_teacher_checkpoint_accepts_meta_and_timm_formats_and_refuses_mismatches(tmp_path):
    """--teacher_ckpt is the offline path; the file an offline user gets is Meta's dinov3_*.pth with its own key names
    (storage_tokens, blocks.N.ls1.gamma, rope_embed.periods, mask_token), which timm renames only on hub download."""
    import timm

    from adas_mt.distill import FrozenTeacher

    name = "vit_small_patch16_dinov3"
    torch.manual_seed(1)
    ref = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.01 * torch.randn_like(p))  # non-default register token / layer scales: a lost tensor is visible
    sd = ref.state_dict()
    meta = {k.replace("reg_token", "storage_tokens").replace("gamma_1", "ls1.gamma").replace("gamma_2", "ls2.gamma"): v.clone()
            for k, v in sd.items()}
    meta["rope_embed.periods"] = torch.ones(16)
    meta["mask_token"] = torch.zeros(1, ref.num_features)
    torch.save(meta, tmp_path / "meta.pth")
    torch.save(sd, tmp_path / "timm.pth")
    for f in ("meta.pth", "timm.pth"):
        t = FrozenTeacher(name, pretrained=False, checkpoint=tmp_path / f)
        got = t.model.state_dict()
        assert all(torch.equal(got[k], sd[k]) for k in sd), f
    partial = dict(sd)
    del partial["reg_token"]  # would silently train against a random register token before
    torch.save(partial, tmp_path / "partial.pth")
    with pytest.raises(ValueError, match="reg_token"):
        FrozenTeacher(name, pretrained=False, checkpoint=tmp_path / "partial.pth")
    torch.save(timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0).state_dict(), tmp_path / "other.pth")
    with pytest.raises(ValueError, match="does not match"):
        FrozenTeacher(name, pretrained=False, checkpoint=tmp_path / "other.pth")


def test_stage_a_image_discovery_skips_masks_and_val(synth_root):
    from adas_mt.distill.pretrain import find_images

    found = find_images(synth_root)  # a converted dataset ROOT: only images/train may be used
    assert found and all("images/train" in str(p) for p in found) and not any(".png" in p.suffix for p in found)
    assert len(find_images(synth_root / "images" / "train")) == len(found)
