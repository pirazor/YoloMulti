from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adas_mt.data.masks import IGNORE, pack_masks
from adas_mt.nn import build_model
from adas_mt.nn.loss import MultiTaskLoss, masked_ce, masked_dice, masked_focal_tversky

from .test_dataset import HW as _HW  # noqa: F401  (ensures fixtures import)


def _batch(bs=2, hw=(128, 224), da_fill=1, ll_fill=1, all_ignore=False):
    H, W = hw
    da = np.zeros((H, W), np.uint8)
    ll = np.zeros((H, W), np.uint8)
    da[H // 2 :, :] = da_fill
    ll[:, W // 3 : W // 3 + 6] = ll_fill
    packed = pack_masks(da, ll, 3, 3)
    if all_ignore:
        packed[:] = IGNORE
    return {
        "img": torch.rand(bs, 3, H, W),
        "semantic_mask": torch.from_numpy(np.stack([packed] * bs)).int(),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]] * bs),
        "cls": torch.zeros(bs, 1),
        "batch_idx": torch.arange(bs).float(),
    }


def _model():
    m = build_model("n", nc=2)
    m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    return m.train()


def test_region_losses_match_cropped_reference():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 20, 30)
    target = torch.randint(0, 3, (2, 20, 30))
    masked = target.clone()
    masked[:, :, 15:] = IGNORE  # right 15 columns ignored
    crop_l, crop_t = logits[..., :15], target[..., :15]
    for fn in (masked_ce, masked_dice, masked_focal_tversky):
        assert torch.allclose(fn(logits, masked), fn(crop_l, crop_t), atol=1e-5), fn.__name__


def test_all_ignored_is_zero_with_zero_grad():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.full((2, 8, 8), IGNORE)
    for fn in (masked_ce, masked_dice, masked_focal_tversky):
        logits.grad = None
        loss = fn(logits, target)
        assert loss.item() == 0.0, fn.__name__
        loss.backward()
        assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad) == 0, fn.__name__


def test_loss_finite_and_gradients_reach_everything():
    m = _model()
    crit = MultiTaskLoss(m)
    b = _batch()
    loss, items = crit(m(b["img"]), b)
    assert loss.shape == (5,) and torch.isfinite(loss).all()
    assert {"box_loss", "cls_loss", "da_loss", "ll_loss"} <= set(items)
    loss.sum().backward()
    for name, mod in (("backbone", m.model[0]), ("neck", m.model[16]), ("da", m.da_head), ("ll", m.ll_head)):
        g = sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
        assert g > 0, f"no gradient reaches {name}"


def test_unannotated_task_gives_zero_seg_loss():
    m = _model()
    crit = MultiTaskLoss(m)
    b = _batch(all_ignore=True)
    loss, items = crit(m(b["img"]), b)
    assert items["da_loss"].item() == 0.0 and items["ll_loss"].item() == 0.0
    assert torch.isfinite(loss).all()


def test_overfits_a_tiny_batch():
    """Trainability: det + DA + lane losses all drop on a fixed batch within a few dozen steps."""
    torch.manual_seed(0)
    m = _model()
    crit = MultiTaskLoss(m)
    b = _batch(bs=2)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for step in range(40):
        loss, items = crit(m(b["img"]), b)
        opt.zero_grad()
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        cur = {k: float(v.detach()) for k, v in items.items() if k in ("da_loss", "ll_loss")} | {"total": float(loss.detach().sum())}
        first = first or cur
        last = cur
    assert last["total"] < 0.6 * first["total"], (first, last)
    assert last["da_loss"] < 0.6 * first["da_loss"], (first, last)
    assert last["ll_loss"] < 0.8 * first["ll_loss"], (first, last)


def test_region_losses_average_over_classes_present_in_the_batch():
    """An absent class has Dice ~0 whatever is predicted (constant ~1.0 term, ~0 gradient): it must not be averaged in."""
    torch.manual_seed(0)
    target = torch.zeros(2, 16, 24, dtype=torch.long)
    target[:, 8:, :] = 1  # class 2 ("alternative") absent
    logits = torch.full((2, 3, 16, 24), -6.0)
    logits[:, 0][target == 0] = 6.0
    logits[:, 1][target == 1] = 6.0  # near-perfect prediction of the present classes
    assert masked_dice(logits, target).item() < 0.01 and masked_focal_tversky(logits, target).item() < 0.05
    only_bg = torch.zeros(2, 16, 24, dtype=torch.long)
    for fn in (masked_dice, masked_focal_tversky):
        logits_g = logits.clone().requires_grad_(True)
        loss = fn(logits_g, only_bg)  # no foreground GT at all: zero, finite zero gradient, CE does the work
        assert loss.item() == 0.0
        loss.backward()
        assert torch.isfinite(logits_g.grad).all() and torch.count_nonzero(logits_g.grad) == 0
    # a present class that is predicted badly still costs (uniform uncertainty on its pixels: Dice loss ~0.5)
    bad = logits.clone()
    bad[:, 1] = -6.0
    assert masked_dice(bad, target).item() > 0.4 and masked_focal_tversky(bad, target).item() > 0.5


def test_class_id_without_a_channel_is_ignored_not_clamped():
    """Previously target 5 with C=3 was clamped onto class 2 and trained as 'dashed'."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 12, 12)
    target = torch.randint(0, 3, (2, 12, 12))
    bad = target.clone()
    bad[:, :, 6:] = 5  # ids the model has no channel for
    ignored = target.clone()
    ignored[:, :, 6:] = IGNORE
    for fn in (masked_ce, masked_dice, masked_focal_tversky):
        assert torch.allclose(fn(logits, bad), fn(logits, ignored)), fn.__name__


def test_check_matches_data_and_loss_gains():
    from adas_mt.nn.model import check_matches_data

    m = build_model("n", nc=2)
    check_matches_data(m, {"nc": 2, "da_classes": 3, "ll_classes": 3})
    with pytest.raises(ValueError, match="ll_classes"):
        check_matches_data(m, {"nc": 2, "da_classes": 3, "ll_classes": 5})
    m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    m.loss_gains = {"da": 2.0, "ll": 0.5}
    crit = m.init_criterion()
    assert (crit.w_da, crit.w_ll) == (2.0, 0.5)
