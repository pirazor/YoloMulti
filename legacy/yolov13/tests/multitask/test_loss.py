"""Loss returns finite values and gradients flow on a synthetic batch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _make_batch(bs: int = 2, H: int = 640, W: int = 640):
    img = torch.zeros(bs, 3, H, W)
    bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]] * bs, dtype=torch.float32)
    cls = torch.zeros(bs, 1, dtype=torch.float32)
    batch_idx = torch.arange(bs, dtype=torch.float32)
    return {
        "img": img,
        "bboxes": bboxes,
        "cls": cls,
        "batch_idx": batch_idx,
        "da_mask": torch.zeros(bs, H, W, dtype=torch.long),
        "ll_mask": torch.zeros(bs, H, W, dtype=torch.long),
        "has_da": torch.tensor([True] * bs),
        "has_ll": torch.tensor([True] * bs),
    }


def test_loss_finite_and_grads():
    from yolov13_multitask.loss.multitask_loss import MultiTaskLoss
    from yolov13_multitask.nn.multitask_model import YOLOv13MultiTask

    model = YOLOv13MultiTask(nc=2, da_classes=3, ll_classes=3, verbose=False)
    model.train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, loss_weights={"det": 1.0, "da": 1.0, "ll": 1.0})
    crit = MultiTaskLoss(model)
    batch = _make_batch()
    preds = model._predict_once(batch["img"])
    total, items = crit(preds, batch)
    assert torch.isfinite(total).item()
    assert items.numel() == 5
    total.backward()
    # at least one decoder param has a grad
    grad_norms = [p.grad.detach().abs().sum().item() for p in model.da_decoder.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms)
