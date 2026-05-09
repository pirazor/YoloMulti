"""Forward-pass shape sanity: random tensor in, dict out with the right shapes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_forward_shapes():
    from yolov13_multitask.nn.multitask_model import YOLOv13MultiTask

    model = YOLOv13MultiTask(nc=9, da_classes=3, ll_classes=3, verbose=False)
    model.eval()
    x = torch.zeros(2, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict)
    assert {"det", "da", "ll"} <= out.keys()
    assert out["da"].shape == (2, 3, 640, 640), out["da"].shape
    assert out["ll"].shape == (2, 3, 640, 640), out["ll"].shape


def test_forward_train_mode_returns_dict():
    from yolov13_multitask.nn.multitask_model import YOLOv13MultiTask

    model = YOLOv13MultiTask(nc=9, da_classes=3, ll_classes=3, verbose=False)
    model.train()
    x = torch.zeros(1, 3, 640, 640)
    out = model(x)
    assert isinstance(out, dict)
    # Detect head in training returns a list of feature maps per FPN level.
    assert isinstance(out["det"], list)
    assert len(out["det"]) == 3
