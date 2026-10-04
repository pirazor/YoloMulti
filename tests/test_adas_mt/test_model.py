from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from adas_mt.nn import build_model
from adas_mt.nn.model import MultiTaskModel
from adas_mt.utils.profile import profile_model


@pytest.fixture(scope="module")
def model_n():
    return build_model("n", nc=2)


@pytest.mark.parametrize("hw", [(384, 640), (128, 224)])
def test_forward_shapes_train_and_eval(model_n, hw):
    x = torch.zeros(2, 3, *hw)
    model_n.train()
    out = model_n(x)
    assert set(out["det"]) == {"one2many", "one2one"}
    for k in ("da", "ll", "da_aux"):
        assert out[k].shape == (2, 3, *hw), k
    model_n.eval()
    with torch.no_grad():
        out = model_n(x)
    det = out["det"][0] if isinstance(out["det"], tuple) else out["det"]
    assert det.shape == (2, 300, 6)  # NMS-free: [x1, y1, x2, y2, score, cls]
    assert out["da"].shape == out["ll"].shape == (2, 3, *hw) and "da_aux" not in out


def test_tap_channels_and_save_list():
    m = build_model("s", nc=9)
    assert (m.p3_layer, m.p4_layer) == (16, 19) and 2 in m.save
    assert m.da_head.lat3.conv.in_channels == 128 and m.da_head.lat4.conv.in_channels == 256
    assert m.ll_head.lat2.conv.in_channels == 128  # backbone P2


def test_non_multiple_of_four_input_is_matched():
    m = build_model("n", nc=2).eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 96, 160))
    assert out["ll"].shape[-2:] == (96, 160)


def test_lane_head_starts_as_background():
    m = build_model("n", nc=2).eval()
    with torch.no_grad():
        ll = m(torch.rand(1, 3, 128, 224))["ll"].softmax(1)
    assert ll[:, 0].mean() > 0.9  # bg prior 0.99 at init


def _ckpt(tmp_path, scale, nc=80):
    det = DetectionModel(f"yolo26{scale}.yaml", nc=nc, verbose=False)
    p = tmp_path / f"yolo26{scale}.pt"
    torch.save({"model": det, "epoch": -1}, p)
    return p, det


def test_pretrained_transfer_is_full(tmp_path):
    p, det = _ckpt(tmp_path, "s")
    m = build_model("s", nc=9, weights=p)
    assert m.transfer_ratio > 0.99
    assert torch.equal(m.model[0].conv.weight, det.model[0].conv.weight)
    assert torch.equal(m.model[13].state_dict()[next(iter(m.model[13].state_dict()))],
                       det.model[13].state_dict()[next(iter(det.model[13].state_dict()))])


def test_pretrained_scale_mismatch_fails_loudly(tmp_path):
    p, _ = _ckpt(tmp_path, "s")
    with pytest.raises(ValueError, match="scale"):
        build_model("n", nc=9, weights=p)  # scale check before load
    n = build_model("n", nc=9)
    with pytest.raises(ValueError, match="transferred"):
        n.load(torch.load(p, weights_only=False))  # direct load: ratio guard


def test_fuse_preserves_outputs():
    m = build_model("n", nc=2).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    x = torch.rand(1, 3, 128, 224)
    with torch.no_grad():
        ref = m(x)
        m.fuse(verbose=False)
        out = m(x)
    assert (ref["da"] - out["da"]).abs().max() < 1e-3
    assert (ref["ll"] - out["ll"]).abs().max() < 1e-3
    a = ref["det"][0] if isinstance(ref["det"], tuple) else ref["det"]
    b = out["det"][0] if isinstance(out["det"], tuple) else out["det"]
    assert (a[..., 4].sort(1).values - b[..., 4].sort(1).values).abs().max() < 1e-3


def test_segmentation_heads_are_cheap():
    p = profile_model(build_model("s", nc=9), (384, 640))
    seg = p["da"]["gflops"] + p["ll"]["gflops"]
    assert seg < 0.1 * p["total"]["gflops"], p  # legacy decoders were ~39 GFLOPs here (3x the detector)
    assert p["total"]["gflops"] < 20


@pytest.mark.skipif(not __import__("os").environ.get("YOLO26S_PT"), reason="set YOLO26S_PT=/path/to/yolo26s.pt")
def test_official_checkpoint_detects_bus():
    """Real pretrained weights transfer 100% and still detect through the multi-task wrapper."""
    import os

    import cv2
    import ultralytics
    from ultralytics.data.augment import LetterBox

    m = build_model("s", nc=80, weights=os.environ["YOLO26S_PT"]).eval()
    assert m.transfer_ratio > 0.99
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    lb = LetterBox((384, 640), auto=False, center=True)(image=img)
    x = torch.from_numpy(lb[..., ::-1].transpose(2, 0, 1).copy()).float()[None] / 255
    with torch.no_grad():
        det = m(x)["det"][0][0]
    cls = det[det[:, 4] > 0.4][:, 5].long().tolist()
    assert cls.count(0) >= 3 and 5 in cls  # persons + bus


def test_profile_reports_the_fused_deployed_graph():
    m = build_model("n", nc=9)
    fused = profile_model(m, (384, 640))
    unfused = profile_model(m, (384, 640), fused=False)
    assert fused["det"]["gflops"] < 0.7 * unfused["det"]["gflops"]  # one-to-many branch dropped by fuse()
    assert fused["total"]["params_M"] < unfused["total"]["params_M"]
    assert not hasattr(m.model[-1], "_fused_marker") and m.model[-1].cv2 is not None  # input model untouched
