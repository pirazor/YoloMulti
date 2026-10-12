"""Matching / losses on synthetic batches: perfect predictions give ~0, gating and vocab masking hold, gradients flow."""

import numpy as np
import pytest
import torch

from sparc.data.dataset import SparcDataset, collate
from sparc.data.synthetic import make_dataset
from sparc.data.vocab import SignVocab
from sparc.data.virtual_camera import VirtualCamera
from sparc.nn.boxes import cxcywh_to_xyxy, elementwise_giou, pairwise_giou, pairwise_iou
from sparc.nn.criterion import Criterion, LossGains, mal_loss, sign_prefix_ids
from sparc.nn.decoder import DecoderCfg
from sparc.nn.model import SparcCfg, SparcNet

CAM = VirtualCamera((192, 320))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    make_dataset(root, n_train=6, n_val=2, seed=7, partial=True)
    vocab = SignVocab.from_manifests([root])
    ds = SparcDataset(root, "train", vocab, camera=CAM, augment=None)
    return ds, vocab


def test_box_utils():
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    iou = pairwise_iou(a, a)
    assert iou[0, 0] == 1 and iou[0, 1] == pytest.approx(1 / 7)
    g = pairwise_giou(a, a)
    assert g[0, 0] == pytest.approx(1.0) and g[0, 1] == pytest.approx(1 / 7 - (9 - 7) / 9)
    assert elementwise_giou(a, a).tolist() == pytest.approx([1.0, 1.0])
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 6.0, 6.0]])
    assert pairwise_giou(b[:1], b[1:])[0, 0] < 0  # disjoint boxes: negative GIoU


def test_mal_loss_shape_and_vocab_mask():
    logits = torch.zeros(3, 4)
    q = torch.zeros(3, 4)
    q[0, 1] = 0.8
    valid = torch.ones(3, 4)
    base = mal_loss(logits, q, valid, 1.5)
    valid[:, 2] = 0  # class 2 unknown for this source: no negative loss from it
    assert mal_loss(logits, q, valid, 1.5) < base
    # a positive with quality 1 and a confident logit costs ~0
    lg = torch.full((1, 2), -10.0)
    lg[0, 0] = 10.0
    qq = torch.tensor([[1.0, 0.0]])
    assert mal_loss(lg, qq, torch.ones(1, 2), 1.5) < 1e-3


def _perfect_outputs(batch, n_sign, n_agents=200, n_elements=50, n_lanes=24):
    """Decoder outputs that reproduce the ground truth exactly (first queries), confident everywhere."""
    B = batch["image"].shape[0]
    A = batch["lane_y"][0].shape[1] if len(batch["lane_y"][0]) else 20
    out = {"logits": torch.full((B, n_agents, 8), -8.0), "boxes": torch.rand(B, n_agents, 4) * 0.2 + 0.4,
           "box3d": torch.zeros(B, n_agents, 6), "id_emb": torch.zeros(B, n_agents, 64),
           "el_logits": torch.full((B, n_elements, 2), -8.0), "el_boxes": torch.rand(B, n_elements, 4) * 0.2 + 0.4,
           "el_attr_logits": torch.zeros(B, n_elements, 13), "el_sign_logits": torch.zeros(B, n_elements, n_sign),
           "lane_y": torch.zeros(B, n_lanes, A), "lane_z": torch.zeros(B, n_lanes, A), "lane_vis_logits": torch.full((B, n_lanes, A), -8.0),
           "lane_exist_logits": torch.full((B, n_lanes), -8.0), "lane_order_logits": torch.zeros(B, n_lanes, 6),
           "lane_dir_logits": torch.zeros(B, n_lanes, 3), "lane_cont_logits": torch.zeros(B, n_lanes, 2), "lane_type_logits": torch.zeros(B, n_lanes, 6)}
    for b in range(B):
        n = len(batch["cls"][b])
        out["boxes"][b, :n] = batch["boxes"][b]
        out["logits"][b, torch.arange(n), batch["cls"][b]] = 8.0
        gt3 = batch["box3d"][b]
        ok = torch.isfinite(gt3).all(-1)
        out["box3d"][b, :n][ok] = torch.cat([gt3[ok, :1].log(), gt3[ok, 1:2].sin(), gt3[ok, 1:2].cos(), gt3[ok, 2:5].log()], -1)
        m = len(batch["el_cls"][b])
        out["el_boxes"][b, :m] = batch["el_boxes"][b]
        out["el_logits"][b, torch.arange(m), batch["el_cls"][b]] = 8.0
        for j in range(m):
            if batch["el_attr"][b][j] >= 0:
                out["el_attr_logits"][b, j, batch["el_attr"][b][j]] = 12.0
            if batch["el_sign"][b][j] >= 0:
                out["el_sign_logits"][b, j, batch["el_sign"][b][j]] = 12.0
        L = len(batch["lane_y"][b])
        if L:
            out["lane_y"][b, :L] = torch.nan_to_num(batch["lane_y"][b])
            out["lane_z"][b, :L] = torch.nan_to_num(batch["lane_z"][b])
            out["lane_vis_logits"][b, :L] = torch.where(batch["lane_vis"][b], 8.0, -8.0)
            out["lane_exist_logits"][b, :L] = 8.0
            attr = batch["lane_attr"][b]
            for j in range(L):
                for k, key in enumerate(("lane_order_logits", "lane_dir_logits", "lane_cont_logits", "lane_type_logits")):
                    if attr[j, k] >= 0:
                        out[key][b, j, attr[j, k]] = 12.0
    return out


def test_perfect_prediction_has_near_zero_query_loss(data):
    ds, vocab = data
    batch = collate([ds[i] for i in range(4)])
    crit = Criterion(sign_prefix=sign_prefix_ids(vocab.names))
    out = _perfect_outputs(batch, len(vocab))
    items = crit.query_losses(out, batch)
    for k, v in items.items():
        assert float(v) < 0.02, (k, float(v))


def test_label_gating_and_vocab_masking(data):
    ds, vocab = data
    items = [ds[i] for i in range(4)]
    batch = collate(items)
    crit = Criterion(sign_prefix=sign_prefix_ids(vocab.names))
    out = _perfect_outputs(batch, len(vocab))
    for k in out:
        out[k] = out[k].clone().requires_grad_(True)
    # a sample whose DA / lane labels are absent contributes nothing to those losses
    batch["label_mask"]["lane_geom"][:] = False
    batch["label_mask"]["agents"][:] = False
    it = {k: float(v.detach()) for k, v in crit.query_losses(out, batch).items()}
    assert it["lane_y"] == 0 and it["lane_exist"] == 0 and it["cls"] == 0 and it["l1"] == 0
    # out-of-vocab negatives get no gradient: restrict the vocab to class 2 (car) and perturb logits of class 5
    batch["label_mask"]["agents"][:] = True
    batch["cls_vocab"][:] = False
    batch["cls_vocab"][:, 2] = True
    for b in range(4):  # keep only cars so every GT is inside the vocab
        keep = batch["cls"][b] == 2
        batch["cls"][b], batch["boxes"][b], batch["box3d"][b] = batch["cls"][b][keep], batch["boxes"][b][keep], batch["box3d"][b][keep]
    out["logits"].grad = None
    it = crit.query_losses(out, batch)
    (it["cls"]).backward(retain_graph=True)
    grad = out["logits"].grad
    assert grad[..., 5].abs().sum() == 0 and grad[..., 2].abs().sum() > 0


def test_sign_loss_restricted_to_the_target_source():
    names = ["a:x", "a:y", "b:z"]
    pref = sign_prefix_ids(names)
    assert pref == [0, 0, 1]
    crit = Criterion(sign_prefix=pref)
    B = 1
    batch = {"label_mask": {"elements": torch.tensor([True]), "el_attr": torch.tensor([False]), "el_sign": torch.tensor([True])},
             "el_boxes": [torch.tensor([[0.5, 0.5, 0.1, 0.1]])], "el_cls": [torch.tensor([1])], "el_attr": [torch.tensor([-1])],
             "el_sign": [torch.tensor([0])], "el_vocab": torch.tensor([[False, True]])}
    out = {"el_logits": torch.full((B, 3, 2), -8.0), "el_boxes": torch.tensor([[[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
           "el_attr_logits": torch.zeros(B, 3, 13), "el_sign_logits": torch.zeros(B, 3, 3, requires_grad=True)}
    out["el_logits"][0, 0, 1] = 8.0
    items = {}
    crit._box_losses(out, batch, "el_", "el_logits", "el_boxes", "el_boxes", "el_cls", "el_vocab", "elements", items,
                     extra=crit._element_extras(out, batch, items, ""))
    items["el_sign"].backward()
    g = out["el_sign_logits"].grad[0, 0]
    assert g[2] == 0 and g[0] < 0 and g[1] > 0  # "b:z" is neither a positive nor a negative for an "a:" target


def test_full_criterion_with_the_network_backward(data):
    ds, vocab = data
    batch = collate([ds[i] for i in range(len(ds))])
    has_sign = any((s >= 0).any() for s in batch["el_sign"])
    cfg = SparcCfg(backbone="yolo26s", imgsz=CAM.hw, bev=False, decoder=DecoderCfg(n_sign=len(vocab), layers=2))
    net = SparcNet(cfg).train()
    crit = Criterion(sign_prefix=sign_prefix_ids(vocab.names))
    out = net(batch["image"].float() / 255, K=batch["K"], cam2ego=batch["cam2ego"])
    total, items = crit(out, batch)
    assert torch.isfinite(total) and all(np.isfinite(v) for v in items.values())
    assert {"cls", "l1", "giou", "box3d", "el_cls", "el_attr", "el_sign", "lane_y", "lane_vis", "lane_exist", "lane_attr",
            "aux0_cls", "enc_cls", "da_ce", "da_dice", "ll_ce", "ll_tversky", "total"} <= set(items)
    total.backward()
    heads = {"decoder.cls": net.decoder.cls, "decoder.el_attr": net.decoder.el_attr, "decoder.el_sign": net.decoder.el_sign,
             "decoder.lane_y": net.decoder.lane_y, "decoder.lane_exist": net.decoder.lane_exist, "decoder.box3d": net.decoder.box3d,
             "da_head": net.da_head, "lane_aux": net.lane_aux, "decoder.enc_cls": net.decoder.enc_cls, "backbone": net.backbone}
    for name, m in heads.items():
        if name == "decoder.el_sign" and not has_sign:
            continue
        gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
        assert gn > 0, name
