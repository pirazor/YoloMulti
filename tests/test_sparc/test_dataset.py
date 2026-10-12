"""Dataset pipeline on synthetic scenes: the virtual-camera warp, augmentation and the label transforms agree."""

import numpy as np
import pytest
import torch

from sparc.data import virtual_camera as vc
from sparc.data.dataset import (AugCfg, MixedDataset, SparcDataset, WeightedSourceSampler, class_to_order, collate,
                                order_to_class)
from sparc.data.geometry import project_ego
from sparc.data.schema import ATTR_FLIP, EL_SIGN, LANE_X_ANCHORS
from sparc.data.synthetic import AGENT_COLORS, SIGN_NAMES, make_dataset
from sparc.data.vocab import SignVocab



NAMES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")


def _check_box_colours(it, W, H):
    """The pixel at each unoccluded box centre shows its class colour (boxes follow the warp)."""
    img = it["image"].permute(1, 2, 0).numpy()
    boxes, cls = it["boxes"].numpy(), it["cls"].numpy()
    xyxy = np.stack([(boxes[:, 0] - boxes[:, 2] / 2) * W, (boxes[:, 1] - boxes[:, 3] / 2) * H,
                     (boxes[:, 0] + boxes[:, 2] / 2) * W, (boxes[:, 1] + boxes[:, 3] / 2) * H], 1)
    eb = it["el_boxes"].numpy()
    exyxy = np.stack([(eb[:, 0] - eb[:, 2] / 2) * W, (eb[:, 1] - eb[:, 3] / 2) * H,
                      (eb[:, 0] + eb[:, 2] / 2) * W, (eb[:, 1] + eb[:, 3] / 2) * H], 1) if len(eb) else np.zeros((0, 4))
    for b in boxes:
        cx, cy = int(round(b[0] * W)), int(round(b[1] * H))
        if b[2] * W < 12 or b[3] * H < 12:
            continue  # tiny far boxes: bilinear blur and JPEG dominate the centre pixel
        if len(exyxy) and ((exyxy[:, 0] <= cx) & (cx <= exyxy[:, 2]) & (exyxy[:, 1] <= cy) & (cy <= exyxy[:, 3])).any():
            continue  # a sign / light (drawn last) covers the centre of a far agent
        cover = (xyxy[:, 0] <= cx) & (cx <= xyxy[:, 2]) & (xyxy[:, 1] <= cy) & (cy <= xyxy[:, 3])
        if cover.sum() != 1:
            continue  # the centre lies inside another agent's box: an occlusion edge, ambiguous by construction
        name = NAMES[int(cls[int(np.argmax(cover))])]
        patch = img[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].reshape(-1, 3)
        colour = np.median(patch, axis=0)
        # JPEG chroma subsampling and the bilinear warp desaturate: the pixel must be closest to its own class
        # colour among the scene palette (agent colours, road, ground, sky)
        palette = {**AGENT_COLORS, "road": (85, 85, 88), "ground": (90, 140, 70), "sky": (150, 190, 240)}
        nearest_name = min(palette, key=lambda k: np.abs(colour - palette[k]).sum())
        assert nearest_name == name, (name, nearest_name, colour)

@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    make_dataset(root, n_train=8, n_val=3, seed=3, partial=True)
    return root


@pytest.fixture(scope="module")
def vocab(syn):
    return SignVocab.from_manifests([syn])


def test_vocab_ids_and_flip(vocab):
    assert vocab.names == sorted(SIGN_NAMES) and vocab.id("") == -1 and vocab.id("nope") == -1
    assert vocab.flip("syn:arrow_left") == "syn:arrow_right" and vocab.flip("syn:stop") == "syn:stop"
    assert SignVocab(["a:left_turn"]).flip("a:left_turn") == ""  # no mirrored class -> unlabelled
    assert order_to_class(-3) == 0 and order_to_class(-1) == 2 and order_to_class(1) == 3 and order_to_class(3) == 5
    assert [class_to_order(order_to_class(o)) for o in (-3, -2, -1, 1, 2, 3)] == [-3, -2, -1, 1, 2, 3]
    assert order_to_class(-99) == -1


def test_item_without_augmentation_matches_geometry(syn, vocab):
    ds = SparcDataset(syn, "train", vocab, augment=None)
    for i in range(len(ds)):
        s = ds.sample(i)
        it = ds[i]
        H, W = ds.camera.hw
        assert it["image"].shape == (3, H, W) and it["image"].dtype == torch.uint8 and it["mask"].shape == (H, W)
        np.testing.assert_allclose(it["K"].numpy(), ds.camera.K, atol=1e-4)
        np.testing.assert_allclose(it["cam2ego"].numpy(), s.cam2ego, atol=1e-6)
        _check_box_colours(it, W, H)  # boxes follow the warp
        # lanes: the lifted targets equal the generator's 3D lane at the anchors (flat ground, exact)
        if it["label_mask"]["lane_geom"]:
            assert it["lane_y"].shape[0] == len(s.lanes)
            for j, lane in enumerate(s.lanes):
                vis = it["lane_vis"][j].numpy()
                assert vis.sum() >= 10
                want = np.interp(LANE_X_ANCHORS[vis], lane.points3d[::-1, 0], lane.points3d[::-1, 1])
                np.testing.assert_allclose(it["lane_y"][j].numpy()[vis], want, atol=0.02)
                assert it["lane_attr"][j, 0].item() == order_to_class(lane.order)
        else:
            assert it["lane_y"].shape[0] == 0
        # drivable mask: a point 10 m ahead in the ego lane is "direct" when the frame has DA labels
        uv, _ = project_ego(np.array([[10.0, 0.0, 0.0]]), it["K"].numpy(), it["cam2ego"].numpy())
        u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
        code = int(it["mask"][v, u])
        if it["label_mask"]["da"]:
            assert code % 4 == 1  # da code 1 (direct) in the packed mask (da_classes = 3 -> base 4)
        else:
            assert code % 4 == 3  # unlabelled DA
        assert it["el_sign"].shape == it["el_cls"].shape and (it["el_sign"][it["el_cls"] == EL_SIGN] >= 0).all()


def test_flip_mirrors_labels(syn, vocab, monkeypatch):
    aug = AugCfg(scale=(1.0, 1.0), translate=0.0, hflip=1.0, hsv_h=0, hsv_s=0, hsv_v=0)
    ds_f = SparcDataset(syn, "train", vocab, augment=aug)
    ds_0 = SparcDataset(syn, "train", vocab, augment=None)
    H, W = ds_0.camera.hw
    for i in range(len(ds_0)):
        a, b = ds_0[i], ds_f[i]
        assert b["flipped"] and not a["flipped"]
        diff = np.abs(b["image"].numpy().astype(int) - a["image"].numpy()[:, :, ::-1].astype(int))
        assert diff.max() <= 1  # the mirrored bilinear warp rounds differently by one level in a few pixels
        assert b["K"][0, 0] > 0 and b["K"][0, 2].item() == pytest.approx(W - 1 - a["K"][0, 2].item(), abs=1e-3)
        bb = b["boxes"].numpy()
        aa = a["boxes"].numpy()
        np.testing.assert_allclose(bb[:, 0], 1.0 - aa[:, 0] - 1.0 / W, atol=1e-5)  # pixel-centre convention
        np.testing.assert_allclose(bb[:, 1:], aa[:, 1:], atol=1e-6)
        if len(a["box3d"]):
            np.testing.assert_allclose(b["box3d"][:, 6], -a["box3d"][:, 6])
        np.testing.assert_allclose(b["lane_y"].numpy(), -a["lane_y"].numpy(), equal_nan=True)
        np.testing.assert_array_equal(b["lane_vis"].numpy(), a["lane_vis"].numpy())
        for oa, ob in zip(a["lane_attr"][:, 0].tolist(), b["lane_attr"][:, 0].tolist()):
            assert class_to_order(ob) == -class_to_order(oa)
        for ea, eb in zip(a["el_attr"].tolist(), b["el_attr"].tolist()):
            assert eb == ATTR_FLIP.get(ea, ea)
        for sa, sb in zip(a["el_sign"].tolist(), b["el_sign"].tolist()):
            if sa >= 0:
                assert vocab.names[sb] == vocab.flip(vocab.names[sa])
        # a mirrored ego point projects onto the mirrored pixel
        p = np.array([[12.0, 2.0, 0.0]])
        uv_a, _ = project_ego(p, a["K"].numpy(), a["cam2ego"].numpy())
        uv_b, _ = project_ego(p * [1, -1, 1], b["K"].numpy(), b["cam2ego"].numpy())
        assert uv_b[0, 0] == pytest.approx(W - 1 - uv_a[0, 0], abs=1e-2) and uv_b[0, 1] == pytest.approx(uv_a[0, 1], abs=1e-2)


def test_scale_translate_keeps_projection_consistent(syn, vocab):
    aug = AugCfg(scale=(0.8, 1.25), translate=0.1, hflip=0.0, hsv_h=0, hsv_s=0, hsv_v=0)
    ds = SparcDataset(syn, "train", vocab, augment=aug, seed=5)
    ds0 = SparcDataset(syn, "train", vocab, augment=None)
    H, W = ds.camera.hw
    for i in range(len(ds)):
        it, base = ds[i], ds0[i]
        K = it["K"].numpy()
        assert K[0, 0] != pytest.approx(base["K"][0, 0].item(), abs=1e-3) or K[0, 2] != pytest.approx(base["K"][0, 2].item(), abs=1e-3)
        _check_box_colours(it, W, H)
        # lane targets are in the ego frame: unchanged by scale / translation
        np.testing.assert_allclose(it["lane_y"].numpy(), base["lane_y"].numpy(), equal_nan=True, atol=1e-6)
        # the packed mask followed the image: a direct-DA point projected with the new K reads 'direct'
        if it["label_mask"]["da"]:
            uv, _ = project_ego(np.array([[10.0, 0.0, 0.0]]), K, it["cam2ego"].numpy())
            u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
            if 0 <= u < W and 0 <= v < H:
                assert int(it["mask"][v, u]) % 4 == 1


def test_mixed_sampler_weights_and_ddp_slices(syn, vocab):
    a = SparcDataset(syn, "train", vocab)
    b = SparcDataset(syn, "val", vocab)
    mixed = MixedDataset([a, b], weights=[3, 1])
    assert len(mixed) == 11 and mixed.sources == ["synthetic", "synthetic"]
    s = WeightedSourceSampler(mixed, epoch_size=4000, seed=1)
    idx = list(iter(s))
    frac_a = np.mean([i < len(a) for i in idx])
    assert abs(frac_a - 0.75) < 0.02
    # every item of a source is seen once before any repeat
    first = [i for i in idx if i < len(a)][: len(a)]
    assert sorted(first) == list(range(len(a)))
    r0 = list(iter(WeightedSourceSampler(mixed, epoch_size=400, seed=1, rank=0, world_size=2)))
    r1 = list(iter(WeightedSourceSampler(mixed, epoch_size=400, seed=1, rank=1, world_size=2)))
    assert len(r0) == len(r1) == 200 and r0 + r1 == list(iter(WeightedSourceSampler(mixed, epoch_size=400, seed=1)))
    s.set_epoch(2)
    assert list(iter(s))[:20] != idx[:20]


def test_collate_shapes(syn, vocab):
    ds = SparcDataset(syn, "train", vocab)
    batch = collate([ds[0], ds[1], ds[2]])
    assert batch["image"].shape == (3, 3, 384, 640) and batch["K"].shape == (3, 3, 3) and batch["mask"].shape == (3, 384, 640)
    assert len(batch["boxes"]) == 3 and batch["cls_vocab"].shape == (3, 8) and batch["el_vocab"].shape == (3, 2)
    assert batch["label_mask"]["agents"].all() and batch["nominal"].all()
    assert batch["meta"][1]["index"] == 1


def test_augmentation_changes_between_epochs(syn, vocab):
    """The augmentation RNG must advance across epochs (a seed from torch.initial_seed() would repeat it)."""
    aug = AugCfg(scale=(0.7, 1.3), translate=0.1, hflip=0.5, hsv_h=0, hsv_s=0, hsv_v=0)
    ds = SparcDataset(syn, "train", vocab, augment=aug)
    torch.manual_seed(0)
    ks = [ds[0]["K"][0, 0].item() for _ in range(6)]
    assert len(set(round(k, 3) for k in ks)) > 1
    # anchors whose lateral offset exceeds the lane head's range are never visible
    from sparc.data.dataset import Y_LIMIT
    from sparc.nn.decoder import Y_RANGE

    assert Y_LIMIT < Y_RANGE
    for i in range(len(ds)):
        it = ds[i]
        if len(it["lane_y"]):
            assert (it["lane_y"][it["lane_vis"]].abs() <= Y_LIMIT).all()


def test_boxes_dropped_by_the_warp_leave_consistent_tables(syn, vocab):
    """A strong translation pushes boxes out of the image: every per-box table must shrink together."""
    aug = AugCfg(scale=(0.6, 0.8), translate=0.45, hflip=0.5, hsv_h=0, hsv_s=0, hsv_v=0)
    ds = SparcDataset(syn, "train", vocab, augment=aug)
    torch.manual_seed(3)
    dropped = 0
    for _ in range(4):
        for i in range(len(ds)):
            it, s = ds[i], ds.sample(i)
            n, m = len(it["boxes"]), len(it["el_boxes"])
            assert len(it["cls"]) == len(it["box3d"]) == len(it["track_id"]) == n
            assert len(it["el_cls"]) == len(it["el_attr"]) == len(it["el_sign"]) == m
            dropped += (len(s.boxes2d) - n) + (len(s.el_boxes) - m)
            assert (it["boxes"][:, 2:] > 0).all() and (it["boxes"] >= -1e-6).all() and (it["boxes"] <= 1 + 1e-6).all()
    assert dropped > 0  # the scenario really happened
