import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc.data.schema import ground_cam2ego
from sparc.nn.backbone import BACKBONES, build_backbone
from sparc.nn.bev import BEVGrid, DepthSplat, IPMSampler
from sparc.nn.deform_attn import MSDeformAttn, ms_deform_attn
from sparc.nn.model import SparcCfg, SparcDeploy, SparcNet, example_inputs
from sparc.nn.stereo import StereoHead, convex_upsample, gwc_volume

H, W = 384, 640
K = torch.tensor([[[560.0, 0, 320], [0, 560, 192], [0, 0, 1]]])


@pytest.mark.parametrize("name", list(BACKBONES))
def test_backbone_contract(name):
    b = build_backbone(name, (128, 192)).eval()
    with torch.no_grad():
        o = b(torch.zeros(1, 3, 128, 192))
    assert o["p2"].shape == (1, 64, 32, 48)
    for k, s, c in zip(("p3", "p4", "p5"), (8, 16, 32), b.out_channels):
        assert o[k].shape == (1, c, 128 // s, 192 // s), (k, o[k].shape)


def test_backbone_rejects_bad_size():
    with pytest.raises(ValueError, match="multiples of 32"):
        build_backbone("vit_t16_dinov3", (380, 640))


def test_deform_attn_gather_fallback_matches_grid_sample():
    torch.manual_seed(0)
    shapes = [(12, 20), (6, 10), (3, 5)]
    s = sum(h * w for h, w in shapes)
    v = torch.randn(2, s, 4, 8)
    loc = torch.rand(2, 7, 4, 3, 4, 2) * 1.2 - 0.1  # some samples outside the map (zero padding)
    a = torch.rand(2, 7, 4, 3, 4).softmax(-1)
    ref = ms_deform_attn(v, shapes, loc, a, "grid_sample")
    alt = ms_deform_attn(v, shapes, loc, a, "gather")
    torch.testing.assert_close(alt, ref, atol=1e-5, rtol=1e-5)


def test_deform_attn_box_and_point_refs():
    m = MSDeformAttn(64, 4, 2, 3)
    shapes = [(8, 8), (4, 4)]
    mem = torch.randn(1, 80, 64)
    q = torch.randn(1, 5, 64)
    assert m(q, torch.rand(1, 5, 4), mem, shapes).shape == (1, 5, 64)
    assert m(q, torch.rand(1, 5, 2), mem, shapes).shape == (1, 5, 64)


def _ground_depth(T, Kc=K, h=H, w=W, max_z=40.0):
    """Depth image of the flat ground for camera pose T (cam2ego)."""
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    ray = torch.stack([(u - Kc[0, 0, 2]) / Kc[0, 0, 0], (v - Kc[0, 1, 2]) / Kc[0, 1, 1], torch.ones_like(u)])
    R, t = torch.as_tensor(T[:3, :3], dtype=torch.float32), torch.as_tensor(T[:3, 3], dtype=torch.float32)
    rz = (R @ ray.view(3, -1))[2].view(h, w)  # ego z of the ray direction
    d = -t[2] / rz
    return torch.where((rz < -1e-6) & (d < max_z), d, torch.zeros_like(d))[None, None]


def test_depth_splat_points_land_in_their_cells():
    T = torch.tensor(ground_cam2ego(1.5, 0.05), dtype=torch.float32)[None]
    splat = DepthSplat(c_in=4, c=4)
    depth = torch.zeros(1, 1, H, W)
    # one 4x4 pixel block with a known depth: a point 12 m ahead, 2 m left on the ground
    pt = torch.tensor([12.0, 2.0, 0.0])
    pc = T[0, :3, :3].T @ (pt - T[0, :3, 3])
    u = (K[0, 0, 0] * pc[0] / pc[2] + K[0, 0, 2]).item()
    v = (K[0, 1, 1] * pc[1] / pc[2] + K[0, 1, 2]).item()
    i, j = int(v // 4), int(u // 4)
    depth[..., i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = pc[2]
    depth[..., i * 4, j * 4] = 0  # an invalid pixel in the block does not bias the mean
    idx = splat.cell_index(depth, K, T, (H // 4, W // 4))
    g = BEVGrid()
    hh, ww = g.hw
    hit = (idx < hh * ww).nonzero()
    assert len(hit) == 1 and hit[0, 1].item() == i * (W // 4) + j
    r, c = divmod(idx[0, i * (W // 4) + j].item(), ww)
    x, y = g.x_max - (r + 0.5) * g.res, g.y_max - (c + 0.5) * g.res
    assert abs(x - 12.0) < 0.3 and abs(y - 2.0) < 0.2  # one cell + the 4-px quantisation of the point
    # mean splat: a constant feature arrives unchanged (one point per feature sub-cell), count channel = log1p(1)
    splat.reduce = nn.Identity()
    out = splat(torch.full((1, 4, H // 8, W // 8), 3.0), depth, K, T)
    assert out.shape == (1, 5, hh, ww)
    assert out[0, :4, r, c].tolist() == [3.0] * 4 and out[0, 4, r, c].item() == pytest.approx(math.log(2))
    assert out[0, :4].abs().sum() == pytest.approx(12.0)


def test_ipm_reproduces_a_ground_checkerboard():
    """Render the ground (a 1 m checkerboard) into the image, IPM-sample it back: the BEV shows the checkerboard."""
    T = ground_cam2ego(1.5, 0.08)
    Tt = torch.tensor(T, dtype=torch.float32)[None]
    depth = _ground_depth(T)
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    d = depth[0, 0]
    p = torch.stack([(u - 320) / 560 * d, (v - 192) / 560 * d, d]).view(3, -1)
    e = (Tt[0, :3, :3] @ p + Tt[0, :3, 3:4]).view(3, H, W)
    board = ((torch.floor(e[0]) + torch.floor(e[1])) % 2).float() * (d > 0)
    ipm = IPMSampler(c_in=1, c=1, stride=1)
    ipm.reduce = nn.Identity()
    out = ipm(board[None, None], K, Tt)[0]
    g = BEVGrid()
    c = g.centers().view(3, *g.hw)
    want = (torch.floor(c[0]) + torch.floor(c[1])) % 2
    valid = out[1] > 0
    # compare away from square edges (bilinear blur) and within 25 m (pixel footprint grows with range)
    inner = ((c[0] % 1 - 0.5).abs() < 0.3) & ((c[1] % 1 - 0.5).abs() < 0.3) & (c[0] < 25) & (c[0] > 3)
    sel = valid & inner
    assert sel.sum() > 1000
    assert ((out[0][sel] > 0.5).float() == want[sel]).float().mean() > 0.98


def test_splat_puts_ground_pixels_at_their_ground_point():
    """With the true ground depth, every splatted point lands in (or next to) the cell holding the ground point of
    its pixel ray: no range bias."""
    T = ground_cam2ego(1.4, 0.03)
    Tt = torch.tensor(T, dtype=torch.float32)[None]
    depth = _ground_depth(T, max_z=30)
    splat = DepthSplat(c_in=1, c=1)
    idx = splat.cell_index(depth, K, Tt, (H // 4, W // 4))[0]
    g = BEVGrid()
    ok = idx < g.hw[0] * g.hw[1]
    assert ok.sum() > 2000
    n = torch.nonzero(ok)[:, 0]
    pu, pv = (n % (W // 4)).float() * 4 + 1.5, (n // (W // 4)).float() * 4 + 1.5
    ray = torch.stack([(pu - 320) / 560, (pv - 192) / 560, torch.ones_like(pu)])
    re = Tt[0, :3, :3] @ ray
    lam = -Tt[0, 2, 3] / re[2]
    gx, gy = Tt[0, 0, 3] + lam * re[0], Tt[0, 1, 3] + lam * re[1]  # true ground point of the pixel ray
    cells = g.centers()[:, idx[ok]]
    err = torch.hypot(cells[0] - gx, cells[1] - gy)
    tol = 0.15 + 0.01 * gx  # half a cell diagonal + the convexity of 1/v inside a 4-row block
    assert (err < tol).float().mean() > 0.97, err.max()
    assert (cells[0] - gx).mean().abs() < 0.05  # no systematic range bias


def test_example_depth_follows_the_pose_and_ipm_survives_fp16_ranges():
    from sparc.nn.model import ground_depth

    T = ground_cam2ego(1.4, 0.03)
    d = ground_depth(K, torch.tensor(T, dtype=torch.float32)[None], (H, W))
    torch.testing.assert_close(d, _ground_depth(T), atol=1e-3, rtol=1e-4)
    # a level camera projects near cells to |u| ~ 1e5: the IPM grid must stay finite and masked
    T0 = torch.tensor(ground_cam2ego(1.4, 0.0), dtype=torch.float32)[None]
    ipm = IPMSampler(c_in=1, c=1)
    ipm.reduce = nn.Identity()
    out = ipm(torch.ones(1, 1, H // 8, W // 8), K, T0)
    assert torch.isfinite(out).all() and out[0, 1].sum() > 100
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out16 = ipm(torch.ones(1, 1, H // 8, W // 8), K, T0)
    assert torch.isfinite(out16).all()


def test_gwc_volume_peaks_at_the_true_disparity():
    torch.manual_seed(1)
    fl = torch.randn(1, 64, 8, 40)
    d0 = 5
    fr = torch.zeros_like(fl)
    fr[..., :-d0] = fl[..., d0:]  # right-eye feature at x - d0 equals left at x
    vol = gwc_volume(fl, fr, levels=12, groups=4).view(1, 4, 12, 8, 40)
    best = vol.mean(1)[..., 12:].argmax(1)  # columns with x >= levels
    assert (best == d0).all()


def test_convex_upsample_keeps_constants_and_stereo_shapes():
    x = torch.full((1, 1, 6, 10), 2.5)
    up = convex_upsample(x, torch.randn(1, 9 * 16, 6, 10), 4)
    assert up.shape == (1, 1, 24, 40) and torch.allclose(up, torch.full_like(up, 2.5))
    head = StereoHead(max_disp=32).eval()
    with torch.no_grad():
        disp = head(torch.randn(1, 3, 64, 96), torch.randn(1, 3, 64, 96))
    assert disp.shape == (1, 1, 64, 96) and (disp >= 0).all() and (disp <= 32).all()
    depth = StereoHead.to_depth(torch.tensor([[[[10.0, 0.1]]]]), K, torch.tensor([[0.1]]))
    assert depth[0, 0, 0].tolist() == pytest.approx([5.6, 0.0])


def test_sparc_deploy_contract_and_ego_position():
    net = SparcNet(SparcCfg(backbone="vit_t16_dinov3", imgsz=(128, 192))).eval()
    inp = example_inputs(net)
    dep = SparcDeploy(net)
    with torch.no_grad():
        outs = dict(zip(dep.output_names(), dep(*inp.values())))
    assert outs["agents"].shape == (1, 200, 18) and outs["da"].dtype == torch.int32
    assert outs["bev"].shape == (1, 7, 240, 160) and outs["lane_meta"].shape == (1, 24, 9)
    a = outs["agents"][0]
    assert (a[:, 2] > a[:, 0]).all() and (a[:, 4] >= 0).all() and (a[:, 4] <= 1).all()
    assert set(outs["lane_meta"][0, :, 1].tolist()) <= {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0}
    # x_ego / y_ego is the box centre (pixel-edge coordinates) back-projected at the predicted depth
    Kc, T = inp["K"][0], inp["cam2ego"][0]
    cx, cy, z = (a[0, 0] + a[0, 2]) / 2 - 0.5, (a[0, 1] + a[0, 3]) / 2 - 0.5, a[0, 6]
    p = torch.stack([(cx - Kc[0, 2]) / Kc[0, 0] * z, (cy - Kc[1, 2]) / Kc[1, 1] * z, z])
    e = T[:3, :3] @ p + T[:3, 3]
    assert a[0, 12].item() == pytest.approx(e[0].item(), rel=1e-4) and a[0, 13].item() == pytest.approx(e[1].item(), abs=1e-4)


def test_stage_cuts_and_input_names():
    net = SparcNet(SparcCfg(backbone="vit_t16_dinov3", imgsz=(128, 192), stereo=True)).eval()
    assert net.input_names("full") == ["image", "K", "cam2ego", "image_right", "baseline"]
    assert net.input_names("backbone") == ["image"]
    with torch.no_grad():
        enc = SparcDeploy(net, "encoder")(example_inputs(net, "encoder")["image"])
    assert [t.shape[1] for t in enc] == [64, 256, 256, 256]
    inp = example_inputs(net)
    with pytest.raises(ValueError, match="image_right"):
        net(**{k: v for k, v in inp.items() if k != "image_right"})
    # a monocular sample with a depth map (e.g. LiDAR) trains the BEV through the depth input instead
    mono = {k: v for k, v in inp.items() if k not in ("image_right", "baseline")}
    with torch.no_grad():
        out = net(**mono, depth=torch.zeros(1, 1, 128, 192))
    assert "bev" in out and "disparity" not in out
