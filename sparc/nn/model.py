"""SPARC network: backbone -> hybrid encoder -> typed-query decoder + dense head + (stereo) + BEV head.

``SparcNet.forward`` returns raw head outputs (training / evaluation in PyTorch). ``SparcDeploy`` wraps it with the
engine I/O contract (``docs/sparc/plan.md`` section 5) and post-processing that belongs in the graph.

Inputs (all float32): ``image`` (B, 3, H, W) RGB in [0, 1], letterboxed rectified left eye; ``K`` (B, 3, 3) at network
pixels; ``cam2ego`` (B, 4, 4); and either ``depth`` (B, 1, H, W) metres (0 = invalid; any depth source) or, with
``stereo=True``, ``image_right`` (B, 3, H, W) + ``baseline`` (B, 1) metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from sparc.data.schema import DA_CLASSES, LANE_ORDER_RANGE
from sparc.nn.backbone import build_backbone
from sparc.nn.bev import BEVGrid, BEVHead
from sparc.nn.common import Normalize, l2_normalize
from sparc.nn.decoder import DecoderCfg, TypedQueryDecoder
from sparc.nn.dense import DenseHead
from sparc.nn.encoder import HybridEncoder
from sparc.nn.stereo import StereoHead

STAGES = ("backbone", "encoder", "decoder", "full")


@dataclass
class SparcCfg:
    backbone: str = "vit_s16_dinov3"
    imgsz: tuple = (384, 640)
    pretrained: bool = False
    decoder: DecoderCfg = field(default_factory=DecoderCfg)
    enc_depth: int = 1
    enc_expansion: float = 0.5
    bev: bool = True
    bev_splat: bool = True
    bev_ipm: bool = True
    stereo: bool = False
    max_disp: int = 192
    lane_aux: bool = True  # training-only lane-marking mask head (not exported)


class SparcNet(nn.Module):
    def __init__(self, cfg: SparcCfg = SparcCfg()):
        super().__init__()
        self.cfg = cfg
        self.norm = Normalize()
        self.backbone = build_backbone(cfg.backbone, cfg.imgsz, cfg.pretrained)
        self.encoder = HybridEncoder(self.backbone.out_channels, cfg.decoder.d, cfg.enc_depth, cfg.enc_expansion)
        self.decoder = TypedQueryDecoder(cfg.decoder, levels=3)
        self.da_head = DenseHead(len(DA_CLASSES), cfg.decoder.d)
        self.lane_aux = DenseHead(2, cfg.decoder.d) if cfg.lane_aux else None
        self.stereo = StereoHead(cfg.max_disp) if cfg.stereo else None
        self.bev = BEVHead(cfg.decoder.d, splat=cfg.bev_splat, ipm=cfg.bev_ipm) if cfg.bev else None

    def forward(self, image, K=None, cam2ego=None, depth=None, image_right=None, baseline=None,
                until: str = "full") -> Dict[str, torch.Tensor]:
        if until not in STAGES:
            raise ValueError(f"until must be one of {STAGES}")
        x = self.norm(image)
        feats = self.backbone(x)
        if until == "backbone":
            return feats
        enc = self.encoder([feats["p3"], feats["p4"], feats["p5"]])
        if until == "encoder":
            return {"p2": feats["p2"], "e3": enc[0], "e4": enc[1], "e5": enc[2]}
        out = self.decoder(enc)
        if until == "decoder":
            return out
        out["da_logits"] = self.da_head(feats["p2"], enc[0])
        if self.lane_aux is not None and self.training:
            out["lane_mask_logits"] = self.lane_aux(feats["p2"], enc[0])
        if self.stereo is not None and image_right is not None:
            if baseline is None or K is None:
                raise ValueError("the stereo head needs baseline and K with image_right")
            out["disparity"] = self.stereo(x, self.norm(image_right))
            if depth is None:  # a given depth (training on a source with better depth) takes precedence for the BEV
                depth = StereoHead.to_depth(out["disparity"], K, baseline)
            out["depth"] = StereoHead.to_depth(out["disparity"], K, baseline)
        elif self.stereo is not None and depth is None and self.bev is not None and self.bev.splat is not None:
            raise ValueError("stereo=True needs image_right and baseline (or a depth map for monocular sources)")
        if self.bev is not None:
            if K is None or cam2ego is None:
                raise ValueError("the BEV head needs K and cam2ego")
            if depth is None:
                if self.bev.splat is not None:
                    raise ValueError("the depth splat needs depth (an input, or stereo=True)")
                depth = torch.zeros_like(image[:, :1])
            out["bev"] = self.bev(enc[0], depth, K, cam2ego)
        return out

    def input_names(self, until: str = "full"):
        names = ["image"]
        if until == "full":
            names += ["K", "cam2ego"] if (self.bev is not None or self.stereo is not None) else []
            if self.stereo is not None:
                names += ["image_right", "baseline"]
            elif self.bev is not None and self.bev.splat is not None:
                names += ["depth"]
        return names


ORDER_VALUES = [o for o in range(-LANE_ORDER_RANGE, LANE_ORDER_RANGE + 1) if o != 0]
AGENT_COLUMNS = ("x1", "y1", "x2", "y2", "score", "cls", "depth", "yaw_sin", "yaw_cos", "length", "width", "height",
                 "x_ego", "y_ego", "tl_id", "tl_p", "sign_id", "sign_p")


def _top1(logits: torch.Tensor):
    p = logits.softmax(-1)
    v, i = p.max(-1)
    return i.to(logits.dtype), v


class SparcDeploy(nn.Module):
    """Engine I/O contract around :class:`SparcNet` (positional inputs in ``input_names`` order)."""

    def __init__(self, net: SparcNet, until: str = "full"):
        super().__init__()
        self.net, self.until = net.eval(), until
        self.inputs = net.input_names(until)
        self.register_buffer("order_values", torch.tensor(ORDER_VALUES, dtype=torch.float32), persistent=False)

    def output_names(self):
        if self.until == "backbone":
            return ["p2", "p3", "p4", "p5"]
        if self.until == "encoder":
            return ["p2", "e3", "e4", "e5"]
        names = ["agents", "agent_emb", "lane_xs", "lane_vis", "lane_meta"]
        if self.until == "full":
            names.append("da")
            if self.net.stereo is not None:
                names.append("depth")
            if self.net.bev is not None:
                names.append("bev")
        return names

    def forward(self, *args):
        kw = dict(zip(self.inputs, args))
        out = self.net(until=self.until, **kw)
        if self.until in ("backbone", "encoder"):
            return tuple(out[k] for k in self.output_names())
        image = kw["image"]
        h, w = image.shape[-2:]
        scale = torch.tensor([w, h, w, h], dtype=image.dtype, device=image.device)
        cxcywh = out["boxes"] * scale
        xyxy = torch.cat([cxcywh[..., :2] - cxcywh[..., 2:] / 2, cxcywh[..., :2] + cxcywh[..., 2:] / 2], -1)
        score, cls = out["logits"].sigmoid().max(-1)
        b3 = out["box3d"]
        depth = b3[..., 0].exp()
        yaw = l2_normalize(b3[..., 1:3])
        lwh = b3[..., 3:6].exp()
        if "K" in kw:
            K, T = kw["K"], kw["cam2ego"]
            xc = (cxcywh[..., 0] - K[:, 0:1, 2]) / K[:, 0:1, 0] * depth
            yc = (cxcywh[..., 1] - K[:, 1:2, 2]) / K[:, 1:2, 1] * depth
            p = torch.stack([xc, yc, depth], 1)  # (B, 3, N)
            e = T[:, :3, :3] @ p + T[:, :3, 3:4]
            x_ego, y_ego = e[:, 0], e[:, 1]
        else:
            x_ego = y_ego = torch.zeros_like(depth)
        tl_id, tl_p = _top1(out["tl_logits"])
        sign_id, sign_p = _top1(out["sign_logits"])
        agents = torch.cat([xyxy, score[..., None], cls[..., None].to(image.dtype), depth[..., None], yaw, lwh,
                            x_ego[..., None], y_ego[..., None], tl_id[..., None], tl_p[..., None],
                            sign_id[..., None], sign_p[..., None]], -1)
        lane_xs = out["lane_x"] * w
        lane_vis = out["lane_vis_logits"].sigmoid()
        o_id, o_p = _top1(out["lane_order_logits"])
        o_val = torch.gather(self.order_values.expand(o_id.shape[0], -1), 1, o_id.long())
        metas = [out["lane_exist_logits"].sigmoid(), o_val, o_p]
        for k in ("lane_dir_logits", "lane_cont_logits", "lane_type_logits"):
            metas += list(_top1(out[k]))
        res = [agents, out["id_emb"], lane_xs, lane_vis, torch.stack(metas, -1)]
        if self.until == "full":
            res.append(out["da_logits"].argmax(1).to(torch.int32))
            if self.net.stereo is not None:
                res.append(out["depth"])
            if self.net.bev is not None:
                res.append(out["bev"])
        return tuple(res)


def example_inputs(net: SparcNet, until: str = "full", batch: int = 1, seed: int = 0,
                   K: Optional[torch.Tensor] = None, cam2ego: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Plausible inputs for export / parity checks: a smooth image, a ground-plane depth map, a nominal camera."""
    from sparc.data.schema import ground_cam2ego

    h, w = net.cfg.imgsz
    g = torch.Generator().manual_seed(seed)
    img = torch.nn.functional.interpolate(torch.rand(batch, 3, h // 16, w // 16, generator=g), size=(h, w),
                                          mode="bilinear", align_corners=False)
    if K is None:
        K = torch.tensor([[0.9 * w, 0, w / 2], [0, 0.9 * w, h / 2], [0, 0, 1]], dtype=torch.float32)
    K = K.expand(batch, 3, 3).contiguous()
    T = (cam2ego if cam2ego is not None else torch.tensor(ground_cam2ego(1.4, 0.03), dtype=torch.float32))
    T = T.expand(batch, 4, 4).contiguous()
    # depth of the ground plane below the horizon, 0 above it and beyond 40 m
    v = torch.arange(h, dtype=torch.float32).view(1, 1, h, 1).expand(batch, 1, h, w)
    fy, cy = K[:, 1, 1].view(-1, 1, 1, 1), K[:, 1, 2].view(-1, 1, 1, 1)
    dz = fy * 1.4 / (v - cy).clamp(min=1e-3)
    depth = torch.where((v > cy + 1) & (dz < 40), dz, torch.zeros_like(dz))
    out = {"image": img, "K": K, "cam2ego": T, "depth": depth,
           "image_right": torch.roll(img, shifts=-4, dims=-1), "baseline": torch.full((batch, 1), 0.1)}
    return {k: out[k] for k in net.input_names(until)}
