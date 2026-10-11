"""One deformable-attention decoder with typed queries: agents (road users, lights, signs) and lanes.

Both query types share the decoder layers: one self-attention over all queries (with a learnt type embedding,
so agents can attend to lanes and the other way round), one multi-scale deformable cross-attention over the
encoder maps (one value projection, two query sets), one FFN. They differ in their reference geometry and heads:

* agent queries (``n_agents``, default 200) are selected from the encoder tokens (RT-DETR query selection) and carry
  a reference **box** refined layer by layer; heads: class, box, 3D state (log-depth, sin/cos yaw, log l/w/h),
  traffic-light colour, fine sign class, and an L2-normalised ID embedding for tracking;
* lane queries (``n_lanes``, default 24) are learnt and carry a reference **point** (the lane's centroid);
  heads: x per row anchor (``rows`` anchors from 0.35 H to H), per-row visibility, existence, order relative to
  ego, travel direction, continuity and type.

Training-only parts (denoising queries, matching, auxiliary losses) come in Phase 2; ``forward`` already returns
every layer's outputs in training mode for the auxiliary losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from sparc.data.schema import DET_CLASSES, LANE_CONTINUITY, LANE_DIRECTIONS, LANE_ORDER_RANGE, LANE_TYPES, TL_COLORS
from sparc.nn.common import MLP, bias_init_with_prob, inverse_sigmoid, l2_normalize
from sparc.nn.deform_attn import MSDeformAttn


@dataclass
class DecoderCfg:
    d: int = 256
    layers: int = 3
    heads: int = 8
    points: int = 4
    ffn: int = 1024
    n_agents: int = 200
    n_lanes: int = 24
    rows: int = 72
    row_top: float = 0.35  # first row anchor, as a fraction of the image height
    n_cls: int = len(DET_CLASSES)
    n_sign: int = 401      # MTSD fine classes + "other"
    id_dim: int = 64
    attn_mode: str = "grid_sample"


N_ORDER = 2 * LANE_ORDER_RANGE  # -3..-1, 1..3


class DecoderLayer(nn.Module):
    def __init__(self, c: DecoderCfg, levels: int):
        super().__init__()
        self.sa = nn.MultiheadAttention(c.d, c.heads, batch_first=True)
        self.ca = MSDeformAttn(c.d, c.heads, levels, c.points, c.attn_mode)
        self.n1, self.n2, self.n3 = nn.LayerNorm(c.d), nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.ffn = nn.Sequential(nn.Linear(c.d, c.ffn), nn.ReLU(inplace=True), nn.Linear(c.ffn, c.d))

    def forward(self, x, pos, ref_box, ref_pt, value, shapes, n_agents):
        q = x + pos
        x = self.n1(x + self.sa(q, q, x, need_weights=False)[0])
        q = x + pos
        ca = torch.cat([self.ca.attend(q[:, :n_agents], ref_box, value, shapes),
                        self.ca.attend(q[:, n_agents:], ref_pt, value, shapes)], 1)
        x = self.n2(x + ca)
        return self.n3(x + self.ffn(x))


class TypedQueryDecoder(nn.Module):
    def __init__(self, cfg: DecoderCfg = DecoderCfg(), levels: int = 3):
        super().__init__()
        c = self.cfg = cfg
        # query selection
        self.enc_out = nn.Sequential(nn.Linear(c.d, c.d), nn.LayerNorm(c.d))
        self.enc_cls = nn.Linear(c.d, c.n_cls)
        self.enc_box = MLP(c.d, c.d, 4, 3)
        # lane queries
        self.lane_content = nn.Embedding(c.n_lanes, c.d)
        self.lane_ref = nn.Embedding(c.n_lanes, 2)
        self.type_emb = nn.Embedding(2, c.d)
        self.pos_box = MLP(4, 2 * c.d, c.d, 2)
        self.pos_pt = MLP(2, 2 * c.d, c.d, 2)
        self.layers = nn.ModuleList(DecoderLayer(c, levels) for _ in range(c.layers))
        self.box_delta = nn.ModuleList(MLP(c.d, c.d, 4, 3) for _ in range(c.layers))
        self.pt_delta = nn.ModuleList(MLP(c.d, c.d, 2, 2) for _ in range(c.layers))
        # agent heads
        self.cls = nn.Linear(c.d, c.n_cls)
        self.box3d = MLP(c.d, c.d, 6, 2)
        self.tl = nn.Linear(c.d, len(TL_COLORS))
        self.sign = nn.Linear(c.d, c.n_sign)
        self.id_emb = MLP(c.d, c.d, c.id_dim, 2)
        # lane heads
        self.lane_x = MLP(c.d, c.d, c.rows, 3)
        self.lane_vis = nn.Linear(c.d, c.rows)
        self.lane_exist = nn.Linear(c.d, 1)
        self.lane_order = nn.Linear(c.d, N_ORDER)
        self.lane_dir = nn.Linear(c.d, len(LANE_DIRECTIONS))
        self.lane_cont = nn.Linear(c.d, len(LANE_CONTINUITY))
        self.lane_type = nn.Linear(c.d, len(LANE_TYPES))
        self._reset()

    def _reset(self):
        b = bias_init_with_prob(0.01)
        for l in (self.enc_cls, self.cls, self.lane_exist):
            nn.init.constant_(l.bias, b)
        for m in list(self.box_delta) + list(self.pt_delta) + [self.enc_box]:
            nn.init.zeros_(m.layers[-1].weight)
            nn.init.zeros_(m.layers[-1].bias)
        with torch.no_grad():  # lane references spread across the lower image
            n = self.cfg.n_lanes
            self.lane_ref.weight[:, 0] = inverse_sigmoid(torch.linspace(0.05, 0.95, n))
            self.lane_ref.weight[:, 1] = inverse_sigmoid(torch.full((n,), 0.75))

    @staticmethod
    def anchors(shapes: Sequence[Tuple[int, int]], device, dtype) -> torch.Tensor:
        """(1, S, 4) anchor boxes in logit space, one per encoder token (RT-DETR)."""
        out = []
        for lvl, (h, w) in enumerate(shapes):
            gy, gx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                    torch.arange(w, device=device, dtype=dtype), indexing="ij")
            xy = torch.stack([(gx + 0.5) / w, (gy + 0.5) / h], -1).flatten(0, 1)
            wh = torch.full_like(xy, 0.05 * 2.0 ** lvl)
            out.append(torch.cat([xy, wh], -1))
        return inverse_sigmoid(torch.cat(out, 0))[None]

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = self.cfg
        shapes = [tuple(f.shape[-2:]) for f in feats]
        memory = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], 1)  # (B, S, d)
        b = memory.shape[0]
        # agent query selection
        enc = self.enc_out(memory)
        enc_logits = self.enc_cls(enc)
        topk = enc_logits.max(-1).values.topk(c.n_agents, dim=1).indices  # (B, N)
        sel = torch.gather(enc, 1, topk[..., None].expand(-1, -1, c.d))
        anchors = self.anchors(shapes, memory.device, memory.dtype).expand(b, -1, -1)
        enc_box = self.enc_box(sel) + torch.gather(anchors, 1, topk[..., None].expand(-1, -1, 4))  # box MLP on top-k only
        ref_box = enc_box.sigmoid().detach()
        x_agent = sel.detach()
        x_lane = self.lane_content.weight[None].expand(b, -1, -1)
        ref_pt = self.lane_ref.weight[None].expand(b, -1, -1).sigmoid()
        x = torch.cat([x_agent, x_lane], 1)
        type_pos = torch.cat([self.type_emb.weight[:1].expand(c.n_agents, -1),
                              self.type_emb.weight[1:].expand(c.n_lanes, -1)], 0)[None]
        outs = []
        for i, layer in enumerate(self.layers):
            pos = torch.cat([self.pos_box(ref_box), self.pos_pt(ref_pt)], 1) + type_pos
            value = layer.ca.project_value(memory)
            x = layer(x, pos, ref_box, ref_pt, value, shapes, c.n_agents)
            xa, xl = x[:, : c.n_agents], x[:, c.n_agents:]
            new_box = (self.box_delta[i](xa) + inverse_sigmoid(ref_box)).sigmoid()
            new_pt = (self.pt_delta[i](xl) + inverse_sigmoid(ref_pt)).sigmoid()
            if self.training or i == len(self.layers) - 1:
                outs.append(self._heads(xa, xl, new_box, new_pt))
            ref_box, ref_pt = (new_box.detach(), new_pt.detach()) if self.training else (new_box, new_pt)
        out = outs[-1]
        if self.training:
            out["aux"] = outs[:-1]
            out["enc_logits"], out["enc_box"], out["topk"] = enc_logits, enc_box.sigmoid(), topk  # enc_box: (B, N, 4) of the top-k
        return out

    def _heads(self, xa, xl, box, pt) -> Dict[str, torch.Tensor]:
        return {
            "logits": self.cls(xa), "boxes": box,  # cxcywh normalised
            "box3d": self.box3d(xa), "tl_logits": self.tl(xa), "sign_logits": self.sign(xa),
            "id_emb": l2_normalize(self.id_emb(xa)),
            "lane_x": self.lane_x(xl).sigmoid(), "lane_vis_logits": self.lane_vis(xl),
            "lane_exist_logits": self.lane_exist(xl).squeeze(-1), "lane_ref": pt,
            "lane_order_logits": self.lane_order(xl), "lane_dir_logits": self.lane_dir(xl),
            "lane_cont_logits": self.lane_cont(xl), "lane_type_logits": self.lane_type(xl),
        }
