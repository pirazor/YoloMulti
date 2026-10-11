"""One deformable-attention decoder with typed queries: road users (agents), traffic elements and lanes.

The three query types share the decoder layers: one self-attention over all queries (with a learnt type
embedding, so lanes and agents can inform each other), one multi-scale deformable cross-attention over the
encoder maps (one value projection, three query sets), one FFN. They differ in reference geometry and heads:

* agent queries (``n_agents``): selected from encoder tokens (RT-DETR query selection over the agent classes),
  reference **box** refined per layer; heads: class (8), box, 3D state (log-depth, sin/cos yaw, log l/w/h), ID
  embedding;
* element queries (``n_elements``): selected over the element classes (light / sign), reference box; heads:
  class (2), attribute (13: colour or meaning), fine sign class (vocabulary size);
* lane queries (``n_lanes``): learnt, reference **point**; heads: lateral offset ``y`` and height ``z`` (metres,
  ego frame) at the forward anchors, per-anchor visibility, existence, order relative to ego, travel direction,
  continuity, type. The image-space lane is the projection of these points (``sparc.data.geometry``).

In training mode ``forward`` also returns every layer's outputs (``aux``) and the encoder's dense proposals
(``enc_logits`` over the 10 agent + element classes, ``enc_box``) for the auxiliary losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from sparc.data.schema import (AGENT_CLASSES, ELEMENT_ATTRS, ELEMENT_CLASSES, LANE_CONTINUITY, LANE_DIRECTIONS,
                               LANE_ORDER_RANGE, LANE_TYPES, LANE_X_ANCHORS)
from sparc.nn.common import MLP, bias_init_with_prob, inverse_sigmoid, l2_normalize
from sparc.nn.deform_attn import MSDeformAttn

N_ORDER = 2 * LANE_ORDER_RANGE  # -3..-1, 1..3
Y_RANGE = 16.0                  # lateral range of the lane head (m), +-16 m = the BEV width


@dataclass
class DecoderCfg:
    d: int = 256
    layers: int = 3
    heads: int = 8
    points: int = 4
    ffn: int = 1024
    n_agents: int = 200
    n_elements: int = 50
    n_lanes: int = 24
    n_anchors: int = len(LANE_X_ANCHORS)
    n_cls: int = len(AGENT_CLASSES)
    n_el_cls: int = len(ELEMENT_CLASSES)
    n_attr: int = len(ELEMENT_ATTRS)
    n_sign: int = 1              # fine sign vocabulary size (set from the data manifest)
    id_dim: int = 64
    attn_mode: str = "grid_sample"


class DecoderLayer(nn.Module):
    def __init__(self, c: DecoderCfg, levels: int):
        super().__init__()
        self.sa = nn.MultiheadAttention(c.d, c.heads, batch_first=True)
        self.ca = MSDeformAttn(c.d, c.heads, levels, c.points, c.attn_mode)
        self.n1, self.n2, self.n3 = nn.LayerNorm(c.d), nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.ffn = nn.Sequential(nn.Linear(c.d, c.ffn), nn.ReLU(inplace=True), nn.Linear(c.ffn, c.d))

    def forward(self, x, pos, refs: Sequence[torch.Tensor], value, shapes):
        """``refs``: one reference tensor per query group, in the order the groups are concatenated in ``x``."""
        q = x + pos
        x = self.n1(x + self.sa(q, q, x, need_weights=False)[0])
        q = x + pos
        parts, start = [], 0
        for r in refs:
            n = r.shape[1]
            parts.append(self.ca.attend(q[:, start:start + n], r, value, shapes))
            start += n
        x = self.n2(x + torch.cat(parts, 1))
        return self.n3(x + self.ffn(x))


class TypedQueryDecoder(nn.Module):
    def __init__(self, cfg: DecoderCfg = DecoderCfg(), levels: int = 3):
        super().__init__()
        c = self.cfg = cfg
        n_sel = c.n_cls + c.n_el_cls
        # query selection (RT-DETR): class logits + box proposal per encoder token
        self.enc_out = nn.Sequential(nn.Linear(c.d, c.d), nn.LayerNorm(c.d))
        self.enc_cls = nn.Linear(c.d, n_sel)
        self.enc_box = MLP(c.d, c.d, 4, 3)
        # lane queries
        self.lane_content = nn.Embedding(c.n_lanes, c.d)
        self.lane_ref = nn.Embedding(c.n_lanes, 2)
        self.type_emb = nn.Embedding(3, c.d)
        self.pos_box = MLP(4, 2 * c.d, c.d, 2)
        self.pos_pt = MLP(2, 2 * c.d, c.d, 2)
        self.layers = nn.ModuleList(DecoderLayer(c, levels) for _ in range(c.layers))
        self.box_delta = nn.ModuleList(MLP(c.d, c.d, 4, 3) for _ in range(c.layers))
        self.pt_delta = nn.ModuleList(MLP(c.d, c.d, 2, 2) for _ in range(c.layers))
        # agent heads
        self.cls = nn.Linear(c.d, c.n_cls)
        self.box3d = MLP(c.d, c.d, 6, 2)
        self.id_emb = MLP(c.d, c.d, c.id_dim, 2)
        # element heads
        self.el_cls = nn.Linear(c.d, c.n_el_cls)
        self.el_attr = nn.Linear(c.d, c.n_attr)
        self.el_sign = nn.Linear(c.d, max(1, c.n_sign))
        # lane heads
        self.lane_y = MLP(c.d, c.d, c.n_anchors, 3)
        self.lane_z = MLP(c.d, c.d, c.n_anchors, 2)
        self.lane_vis = nn.Linear(c.d, c.n_anchors)
        self.lane_exist = nn.Linear(c.d, 1)
        self.lane_order = nn.Linear(c.d, N_ORDER)
        self.lane_dir = nn.Linear(c.d, len(LANE_DIRECTIONS))
        self.lane_cont = nn.Linear(c.d, len(LANE_CONTINUITY))
        self.lane_type = nn.Linear(c.d, len(LANE_TYPES))
        self._reset()

    def _reset(self):
        b = bias_init_with_prob(0.01)
        for l in (self.enc_cls, self.cls, self.el_cls, self.lane_exist):
            nn.init.constant_(l.bias, b)
        for m in list(self.box_delta) + list(self.pt_delta) + [self.enc_box, self.lane_z]:
            nn.init.zeros_(m.layers[-1].weight)
            nn.init.zeros_(m.layers[-1].bias)
        with torch.no_grad():  # lane references spread across the lower image; lanes start 1.75 m apart
            n = self.cfg.n_lanes
            self.lane_ref.weight[:, 0] = inverse_sigmoid(torch.linspace(0.05, 0.95, n))
            self.lane_ref.weight[:, 1] = inverse_sigmoid(torch.full((n,), 0.75))
            nn.init.zeros_(self.lane_y.layers[-1].weight)
            self.lane_y.layers[-1].bias.fill_(0.0)

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

    def _select(self, enc, enc_logits, anchors, scores, k):
        """Top-k tokens by ``scores`` (B, S) -> (content (B, k, d), reference boxes (B, k, 4) sigmoid, indices)."""
        topk = scores.topk(k, dim=1).indices
        sel = torch.gather(enc, 1, topk[..., None].expand(-1, -1, enc.shape[-1]))
        box = self.enc_box(sel) + torch.gather(anchors, 1, topk[..., None].expand(-1, -1, 4))
        return sel.detach(), box.sigmoid().detach(), box, topk

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = self.cfg
        shapes = [tuple(f.shape[-2:]) for f in feats]
        memory = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], 1)  # (B, S, d)
        b = memory.shape[0]
        enc = self.enc_out(memory)
        enc_logits = self.enc_cls(enc)
        anchors = self.anchors(shapes, memory.device, memory.dtype).expand(b, -1, -1)
        x_agent, ref_a, box_a, topk_a = self._select(enc, enc_logits, anchors, enc_logits[..., :c.n_cls].max(-1).values, c.n_agents)
        x_el, ref_e, box_e, topk_e = self._select(enc, enc_logits, anchors, enc_logits[..., c.n_cls:].max(-1).values, c.n_elements)
        x_lane = self.lane_content.weight[None].expand(b, -1, -1)
        ref_l = self.lane_ref.weight[None].expand(b, -1, -1).sigmoid()
        x = torch.cat([x_agent, x_el, x_lane], 1)
        type_pos = torch.cat([self.type_emb.weight[0:1].expand(c.n_agents, -1), self.type_emb.weight[1:2].expand(c.n_elements, -1),
                              self.type_emb.weight[2:3].expand(c.n_lanes, -1)], 0)[None]
        na, ne = c.n_agents, c.n_elements
        outs = []
        for i, layer in enumerate(self.layers):
            pos = torch.cat([self.pos_box(ref_a), self.pos_box(ref_e), self.pos_pt(ref_l)], 1) + type_pos
            value = layer.ca.project_value(memory)
            x = layer(x, pos, (ref_a, ref_e, ref_l), value, shapes)
            xa, xe, xl = x[:, :na], x[:, na:na + ne], x[:, na + ne:]
            new_a = (self.box_delta[i](xa) + inverse_sigmoid(ref_a)).sigmoid()
            new_e = (self.box_delta[i](xe) + inverse_sigmoid(ref_e)).sigmoid()
            new_l = (self.pt_delta[i](xl) + inverse_sigmoid(ref_l)).sigmoid()
            if self.training or i == len(self.layers) - 1:
                outs.append(self._heads(xa, xe, xl, new_a, new_e, new_l))
            if self.training:
                ref_a, ref_e, ref_l = new_a.detach(), new_e.detach(), new_l.detach()
            else:
                ref_a, ref_e, ref_l = new_a, new_e, new_l
        out = outs[-1]
        if self.training:
            out["aux"] = outs[:-1]
            out["enc_logits"] = enc_logits
            out["enc_box"] = (self.enc_box(enc) + anchors).sigmoid()  # dense proposals for the selection loss
            out["topk"] = (topk_a, topk_e)
        return out

    def _heads(self, xa, xe, xl, box_a, box_e, pt_l) -> Dict[str, torch.Tensor]:
        return {
            "logits": self.cls(xa), "boxes": box_a, "box3d": self.box3d(xa), "id_emb": l2_normalize(self.id_emb(xa)),
            "el_logits": self.el_cls(xe), "el_boxes": box_e, "el_attr_logits": self.el_attr(xe),
            "el_sign_logits": self.el_sign(xe),
            "lane_y": Y_RANGE * torch.tanh(self.lane_y(xl) / Y_RANGE), "lane_z": self.lane_z(xl),
            "lane_vis_logits": self.lane_vis(xl), "lane_exist_logits": self.lane_exist(xl).squeeze(-1), "lane_ref": pt_l,
            "lane_order_logits": self.lane_order(xl), "lane_dir_logits": self.lane_dir(xl),
            "lane_cont_logits": self.lane_cont(xl), "lane_type_logits": self.lane_type(xl),
        }
