"""Matching and losses for the typed-query decoder, gated by each sample's label mask.

* Road users and traffic elements: Hungarian matching (class cost + L1 + GIoU), DEIM's matchability-aware
  classification loss (MAL, the IoU of the matched box as the target quality), L1 + GIoU on the boxes, 3D state and
  element attribute / fine sign losses on the matched queries. Classification negatives are restricted to the
  classes the source labels exhaustively (``cls_vocab`` / ``el_vocab``: Null-BCE); the fine sign loss is restricted
  to the vocabulary of the target's own source (names share a prefix), so a sign of another dataset is never a
  negative.
* Lanes: Hungarian matching in the ego frame (L1 on the lateral offset at the ground-truth-visible anchors +
  existence), then L1 on ``y`` and ``z`` (``z`` down-weighted when the sample's camera is nominal, since its lift
  assumes a flat road), BCE on per-anchor visibility and on existence, CE on the attributes (down-weighted when
  the attribute is a rule-derived weak label).
* Dense: masked CE + Dice on the drivable area, masked CE + focal Tversky on the lane-marking mask (both ignore 255,
  which is what an unlabelled task unpacks to).
* Auxiliary: the same query losses on every decoder layer and on the encoder's dense proposals (query selection).

Every term is averaged over the positives of the samples that carry the label, so a rare label is not diluted by
samples that do not have it. The gains are fixed (``LossGains``); ``docs/training.md`` describes the calibration
procedure that sets them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from adas_mt.data.masks import unpack_masks
from adas_mt.nn.loss import masked_ce, masked_dice, masked_focal_tversky
from sparc.data.schema import AGENT_CLASSES, DA_CLASSES, LANE_MASK_CLASSES
from sparc.nn.boxes import cxcywh_to_xyxy, elementwise_giou, elementwise_iou, pairwise_giou


@dataclass
class LossGains:
    cls: float = 1.0
    box_l1: float = 5.0
    giou: float = 2.0
    box3d: float = 1.0
    el_cls: float = 1.0
    el_attr: float = 1.0
    el_sign: float = 1.0
    lane_y: float = 2.0
    lane_z: float = 0.5
    lane_vis: float = 1.0
    lane_exist: float = 1.0
    lane_attr: float = 0.5
    weak: float = 0.3          # multiplier for rule-derived lane attributes
    nominal_z: float = 0.2     # multiplier for the lane height loss of nominal-camera samples
    da: float = 1.0
    lane_mask: float = 1.0
    enc: float = 1.0
    aux: float = 1.0
    match_cls: float = 2.0     # matching costs
    match_l1: float = 5.0
    match_giou: float = 2.0
    match_lane_y: float = 1.0
    match_lane_exist: float = 1.0
    mal_gamma: float = 1.5


def mal_loss(logits: torch.Tensor, quality: torch.Tensor, valid: torch.Tensor, gamma: float) -> torch.Tensor:
    """Matchability-aware loss (DEIM) summed over valid entries.

    ``quality`` (N, C): 0 for negatives, the matched IoU (in (0, 1]) for the positive class of a matched query;
    ``valid`` (N, C) masks out classes the sample does not label."""
    p = logits.sigmoid()
    pos = quality > 0
    q = quality.clamp(min=0) ** gamma
    loss_pos = -(q * F.logsigmoid(logits) + (1 - q) * F.logsigmoid(-logits))
    loss_neg = -(p ** gamma) * F.logsigmoid(-logits)
    loss = torch.where(pos, loss_pos, loss_neg)
    return (loss * valid).sum()


def _hungarian(cost: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    c = cost.detach().float().cpu().numpy()
    c = np.nan_to_num(c, nan=1e6, posinf=1e6, neginf=-1e6)
    return linear_sum_assignment(c)


class Criterion(nn.Module):
    def __init__(self, gains: LossGains = LossGains(), sign_prefix: Optional[Sequence[int]] = None):
        """``sign_prefix``: per fine-sign class, an integer id of its source prefix (names ``<src>:<name>``)."""
        super().__init__()
        self.g = gains
        self.register_buffer("sign_prefix", torch.tensor(list(sign_prefix) if sign_prefix else [0], dtype=torch.long),
                             persistent=False)

    # ------------------------------------------------------------------ matching
    @torch.no_grad()
    def match_boxes(self, logits: torch.Tensor, boxes: torch.Tensor, gt_cls: torch.Tensor, gt_boxes: torch.Tensor):
        """One sample: logits (N, C), boxes (N, 4) cxcywh, gt (M,) / (M, 4) -> (pred idx, gt idx) arrays."""
        if len(gt_cls) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        g = self.g
        p = logits.sigmoid()[:, gt_cls]  # (N, M)
        alpha, gamma = 0.25, 2.0
        cost_cls = alpha * ((1 - p) ** gamma) * (-(p + 1e-8).log()) - (1 - alpha) * (p ** gamma) * (-(1 - p + 1e-8).log())
        cost_l1 = torch.cdist(boxes, gt_boxes, p=1)
        cost_giou = -pairwise_giou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(gt_boxes))
        cost = g.match_cls * cost_cls + g.match_l1 * cost_l1 + g.match_giou * cost_giou
        return _hungarian(cost)

    @torch.no_grad()
    def match_lanes(self, lane_y: torch.Tensor, exist_logits: torch.Tensor, gt_y: torch.Tensor, gt_vis: torch.Tensor):
        """lane_y (L, A), exist (L,), gt_y (M, A) with NaN outside gt_vis (M, A) -> (pred idx, gt idx)."""
        if len(gt_y) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        d = (lane_y[:, None, :] - torch.nan_to_num(gt_y)[None]).abs() * gt_vis[None].float()  # (L, M, A)
        cost_y = d.sum(-1) / gt_vis.sum(-1).clamp(min=1)[None]
        cost_exist = 1 - exist_logits.sigmoid()[:, None]
        return _hungarian(self.g.match_lane_y * cost_y + self.g.match_lane_exist * cost_exist)

    # ------------------------------------------------------------------ query losses (one decoder output)
    def _box_losses(self, out: Dict, batch: Dict, prefix: str, key_logits: str, key_boxes: str, gt_boxes_key: str,
                    gt_cls_key: str, vocab_key: str, mask_key: str, items: Dict, extra=None) -> Dict[str, torch.Tensor]:
        """Classification (MAL) + L1 + GIoU for one query group, with per-sample label gating."""
        logits, boxes = out[key_logits], out[key_boxes]
        B, N, C = logits.shape
        g = self.g
        use = batch["label_mask"][mask_key]
        quality = torch.zeros_like(logits)
        valid = batch[vocab_key].to(logits.device, logits.dtype)[:, None, :].expand(B, N, C).clone()
        valid[~use.to(logits.device)] = 0.0
        matched_p, matched_g, l1, giou = [], [], [], []
        n_pos = 0
        for b in range(B):
            if not use[b]:
                continue
            gt_boxes, gt_cls = batch[gt_boxes_key][b].to(boxes.device), batch[gt_cls_key][b].to(boxes.device)
            pi, gi = self.match_boxes(logits[b], boxes[b], gt_cls, gt_boxes)
            if len(pi) == 0:
                continue
            pi_t, gi_t = torch.as_tensor(pi, device=boxes.device), torch.as_tensor(gi, device=boxes.device)
            pb, gb = boxes[b, pi_t], gt_boxes[gi_t]
            iou = elementwise_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb)).detach().clamp(min=1e-3)
            quality[b, pi_t, gt_cls[gi_t]] = iou.to(quality.dtype)
            l1.append((pb - gb).abs().sum(-1))
            giou.append(1 - elementwise_giou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb)))
            matched_p.append((b, pi_t))
            matched_g.append(gi_t)
            n_pos += len(pi)
        n_pos_t = max(n_pos, 1)
        cls_loss = mal_loss(logits, quality, valid, g.mal_gamma) / n_pos_t
        l1_loss = torch.cat(l1).sum() / n_pos_t if l1 else logits.sum() * 0.0
        giou_loss = torch.cat(giou).sum() / n_pos_t if giou else logits.sum() * 0.0
        items[f"{prefix}cls"] = cls_loss
        items[f"{prefix}l1"] = l1_loss
        items[f"{prefix}giou"] = giou_loss
        if extra is not None:
            extra(matched_p, matched_g, n_pos_t)
        return items

    def _agent_extras(self, out, batch, items, prefix):
        def fn(matched_p, matched_g, n_pos):
            g = self.g
            loss3d, n3d = [], 0
            for (b, pi), gi in zip(matched_p, matched_g):
                if not batch["label_mask"]["box3d"][b]:
                    continue
                gt = batch["box3d"][b].to(out["box3d"].device)[gi]
                ok = torch.isfinite(gt).all(-1)
                if not ok.any():
                    continue
                pred = out["box3d"][b, pi][ok]
                gt = gt[ok]
                depth = (pred[:, 0] - gt[:, 0].clamp(min=0.1).log()).abs()
                yaw = (pred[:, 1:3] - torch.stack([gt[:, 1].sin(), gt[:, 1].cos()], -1)).abs().sum(-1)
                size = (pred[:, 3:6] - gt[:, 2:5].clamp(min=0.05).log()).abs().sum(-1)
                loss3d.append(depth + yaw + size)
                n3d += int(ok.sum())
            items[f"{prefix}box3d"] = torch.cat(loss3d).sum() / max(n3d, 1) if loss3d else out["box3d"].sum() * 0.0
        return fn

    def _element_extras(self, out, batch, items, prefix):
        def fn(matched_p, matched_g, n_pos):
            attr_l, sign_l, n_attr, n_sign = [], [], 0, 0
            for (b, pi), gi in zip(matched_p, matched_g):
                lm = batch["label_mask"]
                if lm["el_attr"][b]:
                    t = batch["el_attr"][b].to(out["el_attr_logits"].device)[gi]
                    ok = t >= 0
                    if ok.any():
                        attr_l.append(F.cross_entropy(out["el_attr_logits"][b, pi][ok], t[ok], reduction="sum"))
                        n_attr += int(ok.sum())
                if lm["el_sign"][b]:
                    t = batch["el_sign"][b].to(out["el_sign_logits"].device)[gi]
                    ok = t >= 0
                    if ok.any():
                        logits = out["el_sign_logits"][b, pi][ok]
                        same_src = self.sign_prefix[None, :] == self.sign_prefix[t[ok]][:, None]  # (n, V)
                        logits = logits.masked_fill(~same_src, float("-inf"))
                        sign_l.append(F.cross_entropy(logits, t[ok], reduction="sum"))
                        n_sign += int(ok.sum())
            items[f"{prefix}el_attr"] = torch.stack(attr_l).sum() / max(n_attr, 1) if attr_l else out["el_attr_logits"].sum() * 0.0
            items[f"{prefix}el_sign"] = torch.stack(sign_l).sum() / max(n_sign, 1) if sign_l else out["el_sign_logits"].sum() * 0.0
        return fn

    def _lane_losses(self, out: Dict, batch: Dict, prefix: str, items: Dict) -> None:
        g = self.g
        lane_y, lane_z = out["lane_y"], out["lane_z"]
        vis_logits, exist_logits = out["lane_vis_logits"], out["lane_exist_logits"]
        B, L, A = lane_y.shape
        use = batch["label_mask"]["lane_geom"]
        ly, lz, lvis, lexist, lattr = [], [], [], [], []
        n_pts, n_lanes = 0, 0
        exist_target = torch.zeros_like(exist_logits)
        for b in range(B):
            if not use[b]:
                continue
            gt_y, gt_vis = batch["lane_y"][b].to(lane_y.device), batch["lane_vis"][b].to(lane_y.device)
            pi, gi = self.match_lanes(lane_y[b], exist_logits[b], gt_y, gt_vis)
            n_lanes += L
            if len(pi) == 0:
                continue
            pi_t, gi_t = torch.as_tensor(pi, device=lane_y.device), torch.as_tensor(gi, device=lane_y.device)
            exist_target[b, pi_t] = 1.0
            v = gt_vis[gi_t].float()
            gy, gz = torch.nan_to_num(gt_y[gi_t]), torch.nan_to_num(batch["lane_z"][b].to(lane_y.device)[gi_t])
            ly.append(((lane_y[b, pi_t] - gy).abs() * v).sum())
            zw = g.nominal_z if bool(batch["nominal"][b]) else 1.0
            lz.append(((lane_z[b, pi_t] - gz).abs() * v).sum() * zw)
            lvis.append(F.binary_cross_entropy_with_logits(vis_logits[b, pi_t], v, reduction="sum"))
            n_pts += int(v.sum())
            # attributes: order / dir / cont / type (CE, -1 ignored), weak lanes down-weighted
            attr = batch["lane_attr"][b].to(lane_y.device)[gi_t]
            weak = batch["lane_weak"][b].to(lane_y.device)[gi_t]
            wts = torch.where(weak, torch.full_like(weak, g.weak, dtype=lane_y.dtype), torch.ones_like(weak, dtype=lane_y.dtype))
            for k, (key, flag) in enumerate((("lane_order_logits", "lane_order"), ("lane_dir_logits", "lane_dir"),
                                             ("lane_cont_logits", "lane_cont"), ("lane_type_logits", "lane_type"))):
                if not batch["label_mask"][flag][b]:
                    continue
                t = attr[:, k]
                ok = t >= 0
                if ok.any():
                    ce = F.cross_entropy(out[key][b, pi_t][ok], t[ok], reduction="none")
                    lattr.append((ce * wts[ok]).sum() / max(int(ok.sum()), 1))
        zero = lane_y.sum() * 0.0
        items[f"{prefix}lane_y"] = torch.stack(ly).sum() / max(n_pts, 1) if ly else zero
        items[f"{prefix}lane_z"] = torch.stack(lz).sum() / max(n_pts, 1) if lz else zero
        items[f"{prefix}lane_vis"] = torch.stack(lvis).sum() / max(n_pts, 1) if lvis else zero
        if n_lanes:
            m = use.to(exist_logits.device)
            items[f"{prefix}lane_exist"] = F.binary_cross_entropy_with_logits(exist_logits[m], exist_target[m], reduction="sum") / n_lanes
        else:
            items[f"{prefix}lane_exist"] = zero
        items[f"{prefix}lane_attr"] = torch.stack(lattr).mean() if lattr else zero

    def query_losses(self, out: Dict, batch: Dict, prefix: str = "") -> Dict[str, torch.Tensor]:
        items: Dict[str, torch.Tensor] = {}
        self._box_losses(out, batch, prefix, "logits", "boxes", "boxes", "cls", "cls_vocab", "agents", items,
                         extra=self._agent_extras(out, batch, items, prefix))
        self._box_losses(out, batch, prefix + "el_", "el_logits", "el_boxes", "el_boxes", "el_cls", "el_vocab",
                         "elements", items, extra=self._element_extras(out, batch, items, prefix))
        self._lane_losses(out, batch, prefix, items)
        return items

    def enc_losses(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Query-selection loss on the encoder's dense proposals: agents and elements as one 10-class problem."""
        n_cls = len(AGENT_CLASSES)
        B = out["enc_logits"].shape[0]
        joint = {"logits": out["enc_logits"], "boxes": out["enc_box"]}
        gt_boxes, gt_cls, vocab, use = [], [], [], []
        for b in range(B):
            lm = batch["label_mask"]
            bx, cl = [], []
            if lm["agents"][b]:
                bx.append(batch["boxes"][b])
                cl.append(batch["cls"][b])
            if lm["elements"][b]:
                bx.append(batch["el_boxes"][b])
                cl.append(batch["el_cls"][b] + n_cls)
            gt_boxes.append(torch.cat(bx) if bx else torch.zeros((0, 4)))
            gt_cls.append(torch.cat(cl) if cl else torch.zeros((0,), dtype=torch.long))
            vocab.append(torch.cat([batch["cls_vocab"][b] & lm["agents"][b], batch["el_vocab"][b] & lm["elements"][b]]))
            use.append(bool(lm["agents"][b] or lm["elements"][b]))
        jb = {"label_mask": {"joint": torch.tensor(use)}, "jboxes": gt_boxes, "jcls": gt_cls, "jvocab": torch.stack(vocab)}
        items: Dict[str, torch.Tensor] = {}
        self._box_losses(joint, jb, "enc_", "logits", "boxes", "jboxes", "jcls", "jvocab", "joint", items)
        return items

    def dense_losses(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        items = {}
        da_t, ll_t = unpack_masks(batch["mask"].to(out["da_logits"].device), len(DA_CLASSES), len(LANE_MASK_CLASSES))
        items["da_ce"] = masked_ce(out["da_logits"], da_t)
        items["da_dice"] = masked_dice(out["da_logits"], da_t)
        if "lane_mask_logits" in out:
            items["ll_ce"] = masked_ce(out["lane_mask_logits"], ll_t)
            items["ll_tversky"] = masked_focal_tversky(out["lane_mask_logits"], ll_t)
        return items

    # ------------------------------------------------------------------ total
    def forward(self, out: Dict, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        g = self.g
        items = self.query_losses(out, batch)
        total = self._weighted(items, "")
        for i, aux in enumerate(out.get("aux", [])):
            aux_items = self.query_losses(aux, batch, prefix=f"aux{i}_")
            total = total + g.aux * self._weighted(aux_items, f"aux{i}_")
            items.update(aux_items)
        if "enc_logits" in out:
            enc = self.enc_losses(out, batch)
            total = total + g.enc * (g.cls * enc["enc_cls"] + g.box_l1 * enc["enc_l1"] + g.giou * enc["enc_giou"])
            items.update(enc)
        if "da_logits" in out:
            dense = self.dense_losses(out, batch)
            total = total + g.da * (dense["da_ce"] + dense["da_dice"])
            if "ll_ce" in dense:
                total = total + g.lane_mask * (dense["ll_ce"] + dense["ll_tversky"])
            items.update(dense)
        items["total"] = total
        return total, {k: float(v.detach()) for k, v in items.items()}

    def _weighted(self, items: Dict[str, torch.Tensor], p: str) -> torch.Tensor:
        g = self.g
        return (g.cls * items[f"{p}cls"] + g.box_l1 * items[f"{p}l1"] + g.giou * items[f"{p}giou"] + g.box3d * items[f"{p}box3d"]
                + g.el_cls * items[f"{p}el_cls"] + g.box_l1 * items[f"{p}el_l1"] + g.giou * items[f"{p}el_giou"]
                + g.el_attr * items[f"{p}el_attr"] + g.el_sign * items[f"{p}el_sign"]
                + g.lane_y * items[f"{p}lane_y"] + g.lane_z * items[f"{p}lane_z"] + g.lane_vis * items[f"{p}lane_vis"]
                + g.lane_exist * items[f"{p}lane_exist"] + g.lane_attr * items[f"{p}lane_attr"])


def sign_prefix_ids(names: Sequence[str]) -> List[int]:
    """Integer id of the ``<source>:`` prefix of each vocabulary name (names without a prefix share id 0)."""
    prefixes: Dict[str, int] = {}
    out = []
    for n in names:
        p = n.split(":", 1)[0] if ":" in n else ""
        out.append(prefixes.setdefault(p, len(prefixes)))
    return out
