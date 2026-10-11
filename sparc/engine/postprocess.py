"""Decoder outputs -> per-image predictions in the native image frame (for evaluation and visualisation)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from sparc.data.geometry import anchors_to_polyline
from sparc.data.schema import LANE_X_ANCHORS
from sparc.nn.boxes import cxcywh_to_xyxy


def _to_native(xyxy_virtual: np.ndarray, A: np.ndarray, hw_native) -> np.ndarray:
    """Boxes in the network image -> native pixels through the inverse of the sample's affine map."""
    if len(xyxy_virtual) == 0:
        return xyxy_virtual.reshape(0, 4)
    Ai = np.linalg.inv(np.asarray(A, dtype=np.float64))
    pts = xyxy_virtual.reshape(-1, 2)
    pts = pts @ Ai[:2, :2].T + Ai[:2, 2]
    b = pts.reshape(-1, 4)
    out = np.concatenate([np.minimum(b[:, :2], b[:, 2:]), np.maximum(b[:, :2], b[:, 2:])], 1)
    h, w = hw_native
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w - 1)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h - 1)
    return out


@torch.no_grad()
def postprocess(out: Dict[str, torch.Tensor], batch: Dict, min_score: float = 0.01, lane_exist: float = 0.3,
                lane_vis: float = 0.5) -> List[Dict]:
    """One dict per image: agents / elements (native xyxy, score, class, extras), lanes (native polylines + attrs)."""
    B = out["logits"].shape[0]
    H, W = batch["image"].shape[-2:]
    res = []
    scale = np.array([W, H, W, H], dtype=np.float64)
    for b in range(B):
        meta = batch["meta"][b]
        A, hw, Kn, Tn = meta["A"], meta["hw"], meta["K_native"], meta["cam2ego_native"]
        # agents
        score, cls = out["logits"][b].sigmoid().max(-1)
        keep = score > min_score
        boxes = cxcywh_to_xyxy(out["boxes"][b][keep]).cpu().numpy() * scale
        agents = {"boxes": _to_native(boxes, A, hw), "scores": score[keep].cpu().numpy(), "cls": cls[keep].cpu().numpy(),
                  "box3d": out["box3d"][b][keep].cpu().numpy()}
        # elements
        es, ec = out["el_logits"][b].sigmoid().max(-1)
        ek = es > min_score
        eb = cxcywh_to_xyxy(out["el_boxes"][b][ek]).cpu().numpy() * scale
        elements = {"boxes": _to_native(eb, A, hw), "scores": es[ek].cpu().numpy(), "cls": ec[ek].cpu().numpy(),
                    "attr": out["el_attr_logits"][b][ek].argmax(-1).cpu().numpy(),
                    "sign": out["el_sign_logits"][b][ek].argmax(-1).cpu().numpy()}
        # lanes: ego anchors -> native polyline
        ex = out["lane_exist_logits"][b].sigmoid().cpu().numpy()
        ys, zs = out["lane_y"][b].cpu().numpy(), out["lane_z"][b].cpu().numpy()
        vis = out["lane_vis_logits"][b].sigmoid().cpu().numpy() > lane_vis
        lanes = []
        for j in np.argsort(-ex):
            if ex[j] < lane_exist:
                break
            poly = anchors_to_polyline(ys[j], zs[j], vis[j], Kn, Tn, hw, LANE_X_ANCHORS)
            if len(poly) < 2:
                continue
            lanes.append({"points": poly, "score": float(ex[j]), "y": ys[j], "z": zs[j], "vis": vis[j],
                          "order": int(out["lane_order_logits"][b, j].argmax()), "dir": int(out["lane_dir_logits"][b, j].argmax()),
                          "cont": int(out["lane_cont_logits"][b, j].argmax()), "type": int(out["lane_type_logits"][b, j].argmax())})
        res.append({"agents": agents, "elements": elements, "lanes": lanes, "meta": meta})
    return res
