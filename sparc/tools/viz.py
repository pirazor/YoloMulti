"""Drawing helpers for notebooks: training items (labels in the virtual camera) and model predictions."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch

from adas_mt.data.masks import unpack_masks
from sparc.data.dataset import class_to_order
from sparc.data.geometry import anchors_to_polyline
from sparc.data.schema import (AGENT_CLASSES, DA_CLASSES, EL_LIGHT, ELEMENT_ATTRS, LANE_MASK_CLASSES, LANE_X_ANCHORS)

DA_COLORS = {1: (42, 120, 214), 2: (235, 104, 52)}                     # direct, alternative (RGB)
AGENT_COLOR = (255, 255, 255)
ATTR_COLORS = {"red": (230, 40, 40), "green": (40, 210, 90), "yellow": (240, 200, 30)}
SIGN_COLOR = (180, 120, 255)
ORDER_COLORS = {-3: (120, 60, 200), -2: (160, 90, 240), -1: (200, 150, 255), 1: (255, 200, 120), 2: (255, 160, 60),
                3: (220, 110, 20)}
LANE_DEFAULT = (250, 250, 120)


def _overlay_da(img: np.ndarray, da: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = img.copy()
    for c, col in DA_COLORS.items():
        m = da == c
        out[m] = (out[m] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)
    return out


def _text(img, txt, org, color, scale=0.4):
    x, y = int(org[0]), int(max(org[1], 10))
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _cxcywh_px(b: np.ndarray, w: int, h: int) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h, (b[:, 0] + b[:, 2] / 2) * w,
                     (b[:, 1] + b[:, 3] / 2) * h], 1)


def draw_item(item: Dict, sign_names: Optional[Sequence[str]] = None) -> np.ndarray:
    """A dataset item (``SparcDataset.__getitem__``) with its labels, in the network frame."""
    img = item["image"].permute(1, 2, 0).numpy().copy()
    h, w = img.shape[:2]
    da, ll = unpack_masks(item["mask"][None], len(DA_CLASSES), len(LANE_MASK_CLASSES))
    img = _overlay_da(img, da[0].numpy())
    llm = ll[0].numpy() == 1
    img[llm] = (img[llm] * 0.5 + np.array((255, 255, 0)) * 0.5).astype(np.uint8)
    K, T = item["K"].numpy(), item["cam2ego"].numpy()
    for j in range(len(item["lane_y"])):
        poly = anchors_to_polyline(item["lane_y"][j].numpy(), item["lane_z"][j].numpy(), item["lane_vis"][j].numpy(), K, T, (h, w))
        if len(poly) >= 2:
            o = int(item["lane_attr"][j, 0])
            col = ORDER_COLORS.get(class_to_order(o), LANE_DEFAULT) if o >= 0 else LANE_DEFAULT
            cv2.polylines(img, [np.round(poly).astype(np.int32)], False, col, 2, cv2.LINE_AA)
    for b, c in zip(_cxcywh_px(item["boxes"].numpy(), w, h), item["cls"].tolist()):
        cv2.rectangle(img, tuple(np.round(b[:2]).astype(int)), tuple(np.round(b[2:]).astype(int)), AGENT_COLOR, 1)
        _text(img, AGENT_CLASSES[c], b[:2] - (0, 2), AGENT_COLOR, 0.35)
    for b, c, a, s in zip(_cxcywh_px(item["el_boxes"].numpy(), w, h), item["el_cls"].tolist(), item["el_attr"].tolist(),
                          item["el_sign"].tolist()):
        col = ATTR_COLORS.get(ELEMENT_ATTRS[a] if a >= 0 else "", SIGN_COLOR if c != EL_LIGHT else (200, 200, 200))
        cv2.rectangle(img, tuple(np.round(b[:2]).astype(int)), tuple(np.round(b[2:]).astype(int)), col, 2)
        if s >= 0 and sign_names:
            _text(img, sign_names[s].split(":")[-1][:24], b[:2] - (0, 2), col, 0.3)
    return img


@torch.no_grad()
def draw_prediction(item: Dict, out: Dict[str, torch.Tensor], b: int = 0, sign_names: Optional[Sequence[str]] = None,
                    score: float = 0.35, lane_exist: float = 0.4) -> np.ndarray:
    """Raw network outputs (``SparcNet.forward`` in eval mode, batch index ``b``) drawn on the item's network image."""
    img = item["image"].permute(1, 2, 0).numpy().copy()
    h, w = img.shape[:2]
    img = _overlay_da(img, out["da_logits"][b].argmax(0).cpu().numpy())
    K, T = item["K"].numpy(), item["cam2ego"].numpy()
    ex = out["lane_exist_logits"][b].sigmoid().cpu().numpy()
    for j in np.where(ex > lane_exist)[0]:
        vis = out["lane_vis_logits"][b, j].sigmoid().cpu().numpy() > 0.5
        poly = anchors_to_polyline(out["lane_y"][b, j].cpu().numpy(), out["lane_z"][b, j].cpu().numpy(), vis, K, T, (h, w),
                                   LANE_X_ANCHORS)
        if len(poly) >= 2:
            o = class_to_order(int(out["lane_order_logits"][b, j].argmax()))
            cv2.polylines(img, [np.round(poly).astype(np.int32)], False, ORDER_COLORS.get(o, LANE_DEFAULT), 2, cv2.LINE_AA)
            _text(img, f"{o:+d}", poly[0], ORDER_COLORS.get(o, LANE_DEFAULT), 0.35)
    sc, cl = out["logits"][b].sigmoid().max(-1)
    keep = (sc > score).cpu().numpy()
    for bx, c, s in zip(_cxcywh_px(out["boxes"][b].cpu().numpy()[keep], w, h), cl.cpu().numpy()[keep], sc.cpu().numpy()[keep]):
        cv2.rectangle(img, tuple(np.round(bx[:2]).astype(int)), tuple(np.round(bx[2:]).astype(int)), AGENT_COLOR, 1)
        _text(img, f"{AGENT_CLASSES[c]} {s:.2f}", bx[:2] - (0, 2), AGENT_COLOR, 0.33)
    es, ec = out["el_logits"][b].sigmoid().max(-1)
    ek = (es > score).cpu().numpy()
    attrs = out["el_attr_logits"][b].argmax(-1).cpu().numpy()[ek]
    signs = out["el_sign_logits"][b].argmax(-1).cpu().numpy()[ek]
    for bx, c, a, s in zip(_cxcywh_px(out["el_boxes"][b].cpu().numpy()[ek], w, h), ec.cpu().numpy()[ek], attrs, signs):
        name = ELEMENT_ATTRS[a]
        col = ATTR_COLORS.get(name, SIGN_COLOR if c != EL_LIGHT else (200, 200, 200))
        cv2.rectangle(img, tuple(np.round(bx[:2]).astype(int)), tuple(np.round(bx[2:]).astype(int)), col, 2)
        label = name if c == EL_LIGHT else (sign_names[s].split(":")[-1][:24] if sign_names else "sign")
        _text(img, label, bx[:2] - (0, 2), col, 0.3)
    return img
