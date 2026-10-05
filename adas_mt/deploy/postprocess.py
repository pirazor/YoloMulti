"""Host-side top-k for the ``--det-head raw`` export (numpy only).

The default export keeps the NMS-free top-k inside the graph (``det`` is ``(B, 300, 6)``). TensorRT 10.3.0 (JetPack 6.x)
is reported to fail building INT8 engines for graphs with that head (Ultralytics issue 23841), so ``--det-head raw``
ends the graph at the dense one-to-one predictions ``(B, A, 4 + nc)`` (decoded ``x1 y1 x2 y2`` in network pixels and
per-class sigmoid scores) and this module does the same selection on the host: exactly the non-agnostic
``Detect.get_topk_index`` + ``postprocess`` of Ultralytics 8.4.
"""

from __future__ import annotations

import numpy as np


def _topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest entries of a 1-D array, best first."""
    if k >= x.size:
        return np.argsort(-x, kind="stable")[:k]
    part = np.argpartition(-x, k - 1)[:k]
    return part[np.argsort(-x[part], kind="stable")]


def topk_det(raw: np.ndarray, max_det: int = 300) -> np.ndarray:
    """``(A, 4 + nc)`` dense predictions -> ``(min(max_det, A), 6)`` rows ``x1 y1 x2 y2 score class``.

    Step 1 keeps the ``k`` anchors with the highest best-class score; step 2 takes the ``k`` highest
    (anchor, class) scores among them, so one anchor may yield several classes, like the in-graph version."""
    boxes, scores = raw[:, :4], raw[:, 4:]
    n_anchors, nc = scores.shape
    k = min(int(max_det), n_anchors)
    anchors = _topk_indices(scores.max(axis=1), k)
    flat = scores[anchors].reshape(-1)
    idx = _topk_indices(flat, k)
    out = np.empty((k, 6), np.float32)
    out[:, :4] = boxes[anchors[idx // nc]]
    out[:, 4] = flat[idx]
    out[:, 5] = idx % nc
    return out


def topk_det_batch(raw: np.ndarray, max_det: int = 300) -> np.ndarray:
    """``(B, A, 4 + nc)`` -> ``(B, k, 6)``."""
    return np.stack([topk_det(r, max_det) for r in np.asarray(raw)])


def decode_det(det: np.ndarray, meta: dict) -> np.ndarray:
    """The ``det`` output of any export -> ``(B, k, 6)`` rows ``x1 y1 x2 y2 score class`` (network pixels)."""
    det = np.asarray(det)
    if meta.get("det_head", "topk") == "raw":
        return topk_det_batch(det, int(meta["max_det"]))
    return det
