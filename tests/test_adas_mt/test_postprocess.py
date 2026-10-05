"""Host-side top-k (``--det-head raw``) must equal the in-graph NMS-free top-k of Ultralytics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adas_mt.deploy.postprocess import decode_det, topk_det, topk_det_batch


def _reference(raw: np.ndarray, max_det: int, nc: int) -> np.ndarray:
    """The real ``Detect.get_topk_index`` + ``postprocess`` on the same dense tensor."""
    from ultralytics.nn.modules.head import Detect

    det = Detect(nc=nc, ch=(16, 32, 64))
    det.max_det, det.agnostic_nms, det.export, det.format = max_det, False, False, None
    return det.postprocess(torch.from_numpy(raw)[None])[0].numpy()


def _rows(a):  # order-insensitive view of the rows
    return sorted(map(tuple, np.round(a, 5).tolist()))


@pytest.mark.parametrize("anchors,nc,max_det", [(5040, 9, 300), (315, 3, 300), (42, 2, 300), (1000, 80, 100), (7, 4, 300)])
def test_matches_the_in_graph_topk(anchors, nc, max_det):
    rng = np.random.default_rng(anchors)
    raw = np.concatenate([rng.uniform(0, 640, (anchors, 4)), rng.uniform(0, 1, (anchors, nc))], 1).astype(np.float32)
    got = topk_det(raw, max_det)
    want = _reference(raw, max_det, nc)
    assert got.shape == want.shape == (min(max_det, anchors), 6)
    assert _rows(got) == _rows(want)
    assert (np.diff(got[:, 4]) <= 0).all(), "rows are sorted by score, best first"


def test_one_anchor_can_yield_several_classes_and_ties_are_deterministic():
    raw = np.zeros((4, 4 + 3), np.float32)
    raw[:, :4] = np.arange(16).reshape(4, 4)
    raw[0, 4:] = [0.9, 0.8, 0.1]  # anchor 0 is confident in two classes
    raw[1, 4:] = [0.5, 0.5, 0.5]  # an exact tie
    out = topk_det(raw, 3)
    assert out[:2, 4].tolist() == pytest.approx([0.9, 0.8]) and out[:2, 5].tolist() == [0, 1]
    assert (out[:2, :4] == raw[0, :4]).all()
    assert (topk_det(raw, 3) == out).all()


def test_batch_and_decode_helpers():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, (2, 50, 4 + 3)).astype(np.float32)
    assert topk_det_batch(raw, 10).shape == (2, 10, 6)
    topk = np.zeros((1, 300, 6), np.float32)
    assert decode_det(topk, {"det_head": "topk"}) is topk or (decode_det(topk, {}) == topk).all()
    assert decode_det(raw, {"det_head": "raw", "max_det": 10}).shape == (2, 10, 6)
