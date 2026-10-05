import numpy as np
import torch

from adas_mt.engine.metrics import SegConfusion


def _reference(pred, target, nc):
    """Plain numpy IoU, ignoring pixels whose target is not a valid class."""
    m = np.zeros((nc, nc), np.int64)
    for p, t in zip(pred.ravel(), target.ravel()):
        if t < nc:
            m[t, p] += 1
    return m


def test_confusion_matches_numpy_reference_and_ignores_255():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 3, (4, 20, 30))
    target[:, :, 25:] = 255  # padding / unannotated
    pred = rng.integers(0, 3, (4, 20, 30))
    c = SegConfusion(3, ["background", "direct", "alternative"])
    c.update(torch.from_numpy(pred), torch.from_numpy(target))
    ref = _reference(pred, target, 3)
    assert np.array_equal(c.mat.numpy(), ref)
    r = c.results("da")
    tp = np.diag(ref)
    iou = tp / (ref.sum(0) + ref.sum(1) - tp)
    assert np.isclose(r["da_IoU_direct"], iou[1]) and np.isclose(r["da_IoU_background"], iou[0])
    assert np.isclose(r["da_mIoU"], iou[1:].mean())  # background excluded from mIoU
    inter = ref[1:, 1:].sum()
    assert np.isclose(r["da_IoU_fg"], inter / (ref[1:].sum() + ref[:, 1:].sum() - inter))
    assert np.isclose(r["da_pixel_acc"], tp.sum() / ref.sum())


def test_streaming_equals_one_shot_and_perfect_prediction_is_one():
    rng = np.random.default_rng(1)
    target = torch.from_numpy(rng.integers(0, 3, (6, 16, 16)))
    one, streamed = SegConfusion(3, []), SegConfusion(3, [])
    one.update(target, target)
    for i in range(6):
        streamed.update(target[i : i + 1], target[i : i + 1])
    assert torch.equal(one.mat, streamed.mat)
    r = one.results("ll")
    assert r["ll_mIoU"] == 1.0 and r["ll_IoU_fg"] == 1.0 and r["ll_recall_fg"] == 1.0 and r["ll_pixel_acc"] == 1.0


def test_classes_without_ground_truth_do_not_drag_down_miou():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1  # only 'direct' exists; 'alternative' never appears
    c = SegConfusion(3, ["background", "direct", "alternative"])
    c.update(target, target)
    assert c.results("da")["da_mIoU"] == 1.0
