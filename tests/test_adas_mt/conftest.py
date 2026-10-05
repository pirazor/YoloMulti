"""Shared fixtures: a synthetic dataset whose objects are colour-coded so that image, boxes and
masks can be cross-checked after augmentation (red rect == class 0 box == DA=1, green rect ==
class 1 box == LL=1)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

H0, W0 = 720, 1280


def make_split(root: Path, split: str, n: int, seed: int = 0, with_da: bool = True, with_ll: bool = True) -> None:
    rng = np.random.default_rng(seed)
    for d in ("images", "labels_det", "labels_da", "labels_ll"):
        (root / d / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = rng.integers(60, 120, (H0, W0, 3), dtype=np.uint8)
        img[...] = np.clip(img // 2 + 70, 0, 255)  # mid-grey texture, never pure red/green
        x1, y1 = int(rng.integers(100, 500)), int(rng.integers(100, 300))
        x2, y2 = x1 + int(rng.integers(150, 350)), y1 + int(rng.integers(100, 250))
        gx1, gy1 = int(rng.integers(600, 900)), int(rng.integers(350, 450))
        gx2, gy2 = gx1 + int(rng.integers(150, 300)), gy1 + int(rng.integers(100, 200))
        img[y1:y2, x1:x2] = (0, 0, 255)  # BGR red
        img[gy1:gy2, gx1:gx2] = (0, 255, 0)  # green
        stem = f"{split}_{i:03d}"
        cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 98])

        def yolo(c, a, b, cx, cy):
            return f"{c} {(a + cx) / 2 / W0:.6f} {(b + cy) / 2 / H0:.6f} {(cx - a) / W0:.6f} {(cy - b) / H0:.6f}"

        (root / "labels_det" / split / f"{stem}.txt").write_text(
            "\n".join([yolo(0, x1, y1, x2, y2), yolo(1, gx1, gy1, gx2, gy2)])
        )
        if with_da:
            da = np.zeros((H0, W0), np.uint8)
            da[y1:y2, x1:x2] = 1
            cv2.imwrite(str(root / "labels_da" / split / f"{stem}.png"), da)
        if with_ll:
            ll = np.zeros((H0, W0), np.uint8)
            ll[gy1:gy2, gx1:gx2] = 1
            cv2.imwrite(str(root / "labels_ll" / split / f"{stem}.png"), ll)


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth")
    make_split(root, "train", 8, seed=0)
    make_split(root, "val", 4, seed=1)
    data = {
        "path": str(root), "train": "images/train", "val": "images/val",
        "nc": 2, "names": ["red", "green"],
        "da_classes": 3, "da_names": ["background", "direct", "alternative"],
        "ll_classes": 3, "ll_names": ["background", "solid", "dashed"],
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data))
    return root


@pytest.fixture(scope="session")
def synth_data(synth_root) -> dict:
    return yaml.safe_load((synth_root / "data.yaml").read_text())


def make_hyp(**overrides):
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.utils import IterableSimpleNamespace

    base = dict(vars(DEFAULT_CFG))
    base.update(dict(mosaic=1.0, mixup=0.0, cutmix=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=3.0,
                     translate=0.1, scale=0.5, shear=0.0, perspective=0.0, fliplr=0.5, bgr=0.0))
    base.update(overrides)
    return IterableSimpleNamespace(**base)


@pytest.fixture
def tiny_supervisely(tmp_path):
    """One 720p image: a car box, a drivable-area polygon, one dashed lane line."""
    img_dir, ann_dir = tmp_path / "ds" / "img", tmp_path / "ds" / "ann"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    def write(name: str, objects: list):
        cv2.imwrite(str(img_dir / f"{name}.jpg"), np.full((H0, W0, 3), 80, np.uint8))
        (ann_dir / f"{name}.json").write_text(json.dumps({"size": {"height": H0, "width": W0}, "objects": objects}))

    car = {"classTitle": "car", "geometryType": "rectangle", "points": {"exterior": [[200, 100], [400, 200]]}, "tags": []}
    da = {"classTitle": "drivable area", "geometryType": "polygon",
          "points": {"exterior": [[100, 500], [600, 500], [600, 700], [100, 700]]},
          "tags": [{"name": "areaType", "value": "{'areaType': 'direct'}"}]}
    lane = {"classTitle": "lane", "geometryType": "line", "points": {"exterior": [[800, 200], [800, 600]]},
            "tags": [{"name": "laneAttrs", "value": "{'laneStyle': 'dashed', 'laneType': 'single white'}"}]}
    write("full", [car, da, lane])
    write("car_only", [car])
    return tmp_path / "ds"


def nontrivial_model(scale: str = "n", nc: int = 3, imgsz=(96, 160), seed: int = 0, det_bias: float = 3.0, calib=None):
    """A random-init model that behaves like a trained one for export/parity tests.

    A random-init YOLO in eval mode has vanishing activations, so every output is just a bias and every
    anchor ties (the tests would then compare arbitrary top-k picks). Training-mode batch statistics are
    copied into the BatchNorm running stats (momentum 1) so activations are normalised layer by layer, and the
    detection class biases are raised so some scores are confident. ``calib``: (N, 3, H, W) images in [0, 1] for the
    BatchNorm calibration instead of smooth random ones (a model calibrated on its own data domain has well-conditioned
    activations on it, like a trained one; on out-of-domain frames fp32 drift is amplified)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from adas_mt.nn import build_model

    torch.manual_seed(seed)
    model = build_model(scale, nc=nc)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = 1.0
    model.train()
    g = torch.Generator().manual_seed(seed)
    x = F.interpolate(torch.rand(8, 3, 6, 10, generator=g), size=tuple(imgsz), mode="bilinear", align_corners=False)
    if calib is not None:
        x = calib
    with torch.no_grad():
        model(x)
        for branch in (model.model[-1].one2one_cv3, model.model[-1].cv3):
            for seq in branch:
                seq[-1].bias.add_(det_bias)
        model.ll_head.cls.bias.zero_()  # drop the 99%-background prior: lane classes must appear in parity checks
    for b in bns:
        b.momentum = 0.1
    return model.eval()


def structured_frame(imgsz=(96, 160), seed: int = 1):
    """(1,3,H,W) smooth random image in [0,1] (spatially varying, unlike per-pixel noise)."""
    import torch
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(seed)
    return F.interpolate(torch.rand(1, 3, 6, 10, generator=g), size=tuple(imgsz), mode="bilinear", align_corners=False)
