"""Visual sanity test for the multi-task dataloader.

This is the most important test in the suite. Multi-task augmentation bugs are
silent killers (boxes drift away from masks, masks drift away from each
other), and the only way to catch them reliably is to **look at the data**.

The test:
    1. Builds a tiny dataset on disk via the converter.
    2. Instantiates ``MultiTaskDataset`` with augmentation enabled.
    3. Pulls 8 augmented samples and writes side-by-side overlays of
       (image+boxes), (image+da mask), (image+ll mask) to
       ``tests/_out/multitask/aug/``.

The test PASSES if the loop runs without error. A human still needs to open
the resulting PNGs and verify alignment looks correct — see ``docs/training.md``
for what to look for.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _synth_supervisely(root: Path, n: int = 16) -> Path:
    src = root / "supervisely"
    img_dir = src / "img"
    ann_dir = src / "ann"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    H, W = 360, 640
    rng = np.random.default_rng(0)
    for i in range(n):
        img = (rng.uniform(20, 100, (H, W, 3))).astype(np.uint8)
        # paint a couple of "objects"
        x1 = int(rng.integers(50, 200))
        y1 = int(rng.integers(50, 150))
        x2 = x1 + int(rng.integers(60, 120))
        y2 = y1 + int(rng.integers(40, 80))
        img[y1:y2, x1:x2] = (200, 200, 200)
        cv2.imwrite(str(img_dir / f"img_{i:02d}.jpg"), img)

        ann = {
            "size": {"height": H, "width": W},
            "objects": [
                {
                    "classTitle": "car",
                    "geometryType": "rectangle",
                    "points": {"exterior": [[x1, y1], [x2, y2]]},
                    "tags": [],
                },
                {
                    "classTitle": "drivable area",
                    "geometryType": "polygon",
                    "points": {"exterior": [[100, 220], [540, 220], [540, 320], [100, 320]]},
                    "tags": [{"name": "areaType", "value": "{'areaType': 'direct'}"}],
                },
                {
                    "classTitle": "lane",
                    "geometryType": "line",
                    "points": {"exterior": [[160, 200], [220, 320]]},
                    "tags": [
                        {
                            "name": "laneAttrs",
                            "value": "{'laneStyle': 'solid', 'laneType': 'single white'}",
                        }
                    ],
                },
            ],
        }
        with (ann_dir / f"img_{i:02d}.json").open("w") as f:
            json.dump(ann, f)
    return src


def test_visual_aug(out_dir, tmp_path):
    from yolov13_multitask.data.convert_supervisely import convert
    from yolov13_multitask.data.multitask_dataset import MultiTaskDataset
    from yolov13_multitask.utils.visualize import (
        draw_boxes,
        overlay_drivable_area,
        overlay_lane_mask,
    )

    src = _synth_supervisely(tmp_path)
    dst = tmp_path / "ds_yolo"
    convert(src=src, dst=dst, lane_grouping="style", val_fraction=0.2, copy_images=True)

    ds = MultiTaskDataset(dst / "data.yaml", split="train", imgsz=320, augment=True, mosaic_prob=0.5)
    aug_dir = out_dir / "aug"
    aug_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(8, len(ds))):
        sample = ds[i]
        img_chw = sample["img"]
        rgb = (img_chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        da = sample["da_mask"].cpu().numpy().astype(np.uint8)
        ll = sample["ll_mask"].cpu().numpy().astype(np.uint8)
        bxs = sample["bboxes"].cpu().numpy()
        cls = sample["cls"].cpu().numpy().reshape(-1)

        a = draw_boxes(rgb, bxs, cls, [1.0] * len(cls), names=ds.names, color=(255, 0, 0))
        b = overlay_drivable_area(rgb, da)
        c = overlay_lane_mask(rgb, ll)
        side = np.concatenate([a, b, c], axis=1)
        cv2.imwrite(str(aug_dir / f"sample_{i:02d}.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

    # collect and check the files exist
    files = sorted(aug_dir.glob("sample_*.png"))
    assert len(files) > 0
