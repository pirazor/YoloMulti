"""Pytest config for multitask tests.

Kept separate from upstream ``tests/conftest.py`` so we don't entangle with the
ultralytics test fixtures.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def out_dir() -> Path:
    """Directory where visual sanity outputs are written."""
    p = Path(__file__).resolve().parent.parent / "_out" / "multitask"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def tiny_supervisely(tmp_path):
    """Synthesize a 1-image Supervisely-style dataset for converter tests."""
    import json

    import cv2
    import numpy as np

    img_dir = tmp_path / "ds" / "img"
    ann_dir = tmp_path / "ds" / "ann"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    H, W = 720, 1280
    img = np.full((H, W, 3), 80, dtype=np.uint8)
    img[100:200, 200:400] = 255  # decorate
    cv2.imwrite(str(img_dir / "frame.jpg"), img)
    ann = {
        "size": {"height": H, "width": W},
        "objects": [
            {  # car bbox
                "classTitle": "car",
                "geometryType": "rectangle",
                "points": {"exterior": [[200, 100], [400, 200]]},
                "tags": [],
            },
            {  # drivable area, direct
                "classTitle": "drivable area",
                "geometryType": "polygon",
                "points": {"exterior": [[100, 500], [600, 500], [600, 700], [100, 700]]},
                "tags": [{"name": "areaType", "value": "{'areaType': 'direct'}"}],
            },
            {  # lane, dashed line
                "classTitle": "lane",
                "geometryType": "line",
                "points": {"exterior": [[800, 200], [900, 600]]},
                "tags": [
                    {
                        "name": "laneAttrs",
                        "value": "{'laneStyle': 'dashed', 'laneType': 'single white', 'laneDirection': 'vertical'}",
                    }
                ],
            },
        ],
    }
    with (ann_dir / "frame.json").open("w") as f:
        json.dump(ann, f)
    return tmp_path / "ds"
