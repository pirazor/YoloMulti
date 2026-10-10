"""Synthetic Supervisely dump that looks like a 1280x720 ADAS frame, for smoke tests and the Colab demo.

Each frame has a road trapezoid (DA "direct"), an adjacent lane area (DA "alternative"), solid and dashed
lane lines drawn as polylines, cars / trucks / buses on the road, pedestrians on the sidewalk, traffic
lights and traffic signs. The objects are painted into the image, so all three tasks are learnable and a
training run that works shows rising mAP / DA IoU / lane IoU within a few epochs. Some frames lack the lane
or the DA annotation to exercise ``--partial_annotation``. File names are ``<clip>-<frame>`` (4 frames per
clip) so a clip-aware split can be tested with ``--group_regex '^([0-9]{4})-'``.

    python -m adas_mt.data.synthetic --dst /data/synthetic_sup --n 600
    python -m adas_mt convert -- --src /data/synthetic_sup --dst /data/synthetic --partial_annotation \
        --group_regex '^([0-9]{4})-'
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

H, W = 720, 1280
GROUP_REGEX = r"^([0-9]{4})-"


def _scene(rng: random.Random, seed: int):
    img = np.empty((H, W, 3), np.uint8)
    t = np.linspace(0.0, 1.0, H)[:, None]
    img[..., 0] = (90 + 100 * t).astype(np.uint8)  # BGR sky -> ground gradient
    img[..., 1] = (120 + 60 * t).astype(np.uint8)
    img[..., 2] = (170 - 60 * t).astype(np.uint8)
    horizon = rng.randint(280, 360)
    objs = []
    cx = rng.randint(560, 720)
    top_w, bot_w = rng.randint(60, 120), rng.randint(500, 700)
    road = [[cx - top_w, horizon], [cx + top_w, horizon], [cx + bot_w, H - 1], [cx - bot_w, H - 1]]
    cv2.fillPoly(img, [np.array(road, np.int32)], (70, 70, 72))
    objs.append(_poly("drivable area", road, "areaType", "direct"))
    side = rng.choice((-1, 1))
    alt = [[cx + side * top_w, horizon], [cx + side * (top_w + 50), horizon], [cx + side * (bot_w + 350), H - 1],
           [cx + side * bot_w, H - 1]]
    cv2.fillPoly(img, [np.array(alt, np.int32)], (85, 80, 80))
    objs.append(_poly("drivable area", alt, "areaType", "alternative"))
    lines = [
        ([[cx - top_w, horizon], [cx - bot_w, H - 1]], "solid", "single white"),
        ([[cx + top_w, horizon], [cx + bot_w, H - 1]], "solid", "single yellow"),
        ([[cx, horizon], [cx + rng.randint(-40, 40), H - 1]], "dashed", "single white"),
    ]
    for pts, style, ltype in lines:
        color = (240, 240, 240) if "white" in ltype else (60, 220, 240)
        if style == "dashed":  # painted as dashes, annotated as one line (like real lane labels)
            p0, p1 = np.array(pts[0], float), np.array(pts[1], float)
            for k in range(0, 20, 2):
                a, b = p0 + (p1 - p0) * k / 20, p0 + (p1 - p0) * (k + 1) / 20
                cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 6)
        else:
            cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, 6)
        objs.append({"classTitle": "lane", "geometryType": "line", "points": {"exterior": pts, "interior": []},
                     "tags": [{"name": "laneAttrs", "value": f"{{'laneStyle': '{style}', 'laneType': '{ltype}'}}"}]})
    for _ in range(rng.randint(1, 4)):  # vehicles on the road, bigger when closer
        y2 = rng.randint(horizon + 40, H - 20)
        s = (y2 - horizon) / (H - horizon)
        bw, bh = int(60 + 260 * s), int(50 + 180 * s)
        x1 = rng.randint(max(0, cx - int(bot_w * s) - bw), max(1, min(W - bw - 1, cx + int(bot_w * s))))
        y1 = y2 - bh
        kind = rng.choices(["car", "truck", "bus"], [0.7, 0.2, 0.1])[0]
        cv2.rectangle(img, (x1, y1), (x1 + bw, y2), {"car": (200, 40, 40), "truck": (40, 160, 200), "bus": (40, 200, 160)}[kind], -1)
        cv2.rectangle(img, (x1 + 5, y1 + 5), (x1 + bw - 5, y1 + bh // 3), (20, 20, 20), -1)  # windscreen
        objs.append(_rect(kind, x1, y1, x1 + bw, y2))
    for _ in range(rng.randint(0, 2)):  # pedestrians on the sidewalk
        x1 = rng.choice([rng.randint(20, 200), rng.randint(W - 220, W - 60)])
        y1 = rng.randint(horizon - 20, horizon + 150)
        bw, bh = rng.randint(18, 40), rng.randint(60, 140)
        cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), (30, 30, 180), -1)
        cv2.circle(img, (x1 + bw // 2, y1), bw // 2, (120, 160, 220), -1)
        objs.append(_rect("person", x1, y1 - bw // 2, x1 + bw, y1 + bh))
    for _ in range(rng.randint(0, 2)):  # traffic lights: small, high
        x1, y1 = rng.randint(100, W - 100), rng.randint(60, horizon - 80)
        bw, bh = rng.randint(10, 22), rng.randint(26, 60)
        cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), (20, 20, 20), -1)
        col = rng.choice([(0, 0, 255), (0, 255, 0), (0, 200, 255)])
        cv2.circle(img, (x1 + bw // 2, y1 + bh // 4), max(3, bw // 3), col, -1)
        objs.append(_rect("traffic light", x1, y1, x1 + bw, y1 + bh))
    for _ in range(rng.randint(0, 2)):  # traffic signs: round, red rim
        x1, y1 = rng.randint(50, W - 100), rng.randint(horizon - 150, horizon + 20)
        bw = rng.randint(16, 50)
        cv2.circle(img, (x1 + bw // 2, y1 + bw // 2), bw // 2, (0, 0, 220), -1)
        cv2.circle(img, (x1 + bw // 2, y1 + bw // 2), bw // 3, (255, 255, 255), -1)
        objs.append(_rect("traffic sign", x1, y1, x1 + bw, y1 + bw))
    img = cv2.add(img, np.random.default_rng(seed).integers(0, 12, (H, W, 3), dtype=np.uint8))  # sensor noise
    return img, objs


def _poly(title, pts, tag, value):
    return {"classTitle": title, "geometryType": "polygon", "points": {"exterior": pts, "interior": []},
            "tags": [{"name": tag, "value": f"{{'{tag}': '{value}'}}"}]}


def _rect(title, x1, y1, x2, y2):
    return {"classTitle": title, "geometryType": "rectangle", "points": {"exterior": [[x1, y1], [x2, y2]], "interior": []},
            "tags": []}


def make_supervisely(dst: str | Path, n: int, seed: int = 0, partial: bool = True) -> Path:
    """Write ``n`` frames as ``dst/ds/img/*.jpg`` + ``dst/ds/ann/*.jpg.json`` (Supervisely layout). Returns ``dst``."""
    dst = Path(dst)
    (dst / "ds" / "img").mkdir(parents=True, exist_ok=True)
    (dst / "ds" / "ann").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        img, objs = _scene(rng, seed * 100003 + i)
        name = f"{i // 4:04d}-{i:05d}"
        if partial and i % 7 == 3:  # frame annotated for detection + DA only
            objs = [o for o in objs if o["classTitle"] != "lane"]
        if partial and i % 11 == 5:  # frame annotated for detection + lanes only
            objs = [o for o in objs if o["classTitle"] != "drivable area"]
        cv2.imwrite(str(dst / "ds" / "img" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (dst / "ds" / "ann" / f"{name}.jpg.json").write_text(json.dumps({"size": {"height": H, "width": W}, "objects": objs}))
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-partial", dest="partial", action="store_false", help="annotate every task on every frame")
    a = ap.parse_args(argv)
    make_supervisely(a.dst, a.n, a.seed, a.partial)
    print(f"wrote {a.n} frames to {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
