# Multi-task YOLOv13 (detection + drivable-area + lane segmentation)

This is a non-invasive extension of [YOLOv13](README.md) that adds two
semantic-segmentation heads on top of the v13 backbone+neck+detect head:

- **Detection** (existing v13 head) — coarse driving classes
  (`car, truck, bus, motorcycle, bicycle, pedestrian, rider, traffic_light, traffic_sign`).
- **Drivable area** — 3 classes: `background, direct, alternative`.
- **Lane** — 3 classes by default (`background, solid, dashed`); configurable.

Plus a separate fine-grained traffic-sign classifier ([`sign_classifier/`](sign_classifier/README.md)).

All new code lives under `yolov13_multitask/` and `sign_classifier/`. **No
upstream files in `ultralytics/` are modified** — `git pull` against upstream
stays clean.

## Architecture

```
                          input image
                              │
                              ▼
            ┌──────────────────────────────────┐
            │  YOLOv13 backbone + neck (FPN)   │   ◀── unchanged upstream
            └──────────────────────────────────┘
              │             │             │
            P3(1/8)       P4(1/16)      P5(1/32)
              │             │             │
              ├─────────────┴─────────────┤
              │                           │
              ▼                           ▼
   ┌──────────────────┐       ┌────────────────────┐
   │   Detect head    │       │   FPN-fused seg    │  ×2 (DA + LL)
   │   (unchanged)    │       │  decoder upsampled │
   └──────────────────┘       │  to input res.     │
              │               └────────────────────┘
              ▼                           │
       det predictions                    ▼
                                  semantic logits
                                  (B, C, H, W)
```

The seg decoder is a 1×1 reduce-and-fuse on P3/P4/P5 (P4 and P5 are
bilinearly upsampled to P3's spatial size first), then 3× (Conv-BN-SiLU + 2×
upsample) back to the input resolution, then a 1×1 classifier. Two
independent decoders — one per task.

## Quickstart

```bash
# 0. Branch and install (upstream YOLOv13 install procedure works as-is)
git checkout multitask
pip install -r requirements.txt
# (the multitask package is pure-source; no extra setup is needed)

# 1. Convert your Supervisely dataset
python -m yolov13_multitask convert \
    --src /path/to/supervisely \
    --dst /path/to/dataset_yolo \
    --lane_grouping style

# 2. Train
python -m yolov13_multitask train \
    --data /path/to/dataset_yolo/data.yaml \
    --cfg yolov13_multitask/cfg/multitask.yaml \
    --weights yolov13s.pt \
    --epochs 100 --batch 16 --imgsz 640

# 3. Validate
python -m yolov13_multitask val \
    --weights runs/multitask/exp/weights/best.pt \
    --data /path/to/dataset_yolo/data.yaml

# 4. Predict + visualize
python -m yolov13_multitask predict \
    --weights runs/multitask/exp/weights/best.pt \
    --source /path/to/video.mp4 --save

# 5. Export
python -m yolov13_multitask export \
    --weights runs/multitask/exp/weights/best.pt \
    --format onnx
```

## Tests

```bash
pytest tests/multitask -q
ls tests/_out/multitask/aug   # open these PNGs to visually verify augmentation alignment
```

The visual sanity test (`test_dataset_visual.py`) is the most important: it
saves 8 augmented samples as side-by-side overlays of (image+boxes),
(image+drivable-area), (image+lane). Open them by eye and confirm boxes and
masks remain spatially aligned through mosaic + affine + flip.

## More documentation

- [docs/architecture.md](docs/architecture.md) — head designs, loss rationale.
- [docs/dataset.md](docs/dataset.md) — Supervisely conversion details, lane-grouping options, sign-crop workflow.
- [docs/training.md](docs/training.md) — hyperparameter notes, loss balancing, common failure modes.
