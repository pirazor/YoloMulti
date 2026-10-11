# ADAS multi-task perception for Jetson Orin Nano

Detection + drivable area (direct / alternative) + lane (solid / dashed) from one
YOLO26-based CNN, trained with dense foundation-model distillation (training-time
only) and deployed as a TensorRT FP16/INT8 engine with NMS-free detection and
in-graph class-map outputs (int32, or uint8 on TensorRT >= 10).

Status: **v0.2**. Phases 0-5 are implemented: re-platform, [data](docs/data.md), [model](docs/architecture.md), [distillation](docs/distillation.md), [trainer / validator](docs/training.md) and [export / Jetson deployment](docs/deployment_jetson.md). Everything that can be verified on a CPU is tested; the TensorRT engine build, its accuracy and the latency on the Orin are **not yet measured** (commands and a results table are in the deployment doc).
The previous YOLOv13-based implementation lives in [`legacy/yolov13/`](legacy/yolov13/)
(tag `yolov13-mt-baseline`) and is kept as the A0 baseline for ablations.

```
sparc/          next-generation network (design: docs/sparc/plan.md), Phase 0: bake-off tooling, rig geometry, schema
adas_mt/        YOLO26-based multi-task model (data, nn, distill, engine, export, deploy); the A0 baseline for SPARC
sign_classifier/ fine-grained traffic-sign classifier (unchanged)
recording/      recording specification for own drives (any stereo + IMU + GNSS rig)
legacy/yolov13/ previous YOLOv13 multi-task implementation (frozen baseline)
```

## SPARC (in progress)

One temporal sparse-query network for a front camera + IMU + GNSS on a Jetson Orin: road users with 3D state and
tracking, traffic-light colour and fine sign class, lanes with ego order and travel direction, drivable area, a BEV
map and light-to-lane topology. Depth is an auxiliary signal (optional input, training target), never a dependency;
training starts on public datasets ([design and plan](docs/sparc/plan.md)). Phase 0
([report](docs/sparc/phase0.md)) measures the backbone candidates and the TensorRT-critical ops on the device;
stage A ([guide](docs/sparc/stage_a.md)) converts the public datasets and trains the 2D model:

```bash
python -m sparc rig --calib stereo_calib.yaml --baseline 0.10   # FOV, network intrinsics, depth precision vs range
python -m sparc profile --stereo                                # params / GFLOPs per component and candidate
python -m sparc bakeoff export --out runs/bakeoff               # then on the Jetson: python3 runs/bakeoff/run_on_device.py
python -m sparc bakeoff report --out runs/bakeoff
python -m sparc convert bdd100k -- --root /data/bdd100k --dst data/sparc/bdd100k   # and culane, openlane, mtsd, zod, ...
python -m sparc train --cfg sparc/cfg/stage_a.yaml --data sparc/cfg/data_stage_a.yaml --out runs/sparc/stage_a
python -m sparc val --weights runs/sparc/stage_a/weights/best.pt --data sparc/cfg/data_stage_a.yaml --out runs/sparc/val
```

## Train on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pirazor/YoloMulti/blob/main/colab/train_adas_mt.ipynb)
[`colab/train_adas_mt.ipynb`](colab/train_adas_mt.ipynb): data (a Supervisely export, a converted dataset, or a
synthetic demo set) → training on a Colab GPU (resumable from Google Drive) → early-results report (metric curves,
best epoch, prediction overlays) → ONNX export with a `.pt` vs ONNX accuracy check. Choose a GPU runtime and
*Run all*; the defaults (`synthetic`, `quick`) are a ~10-minute smoke test that the setup trains. If the repository
is private, add a `GITHUB_TOKEN` Colab secret.

## Quick start

```bash
pip install -r requirements_train.txt            # training machine (CUDA torch first)
python -m adas_mt convert -- --help               # Supervisely -> dataset (docs/data.md)
python -m adas_mt train --data data/dataset.yaml --model yolo26s.pt --distill --teacher dinov3_b
python -m adas_mt val   --weights runs/mt/exp/weights/best.pt --data data/dataset.yaml
python -m adas_mt export --weights runs/mt/exp/weights/best.pt --verify-image frame.jpg   # -> best.onnx + best.json

# on the Jetson (docs/deployment_jetson.md)
python -m adas_mt trt-build --onnx best.onnx --precision fp16
python -m adas_mt eval  --model best.engine --data data/dataset.yaml --device 0   # accuracy of the engine
python -m adas_mt bench --model best.engine --source frame.jpg --gpu-preprocess   # latency, p50-p99, per stage
python -m adas_mt predict --model best.engine --source drive.mp4 --out runs/predict
```
