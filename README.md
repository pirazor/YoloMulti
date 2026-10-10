# ADAS multi-task perception for Jetson Orin Nano

Detection + drivable area (direct / alternative) + lane (solid / dashed) from one
YOLO26-based CNN, trained with dense foundation-model distillation (training-time
only) and deployed as a TensorRT FP16/INT8 engine with NMS-free detection and
in-graph class-map outputs (int32, or uint8 on TensorRT >= 10).

Status: **v0.2**. Phases 0-5 are implemented: re-platform, [data](docs/data.md), [model](docs/architecture.md), [distillation](docs/distillation.md), [trainer / validator](docs/training.md) and [export / Jetson deployment](docs/deployment_jetson.md). Everything that can be verified on a CPU is tested; the TensorRT engine build, its accuracy and the latency on the Orin are **not yet measured** (commands and a results table are in the deployment doc).
The previous YOLOv13-based implementation lives in [`legacy/yolov13/`](legacy/yolov13/)
(tag `yolov13-mt-baseline`) and is kept as the A0 baseline for ablations.

```
adas_mt/        new package (data, nn, distill, engine, export, deploy)
sign_classifier/ fine-grained traffic-sign classifier (unchanged)
legacy/yolov13/ previous YOLOv13 multi-task implementation (frozen baseline)
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
