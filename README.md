# ADAS multi-task perception for Jetson Orin Nano

Detection + drivable area (direct / alternative) + lane (solid / dashed) from one
YOLO26-based CNN, trained with dense foundation-model distillation (training-time
only) and deployed as a TensorRT FP16/INT8 engine with NMS-free detection and
uint8 class-map outputs.

Status: **v0.2 rewrite in progress**: Phase 0 (re-platform), Phase 1 ([data](docs/data.md)) and Phase 2 ([model](docs/architecture.md)) done.
The previous YOLOv13-based implementation lives in [`legacy/yolov13/`](legacy/yolov13/)
(tag `yolov13-mt-baseline`) and is kept as the A0 baseline for ablations.

```
adas_mt/        new package (data, nn, distill, engine, export, deploy)
sign_classifier/ fine-grained traffic-sign classifier (unchanged)
legacy/yolov13/ previous YOLOv13 multi-task implementation (frozen baseline)
```
