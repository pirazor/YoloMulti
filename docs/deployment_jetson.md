# Export and deployment on Jetson Orin Nano (Phase 5)

```
x86 training machine                         Jetson Orin Nano (Super)
────────────────────                         ─────────────────────────────────────────────
best.pt ─► adas_mt export ─► best.onnx ──►   adas_mt trt-build ─► best.engine
            (fused, NMS-free,  best.json      (FP16, or INT8       best.engine.json
             argmax in graph)                  with calibration)        │
                                                                        ▼
                                              adas_mt predict | bench | eval
```

Only the CNN ships. The DINOv3 teacher, the distillation projector and the auxiliary DA classifier are training-time
only and are removed before export. Engines are **not portable**: build on the device, with the TensorRT version you
will run.

## 1. What the exported network is

| | |
|---|---|
| input `images` | float32 `(B, 3, 384, 640)`, **RGB**, `/255`, letterboxed (centred, pad 114) |
| output `det` | float32 `(B, 300, 6)`: `x1 y1 x2 y2 score class` in network pixels. NMS-free (one-to-one head); no NMS, no DFL |
| output `da` | `int32` `(B, 384, 640)`: drivable area per pixel (0 background, 1 direct, 2 alternative) |
| output `ll` | `int32` `(B, 384, 640)`: lane per pixel (0 background, 1 solid, 2 dashed) |

The argmax over the logits is inside the graph (after the bilinear upsample), so the host never copies 3 x H x W
logits. Everything is in `<name>.json` next to the model (class names, `imgsz`, dtypes, preprocessing, which ONNX nodes
belong to which head). `.onnx` files also embed a copy.

Preprocessing is **identical to validation**: `adas_mt.deploy.preprocess.letterbox_bgr` is tested pixel for pixel
against the real validation dataset pipeline for nine frame sizes (720p, 1080p, 4:3, 1:1, portrait, odd sizes, 4K). The
GPU variant (`--gpu-preprocess`, torch bilinear, half-up rounding) matches it to within 2 of 255 levels.

## 2. Export (training machine)

```bash
python -m adas_mt export --weights runs/mt/exp/weights/best.pt --verify-image some_real_frame.jpg
# -> runs/mt/exp/weights/best.onnx + best.json
```

`imgsz` is read from the run's `mt.yaml`. The export runs ONNX Runtime on the result and compares it with PyTorch
(detections tie-robustly, class maps pixel agreement >= 99.9%); a mismatch raises. **Pass a real frame** with
`--verify-image` for a trained model: the default input is a seeded smooth random image, which exercises the heads but
may give no confident detections (the export warns when that happens).

| flag | when |
|---|---|
| `--seg-dtype int32` (default) | every TensorRT version |
| `--seg-dtype uint8` | TensorRT >= 10 (JetPack 6.2 / 7.2): 4x smaller output copy. The engine build refuses it on 8.6 |
| `--seg-dtype logits` | debugging / calibration studies (float32 `(B, C, H, W)`) |
| `--dynamic` | free batch axis (multi-camera). A fixed batch of 1 is the fastest and what you want for one camera |
| `--decompose-pixel-shuffle` | only if the engine build rejects `DepthToSpace` (see Troubleshooting) |
| `--no-simplify` | skip onnxslim |

The export reports an **op audit**: every op in the graph is checked against the set the TensorRT ONNX parser handles in
8.6 and 10.x. The one op it flags for confirmation on the device is `DepthToSpace(mode=CRD)` (the lane head's
PixelShuffle).

Copy `best.onnx` and `best.json` to the Jetson.

## 3. Jetson setup

* JetPack 6.2 (TensorRT 10.3) or 7.2 (TensorRT 10.16). TensorRT 8.6 (JetPack 6.0/6.1) also works with `int32` outputs.
* Create the venv with `--system-site-packages` so the system `tensorrt` and OpenCV are visible. Install NVIDIA's torch wheel
  for your JetPack (the runner uses torch only for CUDA buffers and streams). `requirements_jetson.txt` lists the rest.
* Power mode matters by a large factor. `sudo nvpmodel -q` shows the active mode; select the MAXN SUPER mode (look up its
  id in `/etc/nvpmodel.conf`) and run `sudo jetson_clocks` before benchmarking. `adas_mt bench` records the active
  `nvpmodel` mode in its report.

## 4. Build the engine (on the Jetson)

```bash
# FP16: the safe default
python -m adas_mt trt-build --onnx best.onnx --precision fp16

# INT8: calibrate with training images that cover your conditions (night, rain, glare, tunnels)
python -m adas_mt trt-build --onnx best.onnx --precision int8 --calib data/dataset.yaml --calib-n 512
```

* The build takes minutes. `--timing-cache t.cache` makes rebuilds fast.
* INT8 calibration uses the **deployment preprocessing** on **training** images (`--calib` takes a data.yaml and uses only
  its train split). It samples evenly across the sorted list (consecutive video frames are nearly identical), then shuffles.
* By default INT8 keeps the three heads (detection, DA, lane) in **FP16** (`--keep-fp16 heads`): they are a small part
  of the FLOPs, and the lane head (the bias-initialised sub-pixel classifier) and the box regression are the most
  quantisation-sensitive layers. `--keep-fp16 seg` keeps only the segmentation heads, `none` quantises everything, or pass
  ONNX node-name substrings. Only floating-point layers are pinned; integer layers (TopK indices, Shape, Gather) are left alone.
* A calibration cache is only reused when you pass `--calib-cache` (scales are keyed by tensor name, so a stale cache from
  an older model with the same layer names would silently produce a wrong engine). Delete it when the model changes.
* The command prints the equivalent `trtexec` line, handy for timing the bare engine
  (`/usr/src/tensorrt/bin/trtexec` on JetPack).

Outputs: `best.engine` and `best.engine.json` (the model metadata plus build info: precision, TensorRT version, host).

## 5. Accuracy: what export and quantisation cost

`eval` runs the **training validator** (same dataset pipeline, mAP, DA mIoU, lane IoU, fitness) on an exported model, so
the numbers are directly comparable:

```bash
python -m adas_mt val  --weights runs/mt/exp/weights/best.pt   --data data/dataset.yaml
python -m adas_mt eval --model   best.onnx                      --data data/dataset.yaml
python -m adas_mt eval --model   best.engine                    --data data/dataset.yaml --device 0   # FP16 engine
python -m adas_mt eval --model   best_int8.engine               --data data/dataset.yaml --device 0
```

`eval` reuses the training validator, so it needs the `ultralytics` package (the Jetson venv can install `ultralytics==8.4.171`
next to NVIDIA's torch wheel); `predict`, `bench` and `trt-build` do not.

On a trained toy model the `.pt` and the ONNX give identical metrics to four decimals (tested), so any difference you see on
the device is the engine's precision. Suggested acceptance (adapt to your tolerance): FP16 within ~0.5 points of the `.pt` on
mAP50-95, DA mIoU and lane IoU; INT8 within ~1-2 points. If the lane IoU is what drops, keep more in FP16 and add more
diverse calibration images before blaming the model.

Fill this table on your hardware:

| model | mAP50-95 | DA mIoU | lane IoU(fg) | lane recall | GPU compute (trtexec) | end-to-end p50 / p99 |
|---|---|---|---|---|---|---|
| `.pt` (fp32, PyTorch) | | | | | n/a | n/a |
| FP16 engine | | | | | | |
| INT8 engine | | | | | | |

## 6. Latency

```bash
python -m adas_mt bench --model best.engine --source sample_720p.jpg --n 300 --warmup 50 --gpu-preprocess --json bench.json
```

The report splits **pre** (letterbox), **infer** (host to device, execute, device to host, including the in-graph argmax)
and **post** (score filter, boxes and masks back to the original frame), with mean / p50 / p90 / p99 / max and FPS. Use a
real frame: the network cost does not depend on content, but postprocessing scales with the detections.

Notes on what to expect and where time goes:

* The published Orin Nano Super TensorRT numbers for YOLO26 at 640x640 are 4.6 / 7.2 ms (n / s, FP16). This model adds two
  light heads (about 0.2 / 0.7 GFLOPs at 384x640 for the DA / lane heads) and runs at 384x640 (0.6x the pixels of 640x640), so
  the engine should be in that neighbourhood or faster. **These are estimates, not measurements**: measure with `bench` and `trtexec`.
* The GPU compute time of the engine (`trtexec`) and the `infer` stage differ by the host-device copies and the Python
  call overhead; `pre` and `post` run on the CPU. On the Orin Nano's six ARM cores the CPU stages are not negligible at
  720p and above: use `--gpu-preprocess`, or `masks_at="network"` in the library API (skip resizing the masks to the
  frame) when your consumer can work at 384x640.
* Memory is shared between CPU and GPU; the default `--workspace-mb 1024` is a limit for the build, not a resident cost.

## 7. Using the runner as a library

```python
from adas_mt.deploy.runner import Pipeline, overlay

pipe = Pipeline("best.engine", conf=0.25, gpu_preprocess=True)   # or best.onnx (ONNX Runtime)
res = pipe(frame_bgr)       # numpy BGR uint8, any resolution
res.boxes, res.scores, res.classes   # original-frame pixels
res.da, res.ll                       # uint8 class maps at the frame's resolution
res.timings                          # ms: pre / infer / post / total
vis = overlay(frame_bgr, res, pipe.names)
```

`python -m adas_mt predict --model best.engine --source <image | dir | video | camera index | gstreamer pipeline>`
writes overlays (and `--save-masks`). A GStreamer string (anything containing `!`, e.g. a CSI camera through
`nvarguscamerasrc`) is passed to OpenCV with the GStreamer backend.

`adas_mt.deploy.runner` and `adas_mt.deploy.trt_build` do not import ultralytics (or torch at import time), so the Jetson does not need
the training stack.

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| `trt-build` fails parsing at `DepthToSpace` | re-export with `--decompose-pixel-shuffle` (Reshape/Transpose/Reshape; identical output, tested) |
| "uint8 segmentation outputs need TensorRT >= 10" | export with `--seg-dtype int32` (TensorRT 8.6) |
| "could not deserialize ... engines only load on the TensorRT version and GPU they were built on" | rebuild the engine on this device/version |
| INT8 lane IoU / recall drops | `--keep-fp16 heads` (default), more and more varied `--calib` images, check with `eval`; compare FP16 first |
| ONNX parity failure at export | a real bug or a non-deterministic op: run with `--seg-dtype logits` to see where the maps diverge |
| first frame is slow | CUDA / cuDNN lazy init; `bench` warms up, `predict` excludes the first frame from its statistics |

## 9. What is verified, and what is not

Verified in CI on CPU (no Jetson here): the preprocessing against the validation pipeline; ONNX export contract, op audit and PyTorch/ONNX Runtime
parity (static, dynamic batch, int32/uint8/logits, decomposed PixelShuffle), including a trained model whose `.pt` and ONNX metrics
are identical; the evaluator; the runner's geometry, overlay, sources, predict and bench; the **TensorRT builder and backend control flow
against a fake `tensorrt`** that enforces the 8.6-vs-10 API differences and executes the real ONNX underneath.

**Not verified until you run it on the Orin:** that the TensorRT parser accepts `DepthToSpace` (fallback provided), real FP16 / INT8
accuracy, INT8 calibration quality, latency / FPS, power and thermal behaviour, and GStreamer camera input.
