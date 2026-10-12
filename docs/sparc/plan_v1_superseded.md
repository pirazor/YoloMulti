<!-- Status: design approved 2026-10-11; Phase 0 implemented (see docs/sparc/phase0.md), on-device bake-off pending. Supersedes the YOLO26-based plan. Section 0 records what Phase 0 changed. -->

# SPARC: one sparse-query perception network for the Jetson Orin Nano Super

## Context

The merged `adas_mt` model is a YOLO26 CNN with two segmentation heads. The user rejected it: it is an
adaptation of an existing detector, it cannot grow into the required system, and YOLO26 is no longer the
accuracy/latency frontier. The required system (decided in Q&A on 2026-10-11) is a **single network** for a
front stereo camera (depth from stereo) + IMU + GPS on an **Orin Nano Super at ≤ 33 ms (FP16 TensorRT, 384×640)** that outputs:

1. road users: detection + tracking + **vehicle 3D state** (position/depth, yaw, size);
2. traffic-light colour and **fine traffic-sign class** (Mapillary TSD, ~400 classes) in the same pass;
3. drivable area (direct/alternative) and **lane instances with attributes**: order relative to ego,
   travel direction (same/opposing), continuity, colour/type;
4. a **learned BEV map head**: drivable area, lane lines with heading, agent footprints on an ego-centred grid.

Datasets: BDD100K, CULane, Mapillary Traffic Sign Dataset (MTSD), nuScenes front camera (3D boxes, LiDAR depth,
HD-map BEV labels), and the user's own drives (no recordings yet; a recording protocol is a deliverable).
Training: one A100/H100-class GPU for days. Target branch `multitask`; the old plan below this line is superseded.

## 0. Phase-0 update: the sensor is a stereo pair, so depth costs compute

The "depth camera" is a stereo pair (the example rig: global-shutter eyes, 1440x900 crops, 99.8 mm baseline). Four
consequences, measured with `python -m sparc rig` on the example calibration (`docs/sparc/phase0.md`):

1. **Depth is not a free input.** A separate foundation stereo engine costs ~55 ms per frame on an Orin NX at
   320x320, more than the whole budget. Runtime depth options, decided by the Phase-0 device run and the Phase-3
   accuracy ablation: **D1** the in-network stereo head (`sparc/nn/stereo.py`: the right eye costs a 4-conv stem, a
   48-level group-wise correlation volume and convex upsampling, ~5.6 GFLOPs), **D2** an external stereo engine on a
   second stream, **D3** no runtime stereo (IPM + learned depth cues only). The engine contract supports both a
   `depth` input (D2, or LiDAR during training) and `image_right` + `baseline` inputs (D1).
2. **Depth precision is range-limited**: error < 10 % (0.25 px disparity error) to ~53 m at full eye width, ~24 m at
   the 640-px network width, ~12 m at 320 px. Training depth labels therefore come from a large stereo model run
   **offline at full resolution** on recorded raw pairs (a second, legitimate teacher use case), and the BEV beyond
   ~25 m is supervised by **future-near accumulation**: a region 40 m ahead is labelled from the later frames in
   which it is 5-10 m ahead, moved back with the recorded ego-motion (`recording/protocol.md` section 7).
3. **The perception input is the rectified left eye**: depth lives in the rectified frame; on the example rig the
   rectified and raw left images differ by a 1.5x focal length, so mixing them misplaces every depth sample.
   Rectification with `alpha = 0` cuts the horizontal FOV from 78 to 57 deg; `alpha ~0.5` keeps ~83 deg. Raw
   pairs are recorded, so the choice stays open.
4. **Splat geometry**: taking the nearest depth of a stride-8 block biased ground points towards the camera by
   ~1 m at 10 m; the splat now uses stride-4 points with the mean valid depth of each 4x4 block (unit-tested: no
   range bias on a flat ground).

Measured shared-part cost at 384x640 after slimming to real-time-DETR widths: encoder 8.0, decoder 4.4, drivable
area 1.2, BEV 6.1, stereo head 5.6 GFLOPs; backbones 7.5 (YOLO26s reference) to 44 GFLOPs (ViT-S, ConvNeXt-T).
The rig code is generic (any OpenCV stereo calibration); the recording specification is written for any recorder.

## What the research says (Oct 2026, sources in the chat log)

- **Real-time detection moved from YOLO to DETR-style decoders.** On COCO at T4-FP16 latency: RF-DETR-N 48.4 AP @ 2.3 ms
  (384²), RF-DETR-S 53.0 @ 3.5 ms, RF-DETR-M 54.7 @ 4.4 ms (ICLR 2026, DINOv2 backbone, Apache-2.0 up to L);
  DEIMv2-S 50.9 AP with 9.7 M params / 26 GFLOPs (DINOv3-distilled ViT-Tiny); RT-DETRv4-S 49.7 AP (DINOv3 teacher via a
  Deep Semantic Injector, training-time only). YOLO26s: 48.6 AP @ 2.5 ms. At the small end the DETR family is **+2–4.5 AP
  at equal latency**, and its decoder is what makes one network for many output types possible.
- **Foundation features are the accuracy lever, at zero inference cost.** DEIMv2 and RT-DETRv4 both get their gains from
  DINOv3 (as backbone initialisation or as a frozen teacher). DINOv3 also ships ConvNeXt-T/S/B/L backbones distilled from
  its 7 B ViT (TensorRT-clean CNNs). This is the concrete "teacher use case": +AP for free on the device.
- **Orin Nano Super feasibility.** RF-DETR-Base measures 14.8 ms FP16 / 11.8 ms INT8 (TensorRT) on this device; RF-DETR-Seg
  ~15 ms; DEIMv2-Atto 89 FPS. A ViT-S-class backbone fits the budget; the earlier "ViT is unsafe on Jetson" worry was
  overcautious. All numbers must still be re-measured on the user's device (Phase 0).
- **Per-query structured outputs work in these decoders**: RF-DETR Keypoint (June 2026) predicts keypoints + uncertainty
  per detected object inside the detection transformer; MOTIP (CVPR 2025) learns tracking as ID prediction end to end;
  MonoDETR/MonoDINO-DETR predict 3D state per query from one camera; Sparse LaneFormer / O2SFormer predict lanes as
  queries; SparseDrive / Sparse4D show sparse queries beat dense BEV for latency. Nobody combines these for a single
  front camera on an edge device with depth-grounded BEV supervision: that is the gap this design fills.
- **Labels that do not exist publicly** and must be derived or self-generated: lane travel direction / ego ordering
  (BDD100K only has parallel/perpendicular, continuity, type/colour; CULane has geometry only), 3D boxes on BDD100K
  (use nuScenes front camera; licence CC BY-NC-SA), BEV maps (nuScenes HD-map layers for pretraining; own drives for
  the user's rig).

## The design: SPARC (Sparse Perception with Attributed queries and Range-grounded BEV Context)

```
RGB 384×640 ──► foundation backbone (DINOv2/3, Phase-0 bake-off) ──► hybrid encoder ──► P3 P4 P5 (+P2)
                                                                                         │
            ┌────────────────────────────────────────────────────────────────────────────┤
            │  UNIFIED SPARSE DECODER (3–4 deformable-attention layers, d=256)           │
            │  typed queries share the cross-attention over the image features:         │
            │   • agent queries (≈200): 2D box, class(10), 3D state (depth, yaw, dims),   │
            │       ID embedding (tracking), attributes: light colour, sign class (MTSD)  │
            │   • lane queries (≈24): row-anchored polyline, existence, order vs ego,     │
            │       travel direction, continuity, colour/type                             │
            └──────────────┬───────────────────────────────┬─────────────────────────────┘
                           │                               │
  dense heads: drivable area (stride 8), lane aux mask     │  3D states ──► rendered agent footprints
                           │                               ▼
  stereo depth (in-network head or input) + K + cam2ego ──► DEPTH SPLAT (image features → BEV cells;
                                 learned ground-plane branch where depth is missing) ──► BEV decoder
                                 ──► BEV drivable area, BEV lane lines + heading, agent footprints
```

Key decisions:
- **Backbone** chosen by measurement on the Orin (Phase 0): (a) RF-DETR-S DINOv2-S windowed ViT + its encoder
  (COCO-pretrained, Apache-2.0), (b) DINOv3 ConvNeXt-T distilled, (c) DEIMv2-style distilled ViT-Tiny. Default
  expectation: (a) for accuracy, (b) as the TensorRT-safe fallback.
- **One decoder, typed queries**: the same deformable cross-attention serves agents and lanes; only the query sets,
  heads and matching differ. Tracking v1 = ID embeddings + 3D positions matched in post-processing (MOTIP-style ID
  loss on BDD100K MOT clips); v2 = one-frame track-query memory.
- **Depth-grounded splat BEV**: no learned per-pixel depth distribution (LSS) — metric stereo depth (section 0) places
  features with a per-frame BEV index (ScatterElements-add, TensorRT-safe if the Phase-0 probe passes); a learned
  IPM branch covers cells without depth. On monocular sources (nuScenes) LiDAR depth is fed through the `depth`
  input during training.
- **Supervision without manual BEV/3D labels on the user's rig**: nuScenes pretrains the 3D and BEV heads; the
  user's drives provide depth-derived positions, tracking-derived headings and **self-generated BEV labels**
  (perspective labels lifted by measured depth, accumulated over time with GPS/IMU ego-motion, rasterised).
- **Multi-dataset training with partial-label masking** (each loss only where the source labels it), staged
  curriculum, DINOv3 teacher distillation on the encoder features (training-time only).
- **Derived lane labels**: travel direction from marking colour/type rules (yellow = opposing-traffic boundary in
  BDD/US) + ordering from lane position vs the ego centre + ego trajectory on own drives; a small manually verified
  subset to measure the derived labels' precision before training on them.

## Evaluation: what "superior" will mean, measured

| output | benchmark | baseline to beat |
|---|---|---|
| detection | BDD100K det mAP | RF-DETR-S and YOLO26s fine-tuned on BDD100K |
| tracking | BDD100K MOT mMOTA / mIDF1 | ByteTrack (val 45.5 / 54.8) |
| lanes | CULane F1@50 | CLRNet 80.5 (LaneLM 82.7 as the ceiling reference) |
| drivable area | BDD100K DA mIoU | current adas_mt model |
| 3D | nuScenes front-cam mono-3D AP | MonoDETR-class baseline |
| BEV map | BEV IoU (drivable, lane divider) on nuScenes front cam | LSS single camera |
| signs / lights | MTSD top-1; BDD light colour accuracy | current sign_classifier |
| latency | Orin Nano Super FP16/INT8 per output | ≤ 33 ms total |

## Honest framing of the novelty

Each component is grounded in 2025–26 state of the art (DETR-style real-time decoders, per-query structured outputs,
ID-prediction tracking, mono-3D queries, lane queries, sparse-first perception, DINOv3 distillation). What does not
exist as a published system is their combination for **one front camera + depth + IMU/GPS on an edge device**, with
a **sensor-grounded BEV** (measured depth instead of a learned depth distribution) that is **supervised without manual
BEV/3D labels on the user's rig**. "Superior" is demonstrated only by the benchmark table above, run by this plan's
ablation protocol; nothing is promised before it is measured. Budgets (latency, GPU-days) are estimates until Phase 0.

## 1. Repository layout

New package `sparc/` next to `adas_mt/` (add `sparc*` to `pyproject.toml` `packages.find`). `adas_mt/` stays as the
YOLO26s baseline (A0) and as a library: every core module there subclasses Ultralytics (`DetectionModel`,
`DetectionTrainer`, `YOLODataset`) and `docs/training_requirements.md` lists 55 workarounds for that coupling; a
multi-input DETR-style graph with Hungarian matching, clip sampling and multi-dataset mixing does not fit it.

| reused verbatim | reused with small backward-compatible edits | replaced (new in `sparc/`) |
|---|---|---|
| `adas_mt/deploy/preprocess.py` (letterbox, `LetterboxInfo`, `letterbox_torch`) | `adas_mt/deploy/runner.py`: `TrtBackend`/`OrtBackend` take `input_names` + per-input buffers (today one `(B,3,H,W)` input); `iter_frames`, `bench`, `predict` loop unchanged | `nn/model.py`, `nn/heads.py`, `nn/loss.py` → `sparc/nn/*` |
| `adas_mt/deploy/meta.py` | `adas_mt/deploy/trt_build.py`: multi-input profiles/calibrator; precision map from meta | `engine/{train,val,config}.py` → `sparc/engine/*` (config dataclass pattern kept) |
| `adas_mt/engine/metrics.py` (`SegConfusion` for DA, lane-aux, BEV layers) | `adas_mt/export.py`: `TRT_OPS`, `audit_ops`, `_embed_meta`, `_pin_io_shapes`, parity pattern → `sparc/deploy/export.py` adds `GridSample`, `ScatterElements` | `data/dataset.py`, `data/transforms.py` → `sparc/data/*` |
| `adas_mt/data/masks.py` (DA + lane-aux packed in one uint8) | `tests/test_adas_mt/fake_trt.py`: multi-input context | `deploy/evaluate.py` → `sparc/engine/evaluators/` |
| `adas_mt/distill/teacher.py`, `distill/loss.py` (`FrozenTeacher`, `distill_terms`, `KDProjector`) | `adas_mt/distill/pretrain.py` is the template for the lean trainer loop | `convert_supervisely.py` stays (own labels); `sign_classifier/` stays as the sign baseline |
| `colab/train_adas_mt.ipynb` cell pattern → `colab/train_sparc.ipynb` | | |

```
sparc/
  cli.py __main__.py            convert | bakeoff | train | val | export | trt-build | predict | bench | eval
  cfg/   model_{n,s}.yaml  stage{1,2,3,4}.yaml
  data/  schema.py  converters/{bdd100k,culane,mtsd,nuscenes_front,own_drive}.py  lane_rules.py
         dataset.py  augment.py  targets.py  synthetic.py
  nn/    backbone.py encoder.py deform_attn.py decoder.py heads.py bev.py model.py matcher.py loss.py
  distill/hook.py   engine/{config,trainer,ema,balance}.py   engine/evaluators/{det_coco,mot,culane,seg,nusc3d,bev,sign}.py
  track/{associate,bev_accum}.py   deploy/{export,runner,postprocess}.py   utils/{profile,geometry}.py
tests/test_sparc/   docs/sparc/   recording/{protocol.md,record.py,calibrate.md}
```

## 2. Data layer

**Unified sample** (`sparc/data/schema.py`; one `.npz` + JSON per frame, written by every converter):
```python
@dataclass
class Sample:
    image: str; depth: str | None          # depth: uint16 PNG in mm, 0 = invalid
    K: np.ndarray                          # 3x3 at native resolution
    cam2ego: np.ndarray                    # 4x4, pitch/roll-corrected (identity + camera height for 2D-only sets)
    boxes2d, cls, tl_color, sign_cls       # (N,4) xyxy, (N,), (N,) -1 = unlabelled
    box3d: np.ndarray                      # (N,7) depth, yaw, l, w, h, x, y   | track_id (N,), clip_id, frame_idx
    lanes: list[LaneInstance]              # polyline (M,2) px, order, direction, continuity, ltype, weak: bool
    da_mask, lane_mask: str | None         # uint8 PNG class maps
    bev: str | None                        # uint8 (3,240,160): drivable class, lane line, heading bin (+ agent occupancy)
    label_mask: LabelMask                  # bools: det, cls_vocab(10,), tl, sign, box3d, depth, da, lane_geom,
                                           # lane_{order,dir,cont,type}, bev_{da,lane,agent}, track
    licence: str                           # "bsd" | "research" | "own"
```
`cls_vocab` is the Null-BCE mask: BDD = all 10 classes; MTSD = {traffic sign}; nuScenes = {car, truck, bus,
motorcycle, bicycle, pedestrian}; CULane = none.

**Converters** (`python -m sparc convert <source> --src --dst`):
- `bdd100k.py`: det JSON (10 classes, `trafficLightColor`), MOT (`box_track`, keeps `clip_id`), drivable polygons →
  `da_mask` (1 direct / 2 alternative), lane polylines with `laneStyle/laneType/laneDirection` → `LaneInstance` +
  rasterised `lane_mask` (8 px at 720p, as the existing converter).
- `culane.py`: the 4 row-anchored lanes have fixed semantics → **order labels for free** (ego-left-2, ego-left,
  ego-right, ego-right-2); existence list.
- `mtsd.py`: boxes + ~400 classes; `cls_vocab = {sign}`; also writes crops for `sign_classifier/` (baseline).
- `nuscenes_front.py`: CAM_FRONT keyframes; 3D boxes → camera frame (depth, yaw, dims, ego xy); LiDAR → projected
  sparse depth + IP-Basic completion → `depth`; `cam2ego` from calibration + ego pose; map layers (`drivable_area`,
  `lane_divider`, `road_divider`) rasterised in the ego frame (x forward 0–48 m, y −16..16 m, 0.2 m → 240×160);
  divider heading from the polyline tangent (8 bins + sin/cos); agent footprints from the boxes.
- `own_drive.py`: reads the recording format (section 5), interpolates IMU/GPS to frame timestamps, writes `K`,
  `cam2ego`, `depth`; labels merge from `convert_supervisely.py` output by stem and from the pseudo-label tools.

**Derived lane labels** (`lane_rules.py`, all `weak=True`, loss weight 0.3): reference row y = 0.9 H; ego lane = the
two markings bracketing x = W/2; order = signed index from the ego lane (clipped to ±3); direction: markings on the
far side of a single/double **yellow** line = `opposing` (US rule, BDD), curb = `boundary`, else `same`; continuity
from `laneStyle`, type from `laneType`. Own drives: the GPS/IMU ego trajectory picks the ego lane; tracked agents
with yaw ≈ 180° on a lane mark it `opposing`. A **500-frame manually verified set** (`weak=False`) is the only set
on which direction/order accuracy is reported.

**Mixing** (`dataset.py`): `MixedDataset(sources, weights)` with a DDP-aware `WeightedSourceSampler`;
`ClipPairDataset` yields (t, t+k), k ∈ 1..3, for tracking. Collate pads variable-N targets into lists; dense targets
and `label_mask` are stacked.

**Augmentation** (`augment.py`; every op updates `K`, boxes, polylines, masks, BEV consistently): scale 0.6–1.4 +
translate (`K` scaled/shifted, depth resampled nearest, BEV untouched: the ego frame does not change); horizontal
flip p = 0.5 (3D x → −x, yaw → −yaw, BEV mirrored in y, lane order sign flipped, heading mirrored); photometric HSV
with hue ≤ 0.015 (traffic-light colours); depth noise/holes/range clipping. **No rotation** (breaks row anchors and
the corrected extrinsics) and **no mosaic for 3D/BEV samples** (a mosaic tile has its own principal point and ego
pose; one `K`/`cam2ego`/BEV grid cannot describe the composite). v1 trains without mosaic (DETR-style detectors do);
a 2D-only mosaic for BDD-det/MTSD is a Phase 6 ablation.

## 3. Model layer (`sparc/nn/`, shapes at 384×640, batch 1; all estimates until Phase 0)

| module | in → out | params | GFLOPs | Orin FP16 ms |
|---|---|---|---|---|
| `backbone.py` `BackboneFactory("rfdetr_s" \| "dinov3_convnext_t" \| "deimv2_vit_t")` | image → P2 (64,96,160), P3 (·,48,80), P4 (·,24,40), P5 (·,12,20) | 6–28 M | 12–25 | 7–13 |
| `encoder.py` HybridEncoder: AIFI on P5 (240 tokens) + CCFM PAN → 256 ch | P3/P4/P5 at 256 ch | ~4 M | ~4 | ~3 |
| `decoder.py` TypedQueryDecoder, 3 layers, d = 256, 8 heads × 3 levels × 4 points | 224 queries over 5040 tokens | ~6 M | ~0.5 | ~3 |
| `heads.py` AgentHead | box2d(4), cls(10), box3d(depth, sin, cos, l, w, h), tl(4), sign(401), id_emb(64) | 0.5 M | <0.1 | <1 |
| `heads.py` LaneHead | x per row (72), visibility (72), exist(1), order(7), dir(3), cont(2), type(6) | 0.2 M | <0.1 | |
| `heads.py` DAHead (stride 8, as `adas_mt`) + LaneMaskHead (stride 4, training-only aux) | (3,384,640), argmax in graph | 0.2 M | ~0.8 | ~1 |
| `bev.py` DepthSplat + IPMSampler + BEVDecoder + BEVHead | (7,240,160) | ~2 M | ~2.5 | 3–4 |
| **total** | | 19–41 M | 24–33 | **18–26 GPU + 3–5 CPU** |

**Typed queries.** One self-attention over all 224 queries with a learnt type embedding (agents can attend to lanes);
shared deformable cross-attention; per-type reference points: agent anchors from the top-k encoder tokens refined per
layer (RT-DETR style); lane queries keep one reference point (polyline centroid) with larger learnt offsets. Lane
output is CLRNet-like: 72 row anchors from y = 0.35 H to H, normalised x per row, per-row visibility.

**Deformable attention** (`deform_attn.py`): `grid_sample`-based `MSDeformAttn` (RT-DETR's PyTorch path) → ONNX
`GridSample` (opset 16), which TensorRT ≥ 8.5 has natively (RF-DETR/RT-DETR engines are built this way). Phase 0
confirms it on JetPack 6.2 / TensorRT 10.3; fallback: 4-corner `GatherND` bilinear.

**Known-depth splat** (`bev.py::DepthSplat`, in-graph, static shapes): (1) P3 → 1×1 conv → (64,48,80); depth
min-pooled ×8 (0 → +inf before pooling) → d (3840,); (2) back-project `[X,Y,Z] = K⁻¹[u·d, v·d, d]`, `ego = cam2ego·[X,Y,Z,1]`
(constant pixel grid, `MatMul`); (3) cell index `floor((48−x)/0.2)·160 + floor((y+16)/0.2)`; invalid (d = 0, outside
the grid, height outside [−1, 3] m) → a dump cell; (4) `ScatterElements(reduction="add")` of features and of a ones
channel into (65, 38401), divide by count (mean splat), drop the dump cell, reshape (64,240,160). Ops: MatMul, Mul,
Div, Floor, Clip, Cast, Where, ScatterElements. **Uncertain:** ScatterElements-add in TensorRT 10.3's parser — Phase 0
probes it. Fallbacks in order: one-hot `Equal`+`MatMul` splat at a 0.4 m grid (9600 cells, ~1 GFLOP); a ~40-line
`bev_pool` plugin; two engines with `index_add_` in between (breaks the single-engine goal, documented).

**IPM branch** (`IPMSampler`): constant cell-centre matrix (38400,4) at z = 0 → `K[R|t]⁻¹` from `cam2ego` → (u, v) →
`grid_sample` on the reduced P3 → (64,240,160); cells behind the camera / above the horizon masked.

**BEVDecoder**: concat [splat 64, IPM 64, valid 1] → 3×3 stride 2 → 4 residual blocks @96 ch on 120×80 → upsample →
head (7,240,160): drivable (3 logits), lane line (1), heading sin/cos (2), agent occupancy (1, auxiliary; at inference
footprints are rasterised on the CPU from the tracked 3D states).

**Budget check** (Phase 0): `sparc/utils/profile.py` (FlopCounterMode per submodule, as `adas_mt/utils/profile.py`),
then `trtexec --dumpProfile --useCudaGraph` on the Orin per stage (`export --until backbone|encoder|decoder|bev`).
Knobs if over 33 ms: N backbone variant, 320×576 input, 2 decoder layers, BEV at 0.4 m, INT8 backbone.

## 4. Training layer

**Why not the Ultralytics trainer.** It owns the model type (`DetectionModel`, TAL assigner), the batch dict
(`img/cls/bboxes/batch_idx`), one `YOLODataset`, `get_cfg` validation, and `DetMetrics` over a `(B,300,6)` tensor.
SPARC needs multi-input forward, Hungarian matching, list-valued targets, clip pairs, weighted mixing and seven
evaluators.

**Lean trainer** (`sparc/engine/trainer.py`, ~400 lines, grown from `adas_mt/distill/pretrain.py`): AdamW (backbone
LR ×0.1), linear warm-up + cosine, `torch.autocast` bf16/fp16 + `GradScaler`, grad-clip 0.1 (DETR practice), EMA with
the Ultralytics decay ramp (`ema.py`), `torchrun` DDP through the mixed sampler, `last.pt` = {model, ema, optimizer,
scaler, scheduler, epoch, rng, sampler state}, `results.csv` + `run.yaml` (the `mt.yaml` record; `SparcConfig` reuses
`MTConfig`'s strict `from_dict`/`find_run_cfg`/`save`), per-source metric rows every epoch, `colab/train_sparc.ipynb`
with the existing cell sequence.

**Criterion** (`nn/loss.py`): `HungarianMatcher` per query type (agents: focal cls + L1 + GIoU; lanes: row-x L1 +
existence); every loss gated by `label_mask` and averaged over the **samples carrying that label** (not the batch),
so rare tasks are not diluted. Terms: 2D focal/Null-BCE over `cls_vocab` + L1 + GIoU | 3D: L1 log-depth, sin/cos yaw,
L1 dims | lanes: L1 x, BCE visibility/existence, CE attributes (weak ×0.3) | DA: `masked_ce + masked_dice` (imported
from `adas_mt.nn.loss`) | lane-aux: `masked_ce + masked_focal_tversky` | BEV: CE + Dice per layer, L1 heading on lane
cells | ID: MOTIP-style ID classification over per-clip ID slots + InfoNCE across the frame pair | auxiliary losses on
every decoder layer.

**Loss weighting: fixed gains, calibrated once.** Uncertainty weighting drifts for tasks absent from most batches and
GradNorm doubles the backward cost; `engine/balance.py` logs the gradient L2 norm reaching the encoder per task over
200 steps and the gains are set so every task lands within 0.3–3× of detection, recorded in `run.yaml` (the
`adas_mt` measurement in `docs/training.md` is the precedent). Uncertainty weighting is ablation B3.

**Curriculum** (`cfg/stage*.yaml`, each stage resumes the previous `best.pt`):

| stage | data (sampler weights) | new losses | A100 GPU-days |
|---|---|---|---|
| 1 | BDD det+DA+lanes (0.5), CULane (0.25), MTSD (0.25) | 2D, attributes, lanes, DA, KD | ~1.5 / run |
| 2 | + nuScenes front (0.4), Stage-1 replay (0.6) | 3D, BEV (splat + IPM) | ~1 |
| 3 | BDD MOT clip pairs (0.5), replay (0.5) | ID | ~1 |
| 4 | own drives (0.5: labelled + pseudo + self-BEV), replay (0.5) | all, backbone LR ×0.1 | ~0.5 |

**Distillation hook** (`distill/hook.py`): `FrozenTeacher("dinov3_b")` on the 24×40 patch grid, `KDProjector` on
encoder P4, `distill_terms`, cosine-decayed weight; on in Stage 1 (also for DINO-initialised backbones: it
regularises the encoder), off in Stage 4. This is the teacher's role in one sentence: +1.5–3 AP at equal latency,
as measured by RT-DETRv4/DEIMv2, because the student's features are pulled towards foundation features it could
never learn from 100k driving images alone.

**Ablation protocol** (`docs/sparc/results.md`, one table): A0 `adas_mt` YOLO26s; RF-DETR-S fine-tuned on BDD; A1
Stage 1 without KD; A2 Stage 1 + KD; A3 + Stage 2 (does 3D/BEV cost 2D mAP?); A4 + Stage 3 vs ByteTrack; A5 + Stage
4; B1 BEV splat-only / IPM-only / both vs LSS front-cam; B2 depth input: LiDAR-completed / mono pseudo-depth / none;
B3 fixed vs uncertainty weighting; B4 backbone (Phase 0). Metrics: BDD det mAP (pycocotools), BDD MOT mMOTA/mIDF1
(`bdd100k` toolkit), CULane F1@50 (own numpy port of the 30-px IoU F1, validated against CLRNet's released
predictions), BDD DA mIoU, nuScenes front-cam AP@{0.5,1,2,4} m (devkit with a CAM_FRONT visibility filter; MonoDETR
re-evaluated on the same subset), BEV IoU drivable/divider, MTSD top-1 given GT boxes vs `sign_classifier/`, Orin ms
per output.

## 5. Deployment layer

**Engine contract** (`sparc/deploy/export.py`; meta JSON as today). Inputs: `image` f32 (1,3,384,640) RGB/255
letterboxed rectified left eye; `K` f32 (1,3,3) at network pixels (letterbox-adjusted); `cam2ego` f32 (1,4,4); then
either `depth` f32 (1,1,384,640) in m, letterboxed nearest, pad 0 (D2, or any external depth), or `image_right` f32
(1,3,384,640) + `baseline` f32 (1,1) in m (D1, in-network stereo, which also outputs `depth`). Outputs: `agents` f32 (1,200,18) = xyxy, score, cls, depth, sin, cos,
l, w, h, x_ego, y_ego, tl_id, tl_p, sign_id, sign_p; `agent_emb` f16 (1,200,64); `lane_xs`/`lane_vis` f32 (1,24,72);
`lane_meta` f32 (1,24,9) = exist + (id, p) for order/dir/cont/type; `da` uint8 (1,384,640); `bev` f16 (1,7,240,160).

**Post-processing** (CPU, `sparc/track/`): `associate.py`: ego-motion from IMU/GPS moves the previous 3D positions
into the current ego frame; cost = (1 − cos(emb)) + λ·‖Δxyz‖/σ_cls; Hungarian (scipy); ByteTrack-style two-tier
confirmation; 2D fallback when depth is missing. `bev_accum.py`: log-odds BEV accumulated over time, warped by
ego-motion (`cv2.warpAffine` on 240×160), decay 0.9; agent footprints rasterised from the tracked 3D states.
`SparcPipeline` (`deploy/runner.py`) wraps the `adas_mt.deploy.runner` backends and timing.

**Latency plan**: FP16 first, per-stage `trtexec` profile; INT8 later with calibration on own-drive frames
(multi-input calibrator). Kept in FP16: LayerNorm/softmax/attention of the ViT, deformable sampling offsets, 3D
regression, the BEV head. `_pin_fp16` becomes a `precision_map` in the meta.

**Recording specification**: `recording/protocol.md` (rig-agnostic): hardware-synchronised stereo pair recorded raw
(both eyes, exposure timestamps), raw IMU at its output rate stamped at sample time, GNSS stamped with its own fix
time, optional vehicle speed, OpenCV-format calibration with camera-to-ego and camera-to-IMU, per-drive acceptance
checks, offline depth labels at full resolution and future-near BEV accumulation.

## 6. Phases, acceptance tests, GPU budget, risks

| phase (weeks) | deliverables | acceptance (CPU tests on `sparc/data/synthetic.py`, pattern of `tests/test_adas_mt`) | on the Orin |
|---|---|---|---|
| 0 (2) | `sparc/` scaffold, `BackboneFactory`, `bakeoff` CLI (3 backbones → ONNX → Orin), op probes (GridSample, ScatterElements-add, LayerNorm fp16), `recording/protocol.md`, `schema.py` | schema round-trip; profile table; probe graphs pass `audit_ops` | bake-off table (ms + mAP after a 1-epoch BDD fine-tune, 0.5 GPU-day); op verdicts |
| 1 (3) | converters BDD/CULane/MTSD, `lane_rules.py`, `MixedDataset`, `augment.py` | flip/scale keep K–box–polyline–mask–BEV consistent on synthetic geometry; label masks per source; sampler weights within 2 % | — |
| 2 (3) | encoder/decoder/2D heads/lanes/DA, criterion, trainer, evaluators det/CULane/DA/sign, export + parity, **Stage 1** | 2-epoch learning test on synthetic data (mAP50 > 0.9, lane F1 > 0.8, DA IoU > 0.6), resume equality, ONNX parity | FP16 engine without BEV, `bench` ≤ 25 ms |
| 3 (3) | nuScenes converter, 3D head, `bev.py`, BEV evaluators, **Stage 2** | splat test: synthetic points land in the right cells; IPM of a flat plane reproduces a checkerboard; export audit | one engine with BEV, ≤ 33 ms |
| 4 (2) | MOT converter, ID head, `associate.py`, MOT evaluator, **Stage 3** | association recovers synthetic tracks under ego-motion; `bdd100k` eval runs | — |
| 5 (parallel) | `record.py`, `own_drive.py`, pseudo-label + self-BEV label tools, **Stage 4** | self-BEV on nuScenes (LiDAR as the fake depth camera, CAN bus as IMU/GPS) reaches map IoU > 0.6 against the HD map | first drives |
| 6 (2) | INT8, runner post-processing, ablation table, docs, Colab | bench JSON; FP16 engine vs ONNX eval parity | final numbers |

GPU-days (A100): Stage 1 ×4 runs 6, Stage 2 ×2 2, Stage 3 1, Stage 4 1, baselines (RF-DETR-S, YOLO26s, LSS) 2,
bake-off 0.5 → **~13–15 GPU-days** total. First milestone that already beats the current model: end of Phase 2.

**Risk register**
1. Deformable attention export → `GridSample` probe in Phase 0; `GatherND` fallback.
2. ViT FP16 overflow on Jetson → LayerNorm/softmax pinned to FP32; ConvNeXt-T as the clean-CNN escape; FP16-engine vs
   fp32-ONNX eval parity is a gate.
3. Licences (nuScenes / MTSD / CULane non-commercial) → `licence` field + `--commercial` manifest filter; a research
   model and a product model (BDD + own drives + Stage 4) are trained from separate manifests; weights derived from
   non-commercial data are not shipped in a product.
4. Weak direction/order labels → weight 0.3; the verified set is the only reported number; CULane gives clean order.
5. No drives yet → protocol first; nuScenes stands in for the whole own-drive pipeline (dry run in Phase 5).
6. Latency overrun → per-module trtexec budget table; knobs in section 3.
7. Multi-dataset gradient conflict → per-task sample normalisation, replay ratios, calibrated gains, per-source
   metrics every epoch; PCGrad only if a source regresses > 1 point.
8. Stereo depth range (~24 m at network width) → depth labels offline at full resolution, future-near accumulation for
   the far BEV, training clips the input depth to a random range in [8, 40] m with holes/noise, IPM covers beyond;
   BEV IoU reported per range band (0–10 / 10–25 / 25–48 m).
9. Runtime stereo cost → D1 / D2 / D3 measured on the device (Phase 0) and ablated for accuracy (Phase 3, B2).

## Verification (end to end)

- CPU: `pytest tests/test_sparc` (schema, converters on synthetic dumps, augmentation geometry, matcher/loss gating,
  splat/IPM unit tests, trainer learning test, export parity with ORT, fake-TensorRT multi-input runner).
- GPU: Stage-1 run on BDD+CULane+MTSD with the ablation table (A0–A2) before any 3D/BEV work; each later stage adds its
  row and must not regress the earlier metrics by > 1 point.
- Orin: Phase-0 bake-off and op probes with `trtexec`; `sparc bench` per output; `sparc eval` of the FP16 engine vs
  the ONNX on the same val split.
- Own data: recording protocol executed; `own_drive.py` + self-BEV labels validated first on nuScenes (map IoU > 0.6).

## Open points the user decides later
- Which backbone wins the Phase-0 bake-off (latency vs mAP on the device).
- Recording start date (the specification is ready) and the rectification alpha (field of view vs valid area).
- Licences: settled 2026-10-11 — research use, all datasets in one training manifest (the `licence` field stays for
  provenance).
