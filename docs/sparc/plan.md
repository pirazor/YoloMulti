<!-- Status: v2 design, 2026-10-11. Stage A code implemented (docs/sparc/stage_a.md: converters, virtual camera, dataset, decoder heads, criterion, trainer, evaluators; CPU-verified, untrained); Phase 0 bake-off pending on the device. Replaces plan_v1_superseded.md. -->
starts on public datasets before any own recordings, and the novelty claims are stated honestly. Phase 0 (docs/sparc/phase0.md)
remains valid: the bake-off tooling, rig geometry and schema are reused; the stereo head becomes an optional ablation. -->

# SPARC v2: one temporal sparse-query perception network, trained first on public data

## 0. What the review changed

1. **Depth is an auxiliary signal, not a dependency.** v1 built the BEV map by splatting features with measured depth.
   v2 lifts to BEV without depth (query-based lifting) and uses depth only as (a) a training target for an auxiliary
   depth head and (b) an optional input the network is trained to work without (depth dropout). The stereo head of
   Phase 0 is kept as an ablation, not as the basis of anything.
2. **Public datasets first.** Every output of the network has a public source of supervision (section 3); the
   complete model is trained and benchmarked before a single own frame is recorded. Own recordings are a fine-tuning
   stage, and the recording specification (`recording/protocol.md`) is unchanged.
3. **Temporal processing is the main accuracy lever.** The decoder propagates its queries from the previous frame,
   moved by ego-motion. In the literature this is the single largest gain available for camera 3D detection, tracking
   and online mapping, and it costs almost nothing at inference (section 1). v1 had no temporal model.
4. **Honest novelty.** Section 7 lists what is taken from published work and what is new. No performance jump is
   promised; section 5 defines the measurements that decide every claim.

## 1. Where the gains come from (evidence, and what SPARC does with it)

| lever | measured effect in the literature | in SPARC v2 | inference cost |
|---|---|---|---|
| Foundation features (DINOv3 / DINOv2 trunks) | DEIMv2-S: 50.9 AP with 9.7 M params / 25.6 GFLOPs (ViT-Tiny distilled from DINOv3); RF-DETR-S 53.0 AP @ 3.5 ms T4; YOLO26s 48.6 AP. +2–4 AP at equal latency for small models. | backbone bake-off (Phase 0 tooling): DINOv3 ViT-T/S, DINOv2 ViT-S, DINOv3 ConvNeXt-T, YOLO26s reference | the backbone itself |
| Temporal query propagation | StreamPETR: +11.4 mAP / +13.1 NDS over its single-frame baseline (Focal-PETR, R101) "with negligible computation cost"; Sparse4Dv3: tracking by propagated instances, +7.6 AMOTA; StreamMapNet: streaming map queries beat single-frame mapping at 14 FPS | all typed queries (agents, lanes, map, traffic elements) are propagated frame to frame, moved by measured ego-motion; identities come from propagation | top-k query memory + one ego-motion transform; no extra backbone pass |
| Training recipe for DETR-style heads | DEIM (dense one-to-one matching + matchability-aware loss): beats RT-DETRv2 / D-FINE at half the epochs; D-FINE's distribution refinement; RT-DETRv4's frozen-DINOv3 feature injection (+1–2 AP, training only) | DEIM matching and MAL for agents and traffic elements; D-FINE-style box distribution; DINOv3 teacher on the encoder; same-architecture distillation SPARC-L → SPARC-S at the end | none |
| Depth as supervision, not as input | BEVDepth: explicit depth supervision from LiDAR improves camera-only BEV accuracy and robustness; modality dropout is the standard way to keep a model working when a sensor is missing (UniBEV) | auxiliary depth head trained on LiDAR (nuScenes, Argoverse 2, Waymo, DrivingStereo) and on foundation-model pseudo-depth elsewhere; depth input with dropout | ~0.5 GFLOPs |
| Geometry-aware multi-dataset training | Omni3D / Cube R-CNN: a *virtual camera* (one focal length for all datasets) is what makes mono-3D training across datasets with different intrinsics work, and it enables scale augmentation | every image is warped to a canonical pinhole camera before the network; K and cam2ego are inputs; 3D targets are expressed in virtual depth | none |
| Resolution over lifting cleverness | Simple-BEV: input resolution and batch size move BEV accuracy far more than the lifting method; a parameter-free bilinear lifter is competitive | spend the Orin budget on resolution (384×640 → 448×768 is an ablation) rather than on a heavier BEV module; BEV lifting = cheap deformable sampling at a few heights | resolution-linear |
| One trunk for every task | the current rig runs five separate engines (detector, lane net, sign classifier, stereo net, face/gaze) | one engine; the multi-task baselines on BDD100K (YOLOP 76.4 mAP50 / 89.0 DA mIoU / 44.0 lane IoU; A-YOLOM-s 81.1 / 91.0 / 49.7; RMT-PPAD 84.9 / 92.6 / 56.8, all measured under one protocol in arXiv 2511.05557) are the bar | shared |
| Attention cost | LW-DETR / RF-DETR replace most global self-attention blocks by windowed ones | windowed attention in 3 of every 4 ViT blocks (bake-off variant) | −4–5 GFLOPs on ViT-S |
| INT8 / sparsity on Orin | mixed: RF-DETR-Base 14.8 ms FP16 → 11.8 ms INT8 on Orin Nano Super; but a DEIMv2 INT8 build on Orin NX (JetPack 6.2) ran *slower* than FP16 in one report, and 2:4 sparsity gave no speed-up on Orin in another | FP16 first; INT8 only for the convolutional parts, kept only if measured faster at equal accuracy; no reliance on sparsity | measured, not assumed |

Levers investigated and **not** adopted: separate stereo network (55 ms on Orin NX, more than the budget); learned
per-pixel depth distributions for lifting (LSS) as the only path to BEV (fails when depth fails); token pruning
(dynamic shapes, TensorRT-unfriendly, small gains on dense tasks); DLA offload (the Orin Nano has no DLA);
uncertainty-based loss weighting as the default (drifts with partial labels; kept as ablation B3).

## 2. Architecture

```
                 previous frame: top-k typed queries + their 3D / BEV states ──► moved by ego-motion (IMU+GNSS, or dataset pose)
                                                                                     │
RGB (virtual camera 384×640) ──► backbone ──► hybrid encoder ──► P3 P4 P5 ──────────┼──────────────────────────────────┐
     K, cam2ego (inputs)                         │                                   ▼                                  │
                                                 │        TEMPORAL TYPED-QUERY DECODER (3 layers, d = 256)              │
                                                 │   agent queries      : 2D box, class, 3D state, attributes, ID     │
                                                 │   traffic-element q. : light colour / sign meaning + fine class     │
                                                 │   lane queries       : 3D polyline in the ego frame, order, dir,    │
                                                 │                        continuity, type (2D view by projection)     │
                                                 │   map queries (BEV)  : coarse 60×40 grid, lifted from the image by  │
                                                 │                        deformable sampling at 4 heights (no depth)  │
                                                 │   topology head      : which light/sign governs which lane          │
                                                 ▼                                                                      │
                               aux depth head (stride 8, 64 bins) ──► depth-aware features for 3D and BEV ◄── optional depth input (dropout-trained)
                               drivable-area image head (stride 4)
                               BEV decoder: map queries → 240×160 drivable / lane line / heading / agent occupancy
```

**Inputs.** `image` (rectified, warped to the canonical virtual camera: fixed focal length, principal point at the
centre), `K` and `cam2ego` of the virtual camera, `ego_motion` (4×4 from the previous frame), `memory` (previous
top-k queries and states; zeros on the first frame), and an optional `depth` map. One engine serves any rig, because
the warp to the virtual camera happens in pre-processing from the rig's own calibration.

**Backbone and encoder.** As Phase 0: P2 for detail, hybrid encoder at 256 channels for P3–P5. The DINOv3 teacher
regularises the encoder during training (RT-DETRv4 style); it does not exist at inference.

**Auxiliary depth head.** A 1×1 conv on the stride-8 encoder map predicts a 64-bin depth distribution per cell
(BEVDepth-style), supervised by LiDAR where it exists, by DrivingStereo disparity, and by pseudo-depth from a metric
monocular foundation model (UniDepth / Metric3D class) on datasets without range sensors. Its expected depth and
confidence become a *feature* (a depth positional encoding) for the 3D heads and the BEV lifting. If a `depth` input
is present, it is fused into the same head with 50 % dropout during training, so the network is calibrated to use it
when it is good and ignore it when it is absent or wrong. Nothing downstream requires it.

**Decoder: typed, temporal.** Four query types share the decoder layers (self-attention over all queries with a type
embedding, deformable cross-attention over P3–P5, FFN):
- *agents* (200): selected from encoder tokens (RT-DETR query selection) **or propagated from frame t−1**; heads: 2D box
  (D-FINE distribution), class over the unified vocabulary, 3D state (virtual depth, yaw, size), ID embedding;
- *traffic elements* (50): lights and signs; heads: 2D box, colour / meaning (OpenLane-V2's 13 attributes), fine sign
  class (unified sign vocabulary, section 3), association to a lane via the topology head;
- *lanes* (24): a 3D polyline in the ego frame (x forward, y lateral, z height, 20 points to 60 m), existence, order
  relative to ego, travel direction, continuity, type. The image-space polyline is the *projection* of the 3D one
  through K / cam2ego, so 2D-only datasets (CULane, BDD100K) supervise the 3D lane through the projection with a
  ground-plane prior, and 3D datasets (OpenLane, maps) supervise it directly. Lanes are parallel, 3–4 m apart and
  continuous in time: priors a 2D row-anchor head cannot use;
- *map* (60×40 BEV cells of 0.8 m): each cell samples image features at 4 heights through deformable attention
  (BEVFormer-style spatial cross-attention, depth-free) weighted by the auxiliary depth distribution, then a small
  CNN decodes 240×160 drivable area, lane lines with heading, agent occupancy.

*Temporal propagation.* After frame t, the top-k agents (by score), all lanes with existence > 0.3, the traffic elements
and the map feature grid are stored. At t+1 their 3D states are moved by the measured ego-motion (agents: position;
lanes: polyline; map: warped grid), re-encoded as query content + position, and concatenated with the fresh queries.
A propagated agent keeps its identity, so tracking is the propagation itself (Sparse4Dv3 / MOTR style); the ID
embedding remains as the fallback for re-identification after occlusion. Training uses clip pairs and triplets
with the dataset's ego poses; at deployment the ego-motion comes from the IMU + GNSS (or visual odometry).

**Dense drivable-area head.** Kept (BDD100K labels it in the image; it is cheap). Ablation B5 replaces it by rendering
the BEV drivable area into the image.

**Not in the deployed graph.** The DINOv3 teacher, the lane-mask auxiliary head, denoising queries, and the stereo head
(ablation D1 only).

### Budget (CPU-measured FLOPs from Phase 0 where the module exists; estimates marked ~)

| module | GFLOPs @384×640 | note |
|---|---|---|
| backbone | 7.5 (YOLO26s) … 12.8 (ViT-T) … 44 (ViT-S / ConvNeXt-T) | windowed attention ~−4 on ViT-S |
| encoder | 8.0 | measured |
| decoder (4 query types, 3 layers) | ~5.5 | 4.4 measured for 224 queries; +50 traffic elements |
| aux depth head | ~0.5 | 1×1 conv + softmax |
| drivable-area head | 1.2 | measured |
| BEV lifting (2400 cells × 4 heights) + decoder | ~3 | replaces the 6.1 of the v1 splat path |
| temporal memory | ~0.2 | MLPs on ≤ 300 queries + one 240×160 warp |
| **total** | **26 (YOLO26s) … 31 (ViT-T) … 62 (ViT-S)** | the Orin numbers decide; ≤ 28 ms FP16 is the bar |

## 3. Public datasets: what trains what

| output | datasets (front camera unless noted) | labels used |
|---|---|---|
| road users 2D | BDD100K det (100k, 10 classes), nuImages, ZOD Frames (100k, 2D to 245 m), Waymo 2D, Argoverse 2 | boxes; class vocabulary per source (Null-BCE masking) |
| road users 3D state | nuScenes (CAM_FRONT), Argoverse 2 sensor (ring_front_center, 20 fps, LiDAR), Waymo camera-only 3D (front), ZOD (3D boxes), KITTI | 3D boxes → virtual depth, yaw, size |
| tracking | BDD100K MOT (box_track), nuScenes, Argoverse 2, Waymo, KITTI tracking | track ids across frames + ego poses |
| traffic-light colour | BDD100K (`trafficLightColor`), OpenLane-V2 traffic elements (red / green / yellow + arrow meanings, with topology), DTLD (230k annotations, red-yellow state), Bosch (13k frames), LISA | colour / meaning per box |
| traffic-sign fine class | MTSD (206k signs, 313 classes), ZOD (446k signs, 156 classes), TT100K (221 classes), GTSRB crops | unified sign vocabulary (section 3.1) |
| lanes 2D + attributes | BDD100K lanes (style, type, parallel / perpendicular), CULane (4 lanes with fixed ego order), CurveLanes (up to 9 lanes), ZOD lane-marking instances (colour, cardinality) | polylines, order, continuity, type |
| lanes 3D + order | OpenLane (200k frames, 880k lanes, 14 categories, ego-relative position of up to 4 lanes), ONCE-3DLanes, Apollo synthetic | 3D polylines |
| travel direction | nuScenes map (directed lanes), Argoverse 2 map (lane segments with direction, neighbours, marking types), OpenLane-V2 centerlines (directed) → projected into the front camera with the ego pose | per-lane direction, derived without rules |
| drivable area (image) | BDD100K (direct / alternative), ZOD ego-road, Mapillary Vistas (road) | masks |
| BEV map | nuScenes and Argoverse 2 HD maps rasterised in the ego frame (drivable area, lane dividers with heading, ordering from lane neighbours) | 240×160 rasters |
| topology (element ↔ lane) | OpenLane-V2 (on Argoverse 2 and nuScenes) | lane–traffic-element pairs |
| depth (auxiliary) | nuScenes / Argoverse 2 / Waymo / ZOD LiDAR, DrivingStereo (180k stereo pairs with LiDAR disparity), KITTI; pseudo-depth from a metric mono model on BDD100K / CULane | per-pixel metric depth |

All of these are research licences or more permissive (ZOD is CC BY-SA); the `licence` field in the schema records
provenance. Every converter writes the Phase-0 `Sample` schema; `LabelMask` says which losses apply to a frame.

### 3.1 Harmonisation decisions
- **Virtual camera.** Images are warped to a 60° horizontal-FOV pinhole at 640 px width (fx ≈ 554). Datasets without
  calibration (BDD100K, CULane) get a nominal K estimated once per dataset from lane vanishing geometry; their 3D
  losses are masked anyway, so an error there only affects the lane ground-plane prior.
- **Class vocabularies.** Road users: BDD100K's 10 classes as the base; other sources map onto it with per-source
  `cls_vocab` masks (nuScenes has no "traffic sign" boxes, Waymo has no "rider", etc.). Signs: a unified taxonomy built
  from MTSD's hierarchy; ZOD and TT100K classes map onto it where a one-to-one mapping exists and are masked
  otherwise; region-specific classes stay region-specific (the fine-tuning stage adds the deployment region's signs).
- **Lane semantics.** OpenLane's categories, BDD's `laneType`, ZOD's colour / cardinality all map onto {single /
  double} × {white / yellow} + curb; CULane and CurveLanes supervise geometry and order only.
- **Lane order and direction.** From CULane (fixed order), OpenLane (ego-relative position), and maps (nuScenes,
  Argoverse 2: direction and neighbours); v1's hand-written colour rules become a fallback for BDD only, flagged
  `weak`.
- **Per-source evaluation.** Every stage reports metrics per dataset; a source that regresses by > 1 point when
  another is added is a label conflict to resolve, not noise.

## 4. Training plan (one A100/H100, public data only until stage E)

| stage | data (sampler weights) | what is trained | GPU-days | acceptance (must hold to continue) |
|---|---|---|---|---|
| **A: 2D** (weeks 1–3) | BDD det/DA/lanes 0.35, ZOD 0.2, OpenLane-2D 0.15, CULane 0.1, MTSD + TT100K 0.1, DTLD/Bosch/LISA 0.05, nuImages 0.05 | backbone (bake-off winner), encoder, agents 2D, traffic elements (colour + fine class), lanes (projected 2D + attributes), drivable area; DEIM matching, DINOv3 teacher | 3 per run, 3 runs (bake-off top-2 + recipe ablation) | BDD det mAP ≥ DEIMv2-S fine-tuned on BDD − 0.5 while also producing DA + lanes; DA mIoU ≥ 92; CULane F1 ≥ 79; MTSD top-1 ≥ current `sign_classifier` |
| **B: 3D, depth-aux, BEV** (weeks 4–6) | + nuScenes front 0.15, Argoverse 2 front 0.15, Waymo front 0.1, OpenLane-3D 0.1, DrivingStereo 0.05 (stage-A replay 0.45) | 3D state, aux depth (LiDAR + pseudo-depth, depth dropout), 3D lanes, map queries + BEV decoder, topology head | 3 | OpenLane F1 ≥ 58 (BEV-LaneDet level); nuScenes front-cam 3D AP within 2 of a MonoDETR-class model re-evaluated on the same subset; BEV drivable IoU ≥ a Simple-BEV single-camera baseline trained by us; no stage-A metric down by > 1 |
| **C: temporal** (weeks 7–9) | clip pairs / triplets from BDD MOT, nuScenes, Argoverse 2, Waymo, OpenLane sequences | query propagation, tracking by propagation, map streaming, ID embedding | 2 | ≥ +3 mAP 3D and ≥ +2 OpenLane F1 over the single-frame model at equal cost; BDD MOT mMOTA ≥ ByteTrack (45.5) and mIDF1 ≥ 54.8 |
| **D: efficiency** (weeks 10–11) | stage-C data | SPARC-L → SPARC-S distillation, windowed attention, decoder depth / query counts, resolution ablation, INT8 of conv parts | 2 | ≤ 28 ms FP16 median on the Orin with all outputs; FP16 engine within the parity limits of Phase 0; INT8 kept only if faster at ≤ 0.5 point loss |
| **E: own data** (when recorded) | own drives per `recording/protocol.md` + replay 0.5 | fine-tune; offline stereo depth as aux target; region signs; pseudo-labels from SPARC-L | 1 | per-source metrics on the 500-frame verified set; no public-benchmark regression > 1 |

Total ≈ 19 GPU-days plus baselines (DEIMv2-S / RF-DETR-S on BDD, CLRerNet, Simple-BEV front, MonoDETR subset, ByteTrack): ≈ 3.
The first deliverable that beats the current model on every output it shares (detection, drivable area, lanes,
signs) is the end of stage A, about week 3.

### 4.1 Losses and matching (stage A onward)
Hungarian matching per query type (agents and traffic elements: DEIM dense one-to-one with MAL; lanes: polyline L1 +
existence). Every term is gated by `LabelMask` and normalised over the samples that carry it. Box: D-FINE distribution
loss + GIoU; 3D: L1 on virtual log-depth, sin/cos yaw, log size; lanes: L1 on 3D points (3D sources) or on projected
2D points (2D sources) + CE attributes; map: CE + Dice per layer, L1 heading on lane cells; depth-aux: CE over bins
with LiDAR / stereo / pseudo-depth targets (pseudo-depth weighted 0.3); topology: BCE on element–lane pairs; ID:
InfoNCE across the clip. Gains are fixed and calibrated once by the gradient-norm procedure already documented in
`docs/training.md`.

## 5. Ablations that decide the design (one table in `docs/sparc/results.md`)

| id | question | compared |
|---|---|---|
| A1–A3 | backbone (bake-off top-2) and recipe | DEIM vs plain Hungarian; DINOv3 teacher on/off |
| B1 | BEV lifting | depth-free deformable sampling vs v1 depth splat (with LiDAR depth) vs IPM-only |
| B2 | depth input at inference | none / foundation pseudo-depth / LiDAR-quality / corrupted (noise, holes, 2× scale error): the model must degrade gracefully |
| B3 | loss weighting | fixed calibrated gains vs uncertainty weighting |
| B4 | virtual camera | on vs off (3D AP across datasets) |
| B5 | drivable area | image head vs rendered from BEV |
| C1 | temporal | single frame vs propagation; with measured vs noisy ego-motion (±10 % scale, ±2° yaw) |
| C2 | tracking | propagation only vs propagation + ID embedding vs ByteTrack on the same detections |
| D1 | stereo head | with vs without, accuracy of 3D / BEV vs cost |
| D2 | resolution | 384×640 vs 448×768 (small signs / lights recall vs ms) |
| D3 | precision | FP16 vs mixed INT8 on the Orin |

## 6. Risks specific to public-data-first

1. **Camera domain gap**: public front cameras range from 50° (Waymo) to 80° (KITTI) FOV; the virtual camera removes
   intrinsics differences, not viewpoint height (1.2–2.2 m) → cam2ego is an input and height is augmented ±0.3 m.
2. **Label conflicts across sources** (BDD "lane" means marking; OpenLane-V2 "lane" means centerline) → separate
   query roles (markings vs centerlines are different targets of the lane queries' type head), per-source metrics.
3. **Sign taxonomy** differs by country → unified vocabulary with masking; the shipped model is fine-tuned on the
   deployment region.
4. **Pseudo-depth quality** on BDD → weight 0.3 and B2 measures robustness to wrong depth.
5. **Temporal training cost** (clips double the data loader work) → pairs/triplets only, cached encoder features for
   the backbone-frozen phase of stage C.
6. **TensorRT support** for the new pieces: deformable sampling (GridSample, already probed), the BEV warp (GridSample),
   top-k with dynamic memory (made static: fixed k with validity mask) → Phase-0 probes extended before stage C.

## 7. What is new and what is not

Taken from published work: DINOv3 / DINOv2 trunks and spatial adapters (DEIMv2, RF-DETR); hybrid encoder, deformable
attention and query selection (RT-DETR); DEIM matching; D-FINE box distributions; depth-aware queries (MonoDETR);
auxiliary depth supervision (BEVDepth); depth-free query lifting (BEVFormer, Simple-BEV); temporal query propagation
(StreamPETR, Sparse4D, StreamMapNet); topology heads (TopoNet / TopoMLP); virtual camera (Omni3D); modality dropout
(UniBEV).

New in this design, and testable:
1. **One temporal decoder whose typed queries cover road users, traffic elements, lanes, the BEV map and their
   topology on a single front camera**, propagated jointly with measured ego-motion; published systems do one or two
   of these, and the map / topology ones on multi-camera rigs with server GPUs.
2. **Depth-optional training**: a single network that uses a depth input when present and is measured (B2) to lose
   nothing when it is absent or corrupted.
3. **3D lanes supervised from 2D-only datasets through the projection**, which makes CULane / BDD / CurveLanes usable
   for a 3D lane head and gives order and direction labels from maps instead of hand rules.

Expected magnitude, from the cited numbers: foundation features +2–4 AP; temporal propagation the largest single
step (double digits in 3D on nuScenes in the literature; unknown for a single front camera until stage C);
everything else one to two points each. These are the only promises this document makes.

## 8. Code changes relative to Phase 0

| area | change |
|---|---|
| `sparc/data/schema.py` | add `elements` (traffic elements with attribute + fine class), `lanes3d`, `topology`, `ego_motion`; virtual-camera fields |
| `sparc/data/converters/` | bdd100k, culane, curvelanes, openlane, openlane_v2, nuscenes_front, argoverse2_front, waymo_front, zod, mtsd, tt100k, dtld/bosch/lisa, drivingstereo; `virtual_camera.py` |
| `sparc/nn/decoder.py` | traffic-element and map query types; temporal memory in/out; topology head |
| `sparc/nn/bev.py` | `QueryLifter` (deformable sampling at 4 heights) replaces `DepthSplat` as the default; splat kept for B1 |
| `sparc/nn/depth.py` | auxiliary depth head with depth-input fusion and dropout |
| `sparc/nn/lanes.py` | 3D polyline head + differentiable projection to the image |
| `sparc/engine/` | lean trainer with clip sampling, DEIM matching, label-mask gating, per-source metrics (stage A first) |
| `sparc/tools/bakeoff.py` | probes for the BEV warp and the static top-k memory |

## Sources
- RF-DETR: https://rfdetr.roboflow.com/ ; DEIMv2 "Real-Time Object Detection Meets DINOv3": https://arxiv.org/abs/2509.20787 ;
  DEIM: https://arxiv.org/abs/2412.04234
- StreamPETR: https://arxiv.org/abs/2303.11926 ; Sparse4Dv3: https://arxiv.org/abs/2311.11722 ; StreamMapNet: https://arxiv.org/abs/2308.12570
- Simple-BEV: https://arxiv.org/abs/2206.07959 ; UniBEV (modality dropout): https://arxiv.org/abs/2309.14516
- Omni3D / Cube R-CNN virtual depth: https://arxiv.org/abs/2207.10660 ; MonoDETR: https://arxiv.org/abs/2203.13310 ;
  MonoDINO-DETR: https://arxiv.org/abs/2502.00315 ; UniDepth: https://openaccess.thecvf.com/content/CVPR2024/papers/Piccinelli_UniDepth_Universal_Monocular_Metric_Depth_Estimation_CVPR_2024_paper.pdf
- OpenLane: https://github.com/OpenDriveLab/OpenLane ; OpenLane-V2: https://github.com/OpenDriveLab/OpenLane-V2 ;
  OpenLane leaderboard (aggregator): https://hyper.ai/en/sota/tasks/3d-lane-detection/benchmark/3d-lane-detection-on-openlane
- ZOD: https://zod.zenseact.com/annotations/ and https://arxiv.org/abs/2305.02008 ; Argoverse 2: https://www.argoverse.org/av2 ;
  Waymo camera-only 3D: https://waymo.com/open/challenges/2022/3d-camera-only-detection/ ; DrivingStereo: https://openaccess.thecvf.com/content_CVPR_2019/html/Yang_DrivingStereo_A_Large-Scale_Dataset_for_Stereo_Matching_in_Autonomous_Driving_CVPR_2019_paper.html
- Traffic lights (DTLD, LISA, Bosch): https://arxiv.org/abs/2409.07284 ; UCMCTrack: https://arxiv.org/abs/2312.08952
- BDD100K multi-task numbers under one protocol: https://arxiv.org/abs/2511.05557
- Orin INT8 / sparsity reports: https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549 ,
  https://forums.developer.nvidia.com/t/stuctured-sparsity-2-4-does-not-improve-inference-performance-on-jetson-orin/263685
