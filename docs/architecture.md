# Architecture (Phase 2)

```
image (B,3,384,640)
  └─ YOLO26 backbone ──P2 (layer 2, stride 4)─────────────────────┐
        └─ neck (PAN) ── P3 (layer 16, s8) ── P4 (19, s16) ── P5 (22, s32)
                              │                │                  │
                              ├─ Detect (end2end, reg_max=1) ◄────┴─ P3,P4,P5   -> (B,300,6) [x1,y1,x2,y2,score,cls], no NMS
                              ├─ DAHead  : P3 + up(P4) -> DW 3x3 + 1x1 -> logits s8 -> bilinear x8   (+ aux on P4, training only)
                              └─ LaneHead: P2 + up(P3 + up(P4)) -> DW 3x3 + 1x1 -> nc*16 -> PixelShuffle(4)
```

* Backbone, neck and `Detect` are stock Ultralytics 8.4 YOLO26 (`yolo26{n,s}.yaml`); nothing is patched.
* `MultiTaskModel` only adds layer 2 to the saved-output list and two small heads (`adas_mt/nn/heads.py`).
* The DA head uses the shallow stride-8 features (large smooth regions); the lane head needs the stride-4
  backbone skip and a sub-pixel classifier so 4-8 px lines survive. The lane classifier bias starts at
  p(background)=0.99.
* `MultiTaskModel.load()` transfers a YOLO26 checkpoint and **raises** when < 95% of backbone+neck
  parameters match (loading an `s` checkpoint into an `n` model used to silently train from scratch);
  `build_model(scale, weights=...)` also checks the checkpoint's scale up front.
* `fuse()` also folds Conv+BN inside the new heads.

## Cost of the DEPLOYED graph (GFLOPs, 2 x MACs, batch 1, fused: `python -m adas_mt.utils.profile --scale s --imgsz 384 640`)

`profile_model` fuses a copy first: `fuse()` folds Conv+BN and drops the one-to-many detection branch, which the
training graph still runs (an earlier un-fused table overstated the deployed cost by ~7%).

| model | input | backbone | neck | det | DA | lane | **total** | params |
|---|---|---|---|---|---|---|---|---|
| n | 384x640 | 1.85 | 1.05 | 0.26 | 0.03 | 0.17 | **3.37** | 2.40M |
| s | 384x640 | 7.23 | 4.17 | 1.01 | 0.13 | 0.58 | **13.12** | 9.56M |
| m | 384x640 | 23.60 | 13.45 | 3.77 | 0.52 | 2.11 | **43.44** | 20.69M |
| n | 640x640 | 3.10 | 1.78 | 0.44 | 0.06 | 0.29 | 5.67 | 2.40M |
| s | 640x640 | 12.10 | 6.99 | 1.69 | 0.22 | 0.97 | 21.97 | 9.56M |

The legacy YOLOv13 multi-task decoders, measured with the same counter, cost **65 GFLOPs at 640x640 and 39 at
384x640** for the two of them, i.e. about 10x the whole detector. The new heads are 5% of the s model.
(Counts exclude elementwise ops and resizes; Orin latency must still be measured with TensorRT. The segmentation
heads are nearly free, so the student scale is a pure latency-budget choice: compare `s` and `m` on the device.)

## Loss (`adas_mt/nn/loss.py`)
`E2ELoss` (one-to-many + one-to-one, ProgLoss-style decay) + DA (CE + Dice + 0.4 x aux CE) + lane
(CE + focal Tversky). Segmentation uses only non-255 pixels of the packed mask, so padding and
unannotated tasks give exactly zero loss and zero gradient (a NaN in focal Tversky for all-ignored
batches was found and fixed by the tests).
