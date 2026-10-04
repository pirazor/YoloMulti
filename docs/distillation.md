# Foundation-model distillation (Phase 3)

A frozen DINOv3 / DINOv2 ViT (via `timm`) supplies dense patch-token targets for the YOLO neck during
training. The teacher is **not** part of the deployed model; `strip_training_only()` removes the
projector (`model.kd_proj`) and the DA auxiliary classifier before export.

```
image ─► YOLO26 backbone+neck ─► P3 (avg-pooled to s16) ⊕ P4 ─► projector (1x1-BN-SiLU-1x1) ─► (B,N,D) ─┐
   └───► frozen ViT teacher (no grad) ───────────────────────────────────────────────► (B,N,D) ◄───────┘
                                  loss = lambda(t) * [ cosine distance + token-affinity MSE ]
```
`lambda(t)` decays from `weight` to `weight_end` by cosine over training (`Distiller.set_progress(p)`).
If the teacher grid differs from the student's stride-16 grid (patch 14, or `input_scale` != 1) the
projected student map is resized to the teacher grid.

## Choosing the teacher (measured at 384x640; student = YOLO26s train step 53.6 GFLOPs fwd+bwd)
| teacher (`--teacher`) | `input_scale` | teacher GFLOPs | training step cost |
|---|---|---|---|
| `dinov3_s` (ViT-S/16) | 1.0 / 0.5 | 41.5 / 10.5 | x1.78 / x1.20 |
| **`dinov3_s_plus`** (default) | 1.0 / 0.5 | 55.2 / 14.0 | **x2.03** / x1.26 |
| `dinov3_b` (ViT-B/16) | 1.0 / 0.5 | 165.1 / 41.9 | x4.08 / x1.78 |
| `dinov2_s` / `dinov2_b` | patch 14 | similar | Apache-2.0 alternative |

FLOP ratios, not wall time (ViT matmuls run more efficiently on a GPU than YOLO convs, so the real
overhead is usually lower). `--every k` distils every k-th step to cut it further.
**Licence:** DINOv3 weights use Meta's DINOv3 licence (commercial use allowed, with restrictions): get a legal review
before using it for a product. DINOv2 is Apache-2.0.

## Stage A: distillation-only pretraining on unlabelled frames
```bash
python -m adas_mt.distill.pretrain --images /data/all_frames --scale s --weights yolo26s.pt \
    --teacher dinov3_s_plus --epochs 20 --batch 32 --imgsz 384 640
# -> runs/distill/exp/last.pt  (backbone+neck distilled, projector/aux stripped)
```
Stage B (multi-task training, Phase 4) starts from it: `build_model("s", nc=9, weights="runs/distill/exp/last.pt")`.
Use `--teacher_ckpt /path/teacher.safetensors` when the machine cannot reach the Hugging Face hub.

## In the joint loss
`MultiTaskLoss(model, distiller=Distiller(model, teacher))` appends a sixth loss element (`kd_loss`,
plus `kd_cos` = mean cosine similarity as a monitor). `model.kd_proj` is an ordinary submodule, so the
trainer's optimizer and EMA pick it up with no special casing.

## Tests
`tests/test_adas_mt/test_distill.py` uses randomly initialised ViTs (including the real DINOv3 and
DINOv2 architectures, to check the token grid with RoPE / register tokens / patch 14), because the
pretrained weights cannot be downloaded in CI. They check: frozen teacher and unchanged weights after
an optimizer step, gradients into backbone/neck/projector, `every=k`, schedule, strip-for-export keeps
inference identical, and that Stage A raises student-teacher cosine similarity and writes a checkpoint that
`build_model(weights=...)` reloads.
