# Foundation-model distillation (Phase 3)

A frozen **DINOv3** ViT (via `timm`) supplies dense patch-token targets for the YOLO neck during training.
The teacher runs on the cloud training GPU only. It is not part of the deployed model:
`strip_training_only()` removes the projector (`model.kd_proj`) and the DA auxiliary classifier before export,
so the Jetson Orin Nano Super engine contains only the YOLO26 CNN and the two light heads.

```
image ─► YOLO26 backbone+neck ─► P3 (avg-pooled to s16) ⊕ P4 ─► projector (1x1-BN-SiLU-1x1) ─► (B,N,D) ─┐
   └───► frozen DINOv3 ViT (no grad, bf16) ──────────────────────────────────────────────► (B,N,D) ◄────┘
        loss = lambda(t) * [ cosine distance + relational (centred, scale-free) token-affinity term ]
```
`lambda(t)` decays from `weight` to `weight_end` (default 1.0 -> 0.1) by cosine over training. It is driven by the
criterion itself (`MultiTaskLoss.update()` once per epoch, `criterion.updates = k` on resume), so the trainer needs no
special coupling. If the teacher grid differs from the student's stride-16 grid (`input_scale` != 1) the projected
student map is resized to the teacher grid.

## Choosing the teacher (`--teacher`)
Forward cost at 384x640 (the YOLO26s training step is ~54 GFLOPs fwd+bwd):

| `--teacher` | timm model | params | teacher GFLOPs | ViT dim |
|---|---|---|---|---|
| `dinov3_s` | vit_small_patch16_dinov3 | 22M | 42 | 384 |
| `dinov3_s_plus` | vit_small_plus_patch16_dinov3 | 29M | 55 | 384 |
| **`dinov3_b`** (default) | vit_base_patch16_dinov3 | 86M | 165 | 768 |
| `dinov3_l` | vit_large_patch16_dinov3 | 303M | 584 | 1024 |
| `dinov3_h_plus` | vit_huge_plus_patch16_dinov3 | 841M | 1621 | 1280 |

Because the teacher costs cloud GPU time, not Jetson time, a larger teacher is allowed; quality gains with size
are not guaranteed for a 10M-parameter student (capacity gap), so compare `dinov3_b` vs `dinov3_l` in the ablation.
Teacher runs in bf16 autocast on CUDA (`dtype="auto"`); `--every k` distils every k-th step; `--teacher_scale 0.5` runs
the teacher at half resolution (4x cheaper, coarser targets).
Use `--teacher_ckpt <file>` when the machine cannot reach the Hugging Face hub: both the timm file from the hub
(`model.safetensors` of `timm/vit_base_patch16_dinov3.lvd1689m`) and Meta's official `dinov3_vit*16_pretrain_lvd1689m-*.pth`
are accepted (Meta's key names are converted like timm's downloader does); a file that does not match the architecture
raises instead of loading partially. Smoke-test the real weights once on the training box before a long run:
`python -c "import torch; from adas_mt.distill import FrozenTeacher; t=FrozenTeacher('dinov3_b'); x,g=t(torch.rand(1,3,384,640)); print(g, x.shape, torch.isfinite(x).all())"`
should print `(24, 40) torch.Size([1, 960, 768]) tensor(True)`.
`--teacher_scale` / `dtype="auto"` pick bf16 only on GPUs with native bf16 (A100/H100/L4), fp16 otherwise.

## Stage A: distillation-only pretraining on unlabelled frames
```bash
python -m adas_mt.distill.pretrain --images /data/all_frames --scale s --weights yolo26s.pt \
    --teacher dinov3_b --epochs 20 --batch 32 --imgsz 384 640
# -> runs/distill/exp/last.pt: backbone+neck distilled; the trained projector is saved under "kd_proj"
```
`--images` may be a converted dataset root: only `images/train` is used (never mask PNGs or val frames).
Backbone+neck train at `lr * student_lr_mult` (default 0.2) and the projector at `lr`: a pure distillation loss with no
detection anchor could otherwise erode the pretrained detection features (an unverified default; the A3 ablation
decides). Stage B (multi-task training) starts from it: `build_model("s", nc=9, weights="runs/distill/exp/last.pt")`.
The Stage-A projector is restored when Stage B enables distillation with the same teacher width; with a different
teacher it starts fresh (warned). Starting Stage B with a *fresh* projector would pull the neck away from the Stage-A
alignment at full distillation weight, which is why the projector travels with the checkpoint.

## Wiring it into a trainer (Phase 4 does this)
```python
model = build_model("s", nc=9, weights=..., kd_dim=teacher.dim)          # kd_proj exists from construction
model.set_distiller_factory(lambda m: Distiller(m, teacher))            # teacher is built/moved lazily
```
**Order matters.** The optimizer, EMA and the DDP wrapper are built from `model.parameters()` *before* the first loss
call, which is where the criterion (and so the `Distiller`) is created. A projector registered at that point is in no
optimizer (never trained), not in the EMA or `last.pt` (NaN-recovery `load_state_dict` then fails on missing keys),
and is not synchronised across DDP ranks. Hence `kd_dim` at model construction, and `Distiller(...)` raises if
`model.kd_proj` is missing.

`MultiTaskModel.init_criterion()` builds `MultiTaskLoss(model, distiller=factory(model))`, which is also what
Ultralytics calls when resuming, so distillation survives a resume. The factory, the criterion and the teacher are
excluded from pickles/deepcopies (`__getstate__`), so checkpoints and the EMA never carry the teacher. On skipped
(`every>1`) steps the projector is still run, which keeps DDP safe even with `find_unused_parameters=False`.

## Tests
`tests/test_adas_mt/test_distill.py` uses randomly initialised ViTs (including the real DINOv3 architecture at
384x640, to check the token grid with RoPE and register tokens), because pretrained weights cannot be downloaded in CI. It checks the
frozen teacher, gradients, schedule, resume (distiller + `E2ELoss` counter), DDP-safe skipped steps, strip-for-export
parity, scale-free affinity term, and that Stage A improves alignment and its checkpoint (incl. projector) reloads.
