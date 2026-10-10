# Training and validation (Phase 4)

`MultiTaskTrainer` is Ultralytics' `DetectionTrainer` with the multi-task pieces added (AMP, EMA, warmup,
close_mosaic, DDP, resume, plots and checkpoints are the stock ones).

## Pipeline
```
1. convert   python -m adas_mt convert -- --src <supervisely> --dst data/bdd --group_regex '^([0-9a-f]{8})-'
2. Stage A   python -m adas_mt.distill.pretrain --images data/bdd --scale s --weights yolo26s.pt --teacher dinov3_b   (optional)
3. Stage B   python -m adas_mt train --data data/bdd/data.yaml --model <yolo26s.pt | runs/distill/exp/last.pt> --distill --device 0,1,2,3
4. validate  python -m adas_mt val --weights runs/mt/exp/weights/best.pt --data data/bdd/data.yaml   # geometry read from runs/mt/exp/mt.yaml
```
Ablation A0..A3 from the plan: A1 = Stage B without `--distill`; A2 = with `--distill`; A3 = A2 started from the Stage A checkpoint.

## Configuration: two sections (`adas_mt/cfg/default.yaml`)
* `train:` ordinary Ultralytics arguments (`epochs`, `batch`, `lr0`, `optimizer`, augmentation gains...). CLI flags override them.
* `mt:` multi-task settings Ultralytics would reject: `imgsz: [384, 640]`, `scale`, `loss_gains`, `fitness` weights,
  `head_lr_mult`, and the `distill:` block. Unknown keys raise. The effective values are written to `<run>/mt.yaml`; resume
  and DDP workers read them back from there (DDP workers are re-created from `vars(args)`, so a temporary copy outside the
  run directory is handed over through `ADAS_MT_CFG`, because Ultralytics deletes the run directory before spawning workers).

Defaults worth knowing:
* `optimizer: AdamW, lr0: 0.001, warmup_bias_lr: 0.0` set explicitly. `optimizer=auto` switches to MuSGD on runs over 10k iterations and ignores `lr0`.
* **`nms: false`**: validation uses the NMS-free head that ships. Ultralytics' default (`None`) validates the one-to-many head + NMS, which is not the deployed model.
* `head_lr_mult: 3`: `da_head`, `ll_head` and `kd_proj` are freshly initialised; the pretrained trunk keeps the base LR.
* `loss_gains: {da: 1.0, ll: 1.0}` is **not tuned**. Measured at initialisation with the `yolo26s.pt` trunk at 384x640
  (batch 4): the L2 norm of the gradient reaching the shared neck is box 43 / cls 48 / DA 2.4 / lane 0.17, i.e. the
  trunk is shaped almost entirely by detection at the start (the lane term is small because of the 99% background
  prior and the per-pixel normalisation). The one-to-one head is detached from the trunk upstream, so the trunk's
  detection signal is the one-to-many loss, whose weight `E2ELoss` decays 0.8 -> 0.1 over the run: the detection :
  segmentation balance shifts ~8x by the end. The heads get their full gradient and all three tasks learn with the
  defaults (toy learning test), but a lane-IoU ceiling from detection-shaped P2/P3 features is the risk to ablate:
  run `loss_gains: {da: 2-5, ll: 2-10}` against the default and compare `metrics/ll_IoU_fg` / `da_mIoU` at equal mAP.
* Scale `n` only: Ultralytics' cls branch width is `max(ch[0], min(nc, 100))` = 64 for 9 classes vs 80 for COCO, so the
  whole cls branch except its first conv starts fresh (at `s` only the final 1x1 does); expect a slower cls start on `n`.
* `amp: true` runs the fp16 AMP check (needs to download `yolo26n.pt`); use `amp: bf16` on A100/H100 to skip it.
* Augmentation is ADAS-safe: `hsv_h 0.015`, `degrees 0`, no vertical flip. Mosaic keeps native object scale (`RectMosaic`).
* `multi_scale > 0` works: the packed mask is resized (nearest) together with the image.

## What is validated, and how to read it
Validation runs at exactly the deployed 384x640 (`rect=False`; Ultralytics would use per-batch rect shapes) on the **NMS-free** head.

| key | meaning |
|---|---|
| `metrics/mAP50-95(B)`, `mAP50(B)`, P, R | detection (traffic classes) |
| `metrics/da_mIoU` | drivable area: mean IoU of `direct`, `alternative` (classes without GT excluded, background excluded) |
| `metrics/da_IoU_fg`, `da_IoU_<class>` | all drivable pixels as one region (YOLOP-style), per class incl. background |
| `metrics/ll_IoU_fg`, `ll_recall_fg`, `ll_mIoU`, `ll_IoU_<class>` | lane markings: any-lane IoU, lane recall ("accuracy"), per class |
| `fitness` | `0.5 * mAP50-95 + 0.25 * da_mIoU + 0.25 * ll_IoU_fg` (weights in `mt.fitness`); selects `best.pt` and early stopping |

Segmentation metrics use non-ignored pixels only (letterbox padding and unannotated tasks never count).
**Lane IoU is measured on the training masks (~4 px wide at 640 px, the "8 px at 720p" protocol); it is not comparable with
the 2 px-test numbers in YOLOP-style papers.** Compare runs of this repository with each other.

## Multi-GPU
`--device 0,1,2,3` uses Ultralytics' DDP. Each worker builds its own teacher (rank 0 downloads first). The distillation
projector is part of the model, so it is synchronised like any other parameter. `find_unused_parameters` is on unless
`compile` is set; skipped distillation steps still run the projector, so that is safe either way.

## Resume
`python -m adas_mt train --data ... --resume runs/mt/exp/weights/last.pt` restores epoch, optimizer, EMA, the
`E2ELoss` one-to-many/one-to-one schedule and the distillation schedule/teacher. The run's own `<run>/mt.yaml` is
**authoritative** on resume (multi-task flags such as `--distill` are ignored with a warning, and the file is never rewritten):
toggling distillation or the image size would change the optimizer parameter groups and break the resume.
The run's training arguments are restored from the checkpoint too: `--cfg` is ignored on resume and only flags given
explicitly on the resume command line (`--batch`, `--workers`, `--device`, ...) replace them (Ultralytics applies
`batch`, `close_mosaic`, `patience`, `workers`, `cache`, `val`, `plots` overrides on resume without a warning, so the
CLI no longer merges `default.yaml` back in). `batch: -1` (autobatch) is refused: it profiles a square input and
cannot measure the backward pass of this model.
A clean stop strips `last.pt` (Ultralytics behaviour); only a crash/kill leaves it resumable.
Do not set `ULTRALYTICS_SAFE_LOAD=1` in the training image: it refuses to unpickle any model class that is not
Ultralytics' own, including `MultiTaskModel` checkpoints.

## Precision notes
* With `amp: true` Ultralytics validates in fp16 (also when training with `amp: bf16`): that matches the FP16 TensorRT deployment, so a
  large drop between epoch metrics and fp32 would be a deployment warning, not noise.
* The DINOv3 teacher uses bf16 on GPUs with native bf16 and fp16 otherwise (`mt.distill.teacher_dtype` overrides); non-finite teacher
  output skips the distillation step with a warning.
* `batch: -1` (autobatch) profiles a square `imgsz x imgsz` input and does not account for the teacher: set `batch` explicitly.
* `compile: true` is not tested.
* `val/kd_loss`, `val/kd_cos` are always 0 (the EMA model has no distiller); the train items are averaged over `every` steps.

## Tests
`tests/test_adas_mt/test_trainer.py`: full runs on the colour-coded toy set (distillation on, EMA/optimizer contents, SGD/MuSGD regrouping,
plots on, resume after a simulated crash, multi-scale mask resizing, DDP worker rebuild, Stage-A and official checkpoints, CLI train+val).
`ADAS_SLOW_TESTS=1` adds a learning test: from random init the nano model reaches mAP50 > 0.9 and DA / lane IoU > 0.6 on the toy set
through the real trainer and validator (about 2 minutes on CPU).
