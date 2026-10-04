# What the Phase 4 trainer must satisfy (from the Phase 3 review) - all implemented in `adas_mt/engine/train.py`
# (items 27-35: found while building the Phase 5 export / deployment path)

Each item was reproduced by running a prototype `DetectionTrainer` subclass against Ultralytics 8.4.171 or found in
its source. They are the contract between `adas_mt` and the trainer.

1. **Validation geometry.** Build the val dataset with `rect=False` (`MultiTaskDataset` raises otherwise).
2. **`kd_proj` before the optimizer.** Build the model with `kd_dim=teacher.dim` and `set_distiller_factory(...)`;
   never create the projector inside the criterion (see `docs/distillation.md`).
3. **Resume.** Ultralytics does `criterion = model.init_criterion(); criterion.updates = k; criterion.update()`.
   `MultiTaskLoss.updates` forwards to the inner `E2ELoss` and re-derives the distillation weight, and
   `init_criterion()` re-creates the distiller. (Before the fix, resuming at epoch 51/100 gave a one-to-many weight of
   0.79 instead of 0.44.)
4. **Custom hyperparameters.** `get_cfg` rejects unknown keys (and the DDP temp file re-instantiates the trainer from
   `vars(args)`), so loss gains / distillation settings must travel through the trainer constructor or
   `model.loss_gains`, never through `model.args`.
5. **Class counts.** The packed mask is decoded with the model's class counts: call
   `check_matches_data(model, data)` once at start-up.
6. **`multi_scale`.** Upstream `preprocess_batch` resizes only `img`; the trainer also resizes `semantic_mask`
   (nearest), so `multi_scale > 0` works (tested).
7. **Optimizer.** `optimizer=auto` selects MuSGD for runs over 10k iterations and boosts the learning rate x3 only for
   heads it recognises by name (`cv3`, `SemanticSegment`). `da_head`, `ll_head` and `kd_proj` are new modules and need
   the same boost; set the optimizer explicitly instead of relying on `auto` (which also overrides `lr0`).
8. **DDP.** 8.4.171 wraps with `find_unused_parameters=True`, which hides unused parameters (aux head, `every>1`); that
   does not hold with `compile=True`, which is why skipped distillation steps still run the projector.
9. **Interpolation / preprocessing parity.** Train and val both load with `INTER_LINEAR` in this Ultralytics version.
   The Jetson runner must use RGB, `/255`, linear resize, centred letterbox with value 114 (masks 255), and the same
   384x640 geometry.
10. **ONNX.** Export with `torch.onnx.export(..., dynamo=False)` (the dynamo path needs `onnxscript`). The only
    TensorRT-unverified op is `DepthToSpace(mode=CRD)` from the lane PixelShuffle: build an engine early.
    Prefer an in-graph ArgMax to uint8 for both masks, and calibrate INT8 with the deploy preprocessing (the lane
    classifier carries a +5.3 background bias).

## Found while writing the trainer
11. **`nms` default.** Ultralytics validates the NMS-free head only with `nms=False`; the default `None` validates the
    one-to-many head with NMS. The trainer defaults `nms=False` and warns otherwise.
12. **DDP hand-off.** `generate_ddp_command` deletes the run directory before spawning workers, so the `mt` settings are
    handed over through a file outside it (`ADAS_MT_CFG`); workers re-write `<run>/mt.yaml` themselves.
13. **Stripped checkpoints.** After `final_eval` the EMA weights are in `ckpt["model"]` (`"ema"` is None) and `last.pt` has
    epoch -1: a clean stop cannot be resumed, only a crash can.
14. **Standalone validation** wraps the model in `AutoBackend`, which exposes none of the model's attributes: the
    validator reads class counts from the underlying model or `data.yaml`.
15. **`pretrained=False`** discards weights even when `model` is a `.pt`; the default (`True`) keeps them.

## Found by the independent review of the trainer (all fixed, each with a test)
16. **CLI `--resume`** passed `default.yaml`'s multi-task config, which beat the run's own `mt.yaml` (distillation off), crashed the
    optimizer-state load and overwrote `mt.yaml`. The run's `mt.yaml` is now authoritative on resume and is never rewritten.
17. **`--amp true|false` / `--cache true`** were passed as strings (Ultralytics rejects the first, silently ignores the second).
18. **Standalone `val`** used default.yaml's imgsz instead of the run's (mAP50-95 0.46 -> 0.0 with no warning) and never checked
    the class counts against `data.yaml`; it now reads `<run>/mt.yaml` and raises on a mismatch.
19. **`results.csv`** got a short header when validation was skipped in early epochs and long rows later; the segmentation columns are
    now pre-seeded. (Extending `DetMetrics.keys` was tried and rejected: Ultralytics zips it with four values and sizes the console table from it.)
20. **Tests never stepped the optimizer** (`nbs=64` with batch 4 accumulates 16 batches); they now use `nbs=4` and assert EMA updates.
21. **Output paths:** a relative `project` landed in `runs/detect/<project>`; the default is now an absolute `<cwd>/runs/mt/exp`
    and standalone `val` no longer creates an empty `runs/detect/train`.
22. **`imgsz: 640`** in default.yaml stayed in `args.imgsz` (multi_scale range, autobatch); it now always follows `mt.imgsz`.
23. The **DDP hand-off** ran in every spawned worker; only the launching process does it now (`self.ddp`).
24. **Group names:** the boosted groups had been renamed (`bias_new`) and missed the stock bias warmup and weight-decay rescale; they keep
    the stock names and carry `new_head=True`. default.yaml sets `warmup_bias_lr: 0.0` for the explicit AdamW (what `optimizer=auto` forces).
25. **Config validation** is strict (types, positive multiples of 32, unknown `fitness`/`loss_gains` keys, `head_lr_mult > 0`, stale `ADAS_MT_CFG`,
    unknown top-level YAML sections); the effective model scale is recorded in `mt.yaml`.
26. **Teacher safety:** `mt.distill.teacher_dtype` is configurable and a non-finite teacher output skips the step (with a warning) instead of poisoning the student.

## Found while building export and deployment (Phase 5; all fixed, each with a test)
27. **Random-init networks are degenerate in eval mode.** Activations vanish through ~100 layers, so every output is a bias and every
    anchor ties: parity tests compared arbitrary top-k picks. Tests use `nontrivial_model()` (BatchNorm running stats calibrated in train
    mode, raised class biases) and a spatially smooth input; a real trained model confirms it (`.pt` and ONNX metrics identical).
28. **Top-k ties.** Detection parity compares the sorted score vector and the confident rows strictly above the top-k cutoff; row order
    and the pick among exactly tied scores are runtime-dependent.
29. **Rounding.** cv2 rounds half up, `torch.round` rounds half to even: the GPU letterbox differed from the validation pipeline on 12% of
    the pixels after a 2x downscale (exact .5 averages). It now uses `floor(x + 0.5)`.
30. **Dynamic-batch ONNX** carried stale symbolic output dims (the non-simplified graph even declared `[batch, batch, W]`); input and
    output shapes are rewritten from the known geometry after export.
31. **Output aliasing.** `TrtBackend.infer` returned views of its persistent output buffers; on a CPU device `.cpu().numpy()` is a no-op,
    so the second batch overwrote the first one's results before they were concatenated (found by the fake-engine evaluation test; on CUDA the
    D2H copy hid it). Outputs are now copied explicitly.
32. **INT8.** A calibration cache is only reused when requested (scales are keyed by tensor names, so a stale cache silently gives a wrong
    engine); FP16 pinning skips layers with integer outputs (TopK indices, Shape, Gather) and constants, where a float type is invalid.
33. **TensorRT version differences** handled in the builder: `EXPLICIT_BATCH` before 10 and not after, `set_memory_pool_limit` instead of
    `max_workspace_size`, uint8 network outputs only on TensorRT >= 10 (rejected early on 8.6).
34. **No ultralytics/torch at import time** in `deploy.runner` / `deploy.trt_build` / `deploy.meta` (tested), so the Jetson needs only the engine,
    the sidecar JSON and torch for CUDA buffers.
35. **Still unverified (needs the Orin):** TensorRT acceptance of `DepthToSpace(CRD)` (`--decompose-pixel-shuffle` is the tested fallback), real
    FP16/INT8 accuracy and latency, GStreamer camera input.
