# What the Phase 4 trainer must satisfy (from the Phase 3 review)

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
6. **`multi_scale`.** Upstream `preprocess_batch` resizes only `img`; with `multi_scale > 0` the trainer must also resize
   `semantic_mask` (nearest). Keep `multi_scale=0` until that is implemented.
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
