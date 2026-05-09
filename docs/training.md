# Training notes

## Default hyperparameters (`yolov13_multitask/cfg/multitask.yaml`)

```yaml
epochs: 200
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01
weight_decay: 0.0005
warmup_epochs: 3

box: 7.5      # det internal gain (passed straight to v8DetectionLoss)
cls: 0.5
dfl: 1.5

loss_weights: { det: 1.0, da: 1.0, ll: 1.0 }
use_uncertainty_weighting: false
freeze_backbone_epochs: 0
```

## Loss balancing

The internal det gains (`box, cls, dfl`) are inherited from v8 and shouldn't
need adjustment. The top-level `loss_weights.{det,da,ll}` are the dial.

Typical pathologies:

- **Lane loss vanishing** (drops to ~zero quickly): the network has learned
  "predict all background" and Tversky's recall trade-off is tipping toward
  precision. Increase `loss_weights.ll` to ~2.0, or lower `alpha` (0.3 → 0.2)
  in `FocalTverskyLoss` to penalise FN more.
- **Detection mAP regresses** after a few epochs: drivable-area loss is
  dominating. Either drop `loss_weights.da` to 0.5, or enable
  `use_uncertainty_weighting`.
- **NaN in loss**: usually means the LR is too high for AMP. Halve `lr0` and
  retry. The loss already clips gradients (`grad_norm <= 10`).

If you can't pick weights, set `use_uncertainty_weighting: true` and the
model learns σ_det, σ_da, σ_ll. Inspect them in the logs over time to see
which task it considers "noisier".

## Freeze schedule

`freeze_backbone_epochs > 0` freezes everything except the seg decoders
during warmup. Useful when you load a pretrained `yolov13s.pt`: the detect
head + backbone is already strong, and you let the brand-new seg decoders
catch up before joint training. Default is 0 because BDD-scale data lets
the joint training settle on its own; bump to 3 if you see detection mAP
dropping early.

## Augmentation alignment

The Albumentations pipeline keeps image, boxes, da_mask, ll_mask in lockstep.
**Always** run the visual sanity test (`pytest tests/multitask`) and open
the resulting overlays under `tests/_out/multitask/aug/`. What to look for:

- Box edges should kiss the cars / signs in the image.
- The drivable-area overlay should land on the road, not the sky / hood.
- Lane lines in `c` (third panel) should overlap with the actual lane
  markings in the image.
- After mosaic, all four quadrants should still satisfy the above.

If anything is off:

1. Comment out the post-mosaic Affine in `transforms.py` and re-run. If it
   now aligns, the issue is mask interpolation (should be NEAREST).
2. If still off, set `mosaic_prob=0` and re-run; the issue is in the mosaic
   geometry (`data/mosaic.py`).
3. If still off, the converter is the suspect — re-convert with `--limit 5`
   and check the raw `labels_da/*.png` and `labels_ll/*.png` directly.

## Common failure modes

| symptom                                               | likely cause                     | fix |
|-------------------------------------------------------|----------------------------------|------|
| validation prints `mIoU_da=0.0` from epoch 1         | DA mask not loaded               | check `data.yaml: da_classes`, look for `labels_da/<split>/` files |
| `ll_mask=2` only when GT is `solid`                  | lane grouping mismatch           | re-convert with `--lane_grouping=style` (or `type`) consistently |
| training stalls at `0.000` lane loss                 | empty masks                      | sanity-check that `labels_ll/train/*.png` has nonzero pixels |
| ONNX export verify warns max-abs-diff > 1e-3         | bilinear upsample non-determinism | use `--opset 17` and `--dynamic` to limit constant-folding |
| OOM at imgsz=640 batch=16 on a 12GB GPU              | full-res 640×640 logits          | drop `decoder_mid_ch: 64` in cfg, or use mixed-precision (`amp: true` is default) |

## Reproducibility

`seed: 0` is set on the model, dataset, and augmentation transforms. AMP
introduces nondeterminism in some kernels; if you need bit-exactness, set
`amp: false` and `torch.use_deterministic_algorithms(True)` before training.
