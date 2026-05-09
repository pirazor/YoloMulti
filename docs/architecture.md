# Architecture

## Backbone + neck

Untouched from upstream `ultralytics/cfg/models/v13/yolov13.yaml`. The relevant
indices captured by the multi-task model are:

| layer index | role  | nominal channels (scale `s`, width 0.5) |
|-------------|-------|------------------------------------------|
| 23          | P3    | 256 × 0.5 = 128                          |
| 27          | P4    | 512 × 0.5 = 256                          |
| 31          | P5    | 1024 × 0.5 = 512                         |

`YOLOv13MultiTask._infer_fpn_channels()` runs a one-time dry forward over the
already-built backbone+neck to read the actual channel counts off the layer
outputs, so changing the scale (`n/s/l/x`) Just Works.

## Segmentation decoder

`yolov13_multitask/nn/seg_head.py:FPNFusedSegDecoder`:

```
P3 (B, C3, H/8, W/8)  ─── 1x1  ──┐
P4 (B, C4, H/16, W/16)─── 1x1  ──┼── concat ─── 3x3 fuse ──┐
P5 (B, C5, H/32, W/32)─── 1x1  ──┘                          │
                                                            ▼
                                              up2x ─ Conv 3x3 (mid -> mid/2)  // H/4
                                              up2x ─ Conv 3x3 (mid/2 -> mid/2) // H/2
                                              up2x ─ Conv 3x3 (mid/2 -> mid/4) // H
                                                       │
                                                    1x1 conv -> num_classes (logits)
```

Two independent instances — one for drivable area (3 classes), one for lane
(3 by default). Logits are produced at the **input resolution** (640×640),
matching the GT mask resolution exactly.

## Detection head

`ultralytics.nn.modules.head.Detect`, untouched. The multi-task model wraps
`_predict_once` so we can:

1. Run the entire backbone+neck+Detect graph normally.
2. After the loop, look up `y[23], y[27], y[31]` (saved by `parse_model`)
   and feed them into the two seg decoders.

This is cheap (one extra dict lookup per FPN level) and keeps the upstream
Detect contract identical.

## Loss

`yolov13_multitask/loss/multitask_loss.py:MultiTaskLoss`:

```
L_total = w_det * L_det + w_da * L_da + w_ll * L_ll

L_det  = upstream v8DetectionLoss (already includes box/cls/dfl gains)
L_da   = mean over has_da samples of  CE + Dice(C=3, ignore_bg=True)
L_ll   = mean over has_ll samples of  CE + FocalTversky(C=ll_classes,
                                       alpha=0.3, beta=0.7, gamma=0.75,
                                       ignore_bg=True)
```

### Why CE + Dice for drivable area, CE + FocalTversky for lane?

- Drivable area is a large, contiguous region. CE pushes pixel-wise accuracy;
  Dice protects boundaries from being washed out.
- Lanes are thin and very class-imbalanced. Dice alone overshoots toward
  recall; FocalTversky with `α<β` and the focal exponent γ smoothly trades
  precision for recall while suppressing easy-pixel gradient. With γ=0.75,
  badly-segmented lane pixels weigh more during training.

### Optional Kendall uncertainty weighting

Set `use_uncertainty_weighting: true` in the cfg. The loss adds
3 learnable parameters σ_det, σ_da, σ_ll and uses

    L = Σ_i 0.5 · exp(−2σ_i) · L_i + σ_i

Let the model learn its own weights instead of hand-tuning. Useful if one
task drowns out the others in early training.

## Sample-level mask masking

Some samples may not have one or both segmentation labels (e.g. when you mix
BDD with another dataset that lacks lane annotations). The dataloader returns
`has_da` / `has_ll` booleans per sample, and the loss zeroes out segmentation
contribution for samples where the mask isn't real. This keeps gradients
clean.
