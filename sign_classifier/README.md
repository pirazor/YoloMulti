# Fine-grained traffic-sign classifier

A lightweight second-stage classifier that takes traffic-sign boxes from
`yolov13_multitask` and assigns a fine-grained class (e.g. *stop*, *yield*,
*30 km/h*).

## Why a separate stage?

The BDD/Supervisely export this repo targets only labels signs as a single
`traffic sign` class. Fine-grained labels must come from a different source
(e.g. **GTSRB**, **Mapillary Traffic Sign**) **or** from manual labeling of
crops you extract from your own dataset.

## Workflow

1. Extract crops from your Supervisely dataset:
   ```bash
   python -m sign_classifier extract \
     --src /path/to/supervisely \
     --dst /path/to/sign_crops \
     --padding 0.15 \
     --negative_per_image 4
   ```
   This writes:
   - `sign_crops/crops/*.jpg`              — square padded crops of every sign box
   - `sign_crops/manifest.csv`             — bbox + source image for each crop
   - `sign_crops/unknown/*.jpg`            — random hard-negative patches (only if
     `--negative_per_image` > 0)

2. **Manually label the crops.** Move each crop into the corresponding class
   directory; the final layout must be ImageFolder-style:
   ```
   sign_crops/
     train/
       stop/
       yield/
       speed_30/
       ...
       unknown/   # use the auto-generated negatives + any "not a sign" crops
     val/
       stop/
       ...
   ```
   You can use any tooling for this (Label Studio, CVAT, an Excel + a sort
   script). For a head start, label a few hundred per class and bootstrap the
   rest with active learning using a quickly-trained model.

3. Train:
   ```bash
   python -m sign_classifier train --data sign_crops --epochs 30
   ```

4. Inference (given detector boxes):
   ```python
   from sign_classifier.infer import SignClassifier
   clf = SignClassifier("runs/sign_classifier/exp/best.pt")
   labels = clf.classify(bgr_image, boxes_xyxy)
   ```

## Notes
- Models default to `mobilenetv3_small_100` from `timm`. Switch to
  `efficientnet_b0` (larger) via `--model efficientnet_b0`.
- The augmentations (rotation ±15°, perspective, color jitter, motion blur,
  Gaussian noise) target the typical failure modes of car-mounted cameras.
- `unknown` is treated as a normal class. At inference, you can threshold its
  probability to suppress confident-but-wrong sign labels.
