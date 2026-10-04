# Data pipeline (Phase 1)

## Layout (written by `python -m adas_mt.data.convert_supervisely`)
```
root/images/{train,val}/*.jpg
root/labels_det/{train,val}/*.txt      YOLO boxes
root/labels_da/{train,val}/*.png       uint8: 0=bg, 1=direct, 2=alternative
root/labels_ll/{train,val}/*.png       uint8: 0=bg, 1=solid, 2=dashed (or one id per laneType)
root/data.yaml                         nc, names, da_classes, da_names, ll_classes, ll_names
```
**A missing DA/LL PNG means "not annotated", not "background".** Use `--partial_annotation`
when some images were annotated for only one task; the loss then ignores that task for them.
`--lane_thickness` is the lane line width in px at 720p (default 8, scaled with image height).

## Converter safety
- Re-running into a folder that already holds a converted dataset raises unless `--overwrite` (a second run with a
  different seed would otherwise leave images in both train and val, plus stale masks).
- Two source files with the same stem no longer overwrite each other: the later one is renamed `<stem>__<hash>`.
- Supervisely polygons with `interior` rings (holes) are drawn with the holes left empty.
- A drivable-area / lane object with an unmapped attribute value is counted (`skipped_unmapped`) and does not make
  the task "annotated" for that image, so `--partial_annotation` ignores it instead of training all-background.
- A mask PNG that exists but cannot be read raises at load time (it used to become "unlabelled" silently).
- `MultiTaskDataset` rejects `rect=True`: Ultralytics' `DetectionTrainer.build_dataset` passes it for validation,
  which would letterbox to per-batch shapes (e.g. 384x672) instead of the deployed 384x640.

## Train/val split
If the Supervisely dump has no `train/` / `val/` folders, the converter splits itself.
- `--val_fraction 0` gives no val images (`data.yaml` then validates on `images/train`, with a warning).
- A per-image random split leaks: neighbouring frames of one clip land in both sets and inflate
  val metrics. Pass `--group_regex` (first group = clip id, e.g. `'^([0-9a-f]{8})-'` for BDD-style
  names) to split whole clips.

## One packed mask, upstream augmentation
Ultralytics 8.4 already warps a single `semantic_mask` together with image and boxes (Mosaic,
RandomPerspective, RandomFlip, LetterBox: nearest interpolation, 255 = ignore padding). We pack
both task maps into it (`adas_mt/data/masks.py`):

    packed = da_code + (da_classes + 1) * ll_code      code == classes  ->  task unlabelled
    packed == 255                                      padding: ignore both tasks

`unpack_masks(packed, da_classes, ll_classes)` returns `(da, ll)` with 255 wherever a task is
ignored; the Phase 4 loss uses `ignore_index=255`.

## Rectangular input
`MultiTaskDataset(imgsz=(h, w))` (default 384x640). Images load with long side = max(h, w).
Training: `RectMosaic` (2h x 2w canvas) -> `RandomPerspective(size=(w, h))` crops the central
window at native scale -> HSV -> flip. Validation: centred `LetterBox` with 255 mask padding.
Use `build(..., hsv_h<=0.015, degrees<=3)` for ADAS-safe augmentation (keeps traffic-light colours).

## Tests
`pytest tests/test_adas_mt` builds a synthetic set whose red/green rectangles are simultaneously
a box, an image region and a mask region, and checks that all three stay aligned through
mosaic / affine / flip, that mosaic keeps native scale, and that padding is ignored.
