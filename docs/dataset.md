# Dataset format & conversion

## Input: Supervisely-style annotations

Each image is paired with a JSON file. The JSON has:

```json
{
  "size": {"height": 720, "width": 1280},
  "objects": [
    {
      "classTitle": "car",
      "geometryType": "rectangle",
      "points": {"exterior": [[x1, y1], [x2, y2]]},
      "tags": [{"name": "...", "value": "{'k': 'v'}"}]
    },
    ...
  ]
}
```

`tags[].value` is a **stringified Python dict** with single quotes — it is *not*
valid JSON. The converter parses it with `ast.literal_eval`.

### Recognised classTitles

| classTitle in JSON | role                          |
|--------------------|-------------------------------|
| `car`              | detection class `car`         |
| `truck`            | detection class `truck`       |
| `bus`              | detection class `bus`         |
| `motorcycle`       | detection class `motorcycle`  |
| `bike`,`bicycle`   | detection class `bicycle`     |
| `person`,`pedestrian` | detection class `pedestrian` |
| `rider`            | detection class `rider`       |
| `traffic light`    | detection class `traffic_light` (or split by color, see below) |
| `traffic sign`     | detection class `traffic_sign` |
| `drivable area`    | rasterized into the DA mask, value=1 (direct) or 2 (alternative) |
| `lane`             | rasterized into the LL mask, class id by `--lane_grouping` |

Anything else is silently dropped (with a counter logged at the end).

## Output: YOLO-format multi-task dataset

```
dataset_yolo/
  images/{train,val}/<stem>.jpg                # original images (transcoded to JPG)
  labels_det/{train,val}/<stem>.txt            # YOLO box labels
  labels_da/{train,val}/<stem>.png             # uint8 mask, 0=bg, 1=direct, 2=alternative
  labels_ll/{train,val}/<stem>.png             # uint8 mask, 0=bg, 1..N=lane classes
  data.yaml
  lane_classes.json     # only when --lane_grouping=type
```

Detection labels follow the standard YOLO format: one line per box,
`class_id cx cy w h`, normalized to [0, 1].

DA and LL labels are full-resolution **uint8 PNGs**; pixel values are class
ids (0=background). Loaded with `cv2.IMREAD_GRAYSCALE` and cast to `int64`
in the dataset.

## Lane grouping options

- `--lane_grouping=style` (default, recommended). Class ids:
  - 0 = background
  - 1 = solid
  - 2 = dashed
- `--lane_grouping=type`. One class per `laneType` value seen in the dataset
  (crosswalk, road curb, single white, single yellow, double white, double
  yellow, ...). Mapping is written to `dataset_yolo/lane_classes.json`.

The model derives its head's output channel count from `data.yaml` at
construction time, so switching modes only requires a re-conversion + a fresh
training run.

## Traffic-light color split

`--split_tl_by_color` expands the single `traffic_light` class into three:
`traffic_light_red, traffic_light_yellow, traffic_light_green`. Lights with
unknown/missing color are dropped.

## Train/val split

If the Supervisely directory layout already has train/val subdirectories
(any path part named `train`, `val`, `valid`, `validation`, `test`), the
converter honors that. Otherwise it does a deterministic 90/10 split with
seed 0.

## Fine-grained sign workflow

The BDD-style `traffic sign` class is **coarse** (one class). For fine-grained
sign labels, use `sign_classifier/`:

1. Run `python -m sign_classifier extract` to dump padded square crops of
   every sign box.
2. Manually label the crops (or import labels from GTSRB/Mapillary) into
   `train/<class>/` and `val/<class>/` directories.
3. Train a sign classifier on the crops.
4. At inference, the multi-task detector emits `traffic_sign` boxes; pipe
   them into `SignClassifier.classify()` for fine-grained labels.

See [sign_classifier/README.md](../sign_classifier/README.md) for details.
