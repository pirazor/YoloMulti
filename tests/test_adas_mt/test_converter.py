import cv2
import yaml

from adas_mt.data.convert_supervisely import convert


def _find(dst, task, name):
    """The converter always keeps >=1 val image, so look in both splits."""
    for split in ("train", "val"):
        p = dst / f"labels_{task}" / split / f"{name}.png"
        if p.exists():
            return p
    return None


def _ll(dst, name):
    return cv2.imread(str(_find(dst, "ll", name)), cv2.IMREAD_GRAYSCALE)


def test_default_writes_all_masks_and_thick_lanes(tmp_path, tiny_supervisely):
    dst = tmp_path / "out"
    convert(src=tiny_supervisely, dst=dst, val_fraction=0.5)
    for n in ("full", "car_only"):
        assert _find(dst, "da", n) is not None and _find(dst, "ll", n) is not None
    # vertical dashed line of length 400px, 8px wide at 720p
    assert abs(int((_ll(dst, "full") == 2).sum()) - 400 * 8) < 400 * 3
    assert yaml.safe_load((dst / "data.yaml").read_text())["ll_names"] == ["background", "solid", "dashed"]


def test_partial_annotation_skips_unannotated_tasks(tmp_path, tiny_supervisely):
    dst = tmp_path / "out"
    convert(src=tiny_supervisely, dst=dst, val_fraction=0.5, partial_annotation=True)
    assert _find(dst, "da", "full") is not None and _find(dst, "ll", "full") is not None
    assert _find(dst, "da", "car_only") is None and _find(dst, "ll", "car_only") is None


def test_lane_thickness_option(tmp_path, tiny_supervisely):
    dst = tmp_path / "out"
    convert(src=tiny_supervisely, dst=dst, val_fraction=0.5, lane_thickness=4)
    assert abs(int((_ll(dst, "full") == 2).sum()) - 400 * 4) < 400 * 2


def test_assign_val_zero_means_no_val():
    from adas_mt.data.convert_supervisely import assign_val

    assert assign_val(["a", "b", "c"], 0.0) == set()
    assert len(assign_val([f"x{i}" for i in range(100)], 0.1)) == 10
    assert len(assign_val(["only"], 0.5)) == 0  # a single image stays in train


def test_assign_val_never_splits_a_clip():
    from adas_mt.data.convert_supervisely import assign_val

    names = [f"{clip}-{f:03d}" for clip in ("aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd", "eeeeeeee") for f in range(10)]
    for seed in range(8):
        val = assign_val(names, 0.2, seed=seed, group_regex=r"^([0-9a-z]{8})-")
        val_clips = {names[i][:8] for i in val}
        train_clips = {names[i][:8] for i in range(len(names)) if i not in val}
        assert val and not (val_clips & train_clips), (seed, val_clips, train_clips)


def test_convert_without_val_points_yaml_at_train(tmp_path, tiny_supervisely):
    dst = tmp_path / "out"
    convert(src=tiny_supervisely, dst=dst, val_fraction=0.0)
    data = yaml.safe_load((dst / "data.yaml").read_text())
    assert data["val"] == "images/train"
    assert not list((dst / "images" / "val").glob("*.jpg"))
    assert len(list((dst / "images" / "train").glob("*.jpg"))) == 2


def test_rerun_into_same_dst_is_refused_or_cleaned(tmp_path, tiny_supervisely):
    import pytest

    dst = tmp_path / "out"
    convert(src=tiny_supervisely, dst=dst, val_fraction=0.5, seed=0)
    with pytest.raises(FileExistsError):
        convert(src=tiny_supervisely, dst=dst, val_fraction=0.5, seed=1)  # would leave images in train AND val
    convert(src=tiny_supervisely, dst=dst, val_fraction=0.5, seed=1, overwrite=True)
    train = {p.name for p in (dst / "images" / "train").glob("*.jpg")}
    val = {p.name for p in (dst / "images" / "val").glob("*.jpg")}
    assert train and val and not (train & val)


def _write(root, name, objects, size=(720, 1280)):
    import json

    import numpy as np

    (root / "img").mkdir(parents=True, exist_ok=True)
    (root / "ann").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "img" / f"{name}.jpg"), np.full((*size, 3), 90, np.uint8))
    (root / "ann" / f"{name}.json").write_text(json.dumps({"size": {"height": size[0], "width": size[1]}, "objects": objects}))


def test_same_stem_in_two_source_datasets_does_not_overwrite(tmp_path):
    for ds in ("a", "b"):
        _write(tmp_path / ds, "frame_0001", [])
    stats = convert(src=tmp_path, dst=tmp_path / "out", val_fraction=0.0)
    assert stats.images == 2 and stats.renamed_stems == 1
    assert len(list((tmp_path / "out" / "images" / "train").glob("*.jpg"))) == 2


def test_polygon_holes_and_unmapped_attributes(tmp_path):
    hole_poly = {"classTitle": "drivable area", "geometryType": "polygon",
                 "points": {"exterior": [[100, 100], [500, 100], [500, 500], [100, 500]],
                            "interior": [[[200, 200], [400, 200], [400, 400], [200, 400]]]},
                 "tags": [{"name": "areaType", "value": "{'areaType': 'direct'}"}]}
    weird = {"classTitle": "drivable area", "geometryType": "polygon",
             "points": {"exterior": [[10, 10], [60, 10], [60, 60], [10, 60]]},
             "tags": [{"name": "areaType", "value": "{'areaType': 'mystery'}"}]}
    _write(tmp_path / "ds", "holes", [hole_poly])
    _write(tmp_path / "ds", "weird", [weird])
    dst = tmp_path / "out"
    stats = convert(src=tmp_path / "ds", dst=dst, val_fraction=0.0, partial_annotation=True)
    da = cv2.imread(str(_find(dst, "da", "holes")), cv2.IMREAD_GRAYSCALE)
    assert da[300, 300] == 0 and da[150, 150] == 1  # the hole is NOT drivable
    assert stats.skipped_unmapped == 1
    assert _find(dst, "da", "weird") is None  # unmapped value must not become an annotated all-background mask


def _bitmap_object(title, tags, origin, shape, compress=True):
    """A Supervisely 'bitmap' object: base64 of a (zlib-compressed) PNG placed at origin [x, y]."""
    import base64
    import zlib

    import numpy as np

    bm = np.zeros(shape, np.uint8)
    bm[2:-2, 2:-2] = 255
    ok, png = cv2.imencode(".png", bm)
    data = png.tobytes()
    return {"classTitle": title, "geometryType": "bitmap", "tags": tags,
            "bitmap": {"data": base64.b64encode(zlib.compress(data) if compress else data).decode(), "origin": list(origin)}}


def test_bitmap_geometry_and_unknown_geometry_are_handled(tmp_path):
    da_tag = [{"name": "areaType", "value": "{'areaType': 'alternative'}"}]
    lane_tag = [{"name": "laneAttrs", "value": "{'laneStyle': 'solid', 'laneType': 'single white'}"}]
    objs = [_bitmap_object("drivable area", da_tag, (100, 200), (50, 80)),
            _bitmap_object("lane", lane_tag, (1250, 700), (40, 60), compress=False),  # partly outside the image
            {"classTitle": "car", "geometryType": "polygon", "points": {"exterior": [[0, 0], [10, 0], [10, 10]]}, "tags": []},
            {"classTitle": "drivable area", "geometryType": "point", "points": {"exterior": [[5, 5]]}, "tags": da_tag}]
    _write(tmp_path / "ds", "bm", objs)
    dst = tmp_path / "out"
    stats = convert(src=tmp_path / "ds", dst=dst, val_fraction=0.0)
    da = cv2.imread(str(_find(dst, "da", "bm")), cv2.IMREAD_GRAYSCALE)
    ll = _ll(dst, "bm")
    # origin is [x, y]: the 50x80 bitmap covers rows 200..250, cols 100..180, with a 2 px empty rim
    assert da[200 + 2, 100 + 2] == 2 and da[200 + 47, 100 + 77] == 2 and da[200, 100] == 0 and da[200 + 25, 100 + 90] == 0
    assert int((da == 2).sum()) == 46 * 76
    assert ll[702, 1252] == 1 and int((ll == 1).sum()) == (720 - 702) * (1280 - 1252)  # clipped at the border
    assert stats.da_polys == 1 and stats.ll_polys == 1
    assert dict(stats.skipped_geometry) == {"car/polygon": 1, "drivable area/point": 1}


def test_annotation_size_mismatch_scales_coordinates(tmp_path):
    """The JSON says 640x360 but the image file is 1280x720 (resized after annotation): scale, count, warn."""
    import json

    import numpy as np

    car = {"classTitle": "car", "geometryType": "rectangle", "points": {"exterior": [[100, 50], [200, 100]]}, "tags": []}
    da = {"classTitle": "drivable area", "geometryType": "polygon", "points": {"exterior": [[0, 180], [640, 180], [640, 360], [0, 360]]},
          "tags": [{"name": "areaType", "value": "{'areaType': 'direct'}"}]}
    root = tmp_path / "ds"
    (root / "img").mkdir(parents=True)
    (root / "ann").mkdir(parents=True)
    cv2.imwrite(str(root / "img" / "f.jpg"), np.full((720, 1280, 3), 90, np.uint8))
    (root / "ann" / "f.json").write_text(json.dumps({"size": {"height": 360, "width": 640}, "objects": [car, da]}))
    dst = tmp_path / "out"
    stats = convert(src=root, dst=dst, val_fraction=0.0)
    assert stats.size_mismatch == 1
    line = next((dst / "labels_det" / "train").glob("*.txt")).read_text().split()
    assert [round(float(v), 4) for v in line[1:]] == [round(300 / 1280, 4), round(150 / 720, 4), round(200 / 1280, 4), round(100 / 720, 4)]
    m = cv2.imread(str(_find(dst, "da", "f")), cv2.IMREAD_GRAYSCALE)
    assert m.shape == (720, 1280) and m[359, 640] == 0 and m[361, 640] == 1 and m[719, 1279] == 1


def test_lane_thickness_scales_with_the_long_side():
    from adas_mt.data.convert_supervisely import _line_thickness

    assert _line_thickness((720, 1280)) == 8 and _line_thickness((1280, 720)) == 8  # 16:9 and portrait 16:9
    assert _line_thickness((1080, 1920)) == 12 and _line_thickness((480, 640)) == 4  # 4:3 used to get 5 (by height)
    assert _line_thickness((60, 100)) == 2
