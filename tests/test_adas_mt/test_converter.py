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
