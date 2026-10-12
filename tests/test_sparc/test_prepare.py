"""prepare: archives in a datasets folder -> unpacked, converted, cached, manifests; viz helpers render."""

import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from sparc.tools import prepare as P

from .converters.test_bdd100k import _det_labels, _drv_labels, _img, _lane_labels


def _bdd_download(tmp: Path, folder: Path):
    """The four official BDD100K zips (images, det_20, drivable, lane), each rooted at ``bdd100k/``."""
    src = tmp / "bdd_src" / "bdd100k"
    for split, names in (("train", ["a1", "a2"]), ("val", ["v1"])):
        for n in names:
            _img(src / "images" / "100k" / split / f"{n}.jpg")
        (src / "labels" / "det_20").mkdir(parents=True, exist_ok=True)
        (src / "labels" / "det_20" / f"det_{split}.json").write_text(json.dumps(
            [{"name": f"{n}.jpg", "labels": _det_labels()} for n in names]))
        (src / "labels" / "lane" / "polygons").mkdir(parents=True, exist_ok=True)
        (src / "labels" / "lane" / "polygons" / f"lane_{split}.json").write_text(json.dumps(
            [{"name": f"{n}.jpg", "labels": _lane_labels(True)} for n in names]))
        (src / "labels" / "drivable" / "polygons").mkdir(parents=True, exist_ok=True)
        (src / "labels" / "drivable" / "polygons" / f"drivable_{split}.json").write_text(json.dumps(
            [{"name": f"{n}.jpg", "labels": _drv_labels(True)} for n in names]))
    folder.mkdir(parents=True)
    for zname, sub in (("bdd100k_images_100k.zip", "images"), ("bdd100k_det_20_labels_trainval.zip", "labels/det_20"),
                       ("bdd100k_drivable_labels_trainval.zip", "labels/drivable"), ("bdd100k_lane_labels_trainval.zip", "labels/lane")):
        with zipfile.ZipFile(folder / zname, "w") as z:
            for f in (src / sub).rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(src.parent))
    # split the images zip into byte parts like a multi-part download
    data = (folder / "bdd100k_images_100k.zip").read_bytes()
    (folder / "bdd100k_images_100k.zip").unlink()
    half = len(data) // 2
    (folder / "bdd100k_images_100k.zip.001").write_bytes(data[:half])
    (folder / "bdd100k_images_100k.zip.002").write_bytes(data[half:])


def _culane_download(tmp: Path, folder: Path):
    """CULane: a driver tar.gz with images + .lines.txt, list.tar.gz, and annotations_new.tar.gz overriding one lines file."""
    import cv2

    src = tmp / "cu_src"
    rel = "driver_23_30frame/05151649_0422.MP4"
    (src / rel).mkdir(parents=True)
    lines = {}
    for i in range(3):
        cv2.imwrite(str(src / rel / f"{i:05d}.jpg"), np.full((590, 1640, 3), 80, np.uint8))
        lines[i] = "500 589 560 400 600 300\n1100 589 1040 400 1000 300\n"
        (src / rel / f"{i:05d}.lines.txt").write_text(lines[i])
    (src / "list").mkdir()
    imgs = [f"/{rel}/{i:05d}.jpg" for i in range(3)]
    (src / "list" / "train.txt").write_text("\n".join(imgs[:2]) + "\n")
    (src / "list" / "val.txt").write_text(imgs[2] + "\n")
    (src / "list" / "train_gt.txt").write_text("".join(f"{p} /laneseg{p[:-4]}.png 0 1 1 0\n" for p in imgs[:2]))
    (src / "list" / "val_gt.txt").write_text(f"{imgs[2]} /laneseg{imgs[2][:-4]}.png 0 1 1 0\n")
    folder.mkdir(parents=True)
    with tarfile.open(folder / "driver_23_30frame.tar.gz", "w:gz") as t:
        t.add(src / "driver_23_30frame", "driver_23_30frame")
    with tarfile.open(folder / "list.tar.gz", "w:gz") as t:
        t.add(src / "list", "list")
    upd = tmp / "cu_upd" / rel
    upd.mkdir(parents=True)
    (upd / "00000.lines.txt").write_text("400 589 460 400\n1100 589 1040 400 1000 300\n")  # corrected first lane
    with tarfile.open(folder / "annotations_new.tar.gz", "w:gz") as t:
        t.add(tmp / "cu_upd" / "driver_23_30frame", "driver_23_30frame")


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dl")
    ds = tmp / "datasets"
    _bdd_download(tmp, ds / "bdd100k")
    _culane_download(tmp, ds / "culane")
    (ds / "notes").mkdir()  # an unrelated folder is ignored
    return ds


def test_discover_groups_split_parts_and_orders_late_archives(datasets):
    found = P.discover(datasets)
    assert set(found) == {"bdd100k", "culane"}
    names = [[p.name for p in parts] for parts in found["bdd100k"].archives]
    assert ["bdd100k_images_100k.zip.001", "bdd100k_images_100k.zip.002"] in names
    assert [parts[0].name for parts in found["culane"].archives][-1] == "annotations_new.tar.gz"
    with pytest.raises(ValueError, match="unknown sources"):
        P.discover(datasets, "bdd100k,kitti")


def test_prepare_converts_caches_and_writes_manifests(datasets, tmp_path, monkeypatch):
    from sparc.data.schema import Sample

    work, cache = tmp_path / "work", tmp_path / "cache"
    res = P.prepare(datasets, work, cache, jobs=2, margin_gb=0)
    assert set(res) == {"bdd100k", "culane"} and not any(r.from_cache for r in res.values())
    assert res["bdd100k"].frames == {"train": 2, "val": 1} and res["culane"].frames == {"train": 2, "val": 1}
    # the root was found through the zip's top folder; annotations_new overrode the driver's lines file
    s = Sample.load(work / "converted" / "culane" / "train" / "samples" / sorted(
        p.name for p in (work / "converted" / "culane" / "train" / "samples").glob("*.npz"))[0])
    assert any(abs(l.points[0, 0] - 400) < 1 for l in s.lanes)
    assert len(list(cache.glob("*.tar"))) == 2
    t, v = P.write_manifests(res, work, train_limit=100, val_limit=10)
    m = yaml.safe_load(t.read_text())
    assert [Path(e["root"]).name for e in m["train"]] == ["bdd100k", "culane"]
    assert sum(e["weight"] for e in m["train"]) == pytest.approx(1.0, abs=1e-3) and m["train"][0]["limit"] == 100
    assert m["val"][0]["limit"] == 10 and "limit" not in yaml.safe_load(v.read_text())["val"][0]
    # second session: the local conversion is gone, the cache restores it without running a converter
    shutil.rmtree(work / "converted")

    def boom(*a, **k):
        raise AssertionError("converter must not run")

    calls = []
    real_run = P._run

    def run(cmd, log=None):
        if "convert" in cmd:
            boom()
        calls.append(cmd[0])
        return real_run(cmd, log)

    monkeypatch.setattr(P, "_run", run)
    res2 = P.prepare(datasets, work, cache, jobs=2, margin_gb=0)
    assert all(r.from_cache for r in res2.values()) and res2["bdd100k"].frames == {"train": 2, "val": 1}
    img = work / "converted" / "bdd100k" / "train" / "images"
    assert all(p.exists() for p in img.iterdir())  # the image links still resolve into raw/


def test_disk_planning_skips_sources_that_do_not_fit(datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "free_bytes", lambda path: 1)
    res = P.prepare(datasets, tmp_path / "w", None, margin_gb=0)
    assert res == {}


def test_viz_renders_items_and_predictions(tmp_path):
    import torch

    from sparc.data.dataset import SparcDataset, collate
    from sparc.data.synthetic import make_dataset
    from sparc.data.virtual_camera import VirtualCamera
    from sparc.data.vocab import SignVocab
    from sparc.nn.decoder import DecoderCfg
    from sparc.nn.model import SparcCfg, SparcNet
    from sparc.tools.viz import draw_item, draw_prediction

    make_dataset(tmp_path / "syn", 3, 1, seed=2)
    vocab = SignVocab.from_manifests([tmp_path / "syn"])
    ds = SparcDataset(tmp_path / "syn", "train", vocab, VirtualCamera((192, 320)))
    it = ds[0]
    a = draw_item(it, vocab.names)
    assert a.shape == (192, 320, 3) and a.dtype == np.uint8 and not np.array_equal(a, it["image"].permute(1, 2, 0).numpy())
    net = SparcNet(SparcCfg(backbone="yolo26s", imgsz=(192, 320), bev=False, decoder=DecoderCfg(layers=2, n_sign=len(vocab)))).eval()
    batch = collate([it])
    with torch.no_grad():
        out = net(batch["image"].float() / 255, K=batch["K"], cam2ego=batch["cam2ego"])
    b = draw_prediction(it, out, 0, vocab.names, score=0.0, lane_exist=0.0)
    assert b.shape == (192, 320, 3)
