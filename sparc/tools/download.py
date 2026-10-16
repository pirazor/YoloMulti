"""Download the public datasets that need no account into the folder layout ``prepare`` reads.

    python -m sparc download -- --dest /content/drive/MyDrive/datasets --sources bdd100k,culane,tt100k

Every source lands in ``<dest>/<source>/`` as its original archives, so ``python -m sparc prepare`` picks it up
unchanged. Downloads resume (HTTP range requests, ``gdown --continue``), are checked against the published MD5 where
one exists, and are skipped when already complete (``<dest>/<source>/.complete.json`` records what was fetched), so
re-running in a new Colab session costs nothing.

What is downloaded (all research licences; by downloading you accept each dataset's terms):

=========  ===================================================  ==================================================
source     files                                                supervises
=========  ===================================================  ==================================================
bdd100k    100k train + val images, det_20, drivable, lane       road users, light colour, drivable area, lanes
           labels (ETH Zurich mirror, ~6 GB)
culane     3 train/val driver archives, list, annotations_new   lanes with ego order (the CULane F1 benchmark)
           (Google Drive, ~40 GB); ``--culane-test`` adds the
           3 test archives (~10 GB)
tt100k     2016 release ``data.zip`` (~20 GB)                    fine sign classes (221 Chinese sign classes)
=========  ===================================================  ==================================================

The other stage A sources need a login and cannot be fetched by a script: ``MANUAL`` lists where to register; put
their archives into ``<dest>/<source>/`` by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

LOG = logging.getLogger("sparc.download")
GB = 2 ** 30
BDD = "https://dl.cv.ethz.ch/bdd100k/data/"


@dataclass(frozen=True)
class Item:
    key: str                  # stable id of the file within its source (also the record key in .complete.json)
    url: str                  # http(s) URL, or a Google Drive file id for kind="gdrive"
    gb: float                 # approximate size, for the disk check
    kind: str = "http"        # "http" | "gdrive"
    md5_url: Optional[str] = None
    optional: str = ""        # name of the flag that enables it ("" = always)


CATALOG: Dict[str, List[Item]] = {
    "bdd100k": [
        Item("100k_images_train.zip", BDD + "100k_images_train.zip", 3.7, md5_url=BDD + "100k_images_train.zip.md5"),
        Item("100k_images_val.zip", BDD + "100k_images_val.zip", 0.6, md5_url=BDD + "100k_images_val.zip.md5"),
        Item("bdd100k_det_20_labels_trainval.zip", BDD + "bdd100k_det_20_labels_trainval.zip", 0.1),
        Item("bdd100k_drivable_labels_trainval.zip", BDD + "bdd100k_drivable_labels_trainval.zip", 0.5),
        Item("bdd100k_lane_labels_trainval.zip", BDD + "bdd100k_lane_labels_trainval.zip", 0.5),
    ],
    # Google Drive ids of the official CULane release (as listed by the CULane page and LaneATT's DATASETS.md);
    # gdown saves each under its original file name (driver_*_*frame.tar.gz, list.tar.gz, annotations_new.tar.gz).
    "culane": [
        Item("trainval_1", "1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk", 13.0, kind="gdrive"),
        Item("trainval_2", "1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL", 13.0, kind="gdrive"),
        Item("trainval_3", "14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL", 13.0, kind="gdrive"),
        Item("annotations_1", "1QbB1TOk9Fy6Sk0CoOsR3V8R56_eG6Xnu", 0.2, kind="gdrive"),
        Item("annotations_2", "18alVEPAMBA9Hpr3RDAAchqSj5IxZNRKd", 0.2, kind="gdrive"),
        Item("test_1", "1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8", 3.5, kind="gdrive", optional="culane_test"),
        Item("test_2", "1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV", 3.5, kind="gdrive", optional="culane_test"),
        Item("test_3", "1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7", 3.5, kind="gdrive", optional="culane_test"),
    ],
    "tt100k": [
        Item("data.zip", "https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip", 20.0),
    ],
}
DEFAULT = ("bdd100k", "culane", "tt100k")

# sources that need an account: where to get them (the archives go into <dest>/<source>/ unchanged)
MANUAL = {
    "mtsd": "https://www.mapillary.com/dataset/trafficsign (Mapillary login): mtsd_v2_fully_annotated_*.zip",
    "openlane": "https://github.com/OpenDriveLab/OpenLane (OpenDataLab / Google Drive form): images_*.tar, lane3d_1000*.tar",
    "zod": "https://zod.zenseact.com (registration; the zod CLI downloads Frames)",
    "nuimages": "https://www.nuscenes.org/nuimages (nuScenes account): nuimages-v1.0-all-*.tgz",
    "bosch_tl": "https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset (registration)",
    "lisa_tl": "https://www.kaggle.com/datasets (search 'LISA traffic light dataset'; Kaggle account)",
    "dtld": "https://www.uni-ulm.de/in/driveu/projekte/driveu-traffic-light-dataset (registration)",
    "curvelanes": "https://github.com/SoulmateB/CurveLanes (Google Drive / Baidu links)",
}


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: List[str]) -> int:
    LOG.info("$ %s", " ".join(cmd))
    return subprocess.call(cmd)


def _http(item: Item, folder: Path) -> Path:
    out = folder / item.key
    part = out.with_name(out.name + ".part")
    for attempt in range(4):
        rc = _run(["curl", "-fL", "--retry", "5", "--retry-delay", "10", "-C", "-", "-o", str(part), item.url])
        if rc == 0:
            break
        if rc == 33:  # server refuses the range request: restart from zero
            part.unlink(missing_ok=True)
        LOG.warning("%s: curl exit %d (attempt %d)", item.key, rc, attempt + 1)
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"download failed: {item.url}")
    if item.md5_url:
        tmp = folder / (item.key + ".md5")
        if _run(["curl", "-fsSL", "-o", str(tmp), item.md5_url]) == 0:
            want = tmp.read_text().split()[0].lower()
            tmp.unlink(missing_ok=True)
            got = _md5(part)
            if got != want:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"{item.key}: MD5 {got} != published {want}; the partial file was removed, re-run")
            LOG.info("%s: MD5 ok", item.key)
        else:
            LOG.warning("%s: no MD5 file at %s; size check only", item.key, item.md5_url)
    part.rename(out)
    return out


def _gdrive(item: Item, folder: Path) -> Path:
    if shutil.which("gdown") is None:
        raise RuntimeError("gdown is not installed: pip install gdown")
    before = {p.name for p in folder.iterdir()}
    # --continue resumes a partial file; the output folder keeps Google Drive's original file name
    rc = _run(["gdown", "--continue", "-O", str(folder) + "/", item.url])
    new = [p for p in folder.iterdir() if p.name not in before and p.is_file() and not p.name.endswith(".part")]
    if rc != 0 or not new:
        raise RuntimeError(
            f"Google Drive refused {item.url} (often 'too many users downloaded this file recently': the quota resets "
            f"within 24 h). Retry later, or open https://drive.google.com/file/d/{item.url} , download it in the browser "
            f"and put it into {folder}")
    return max(new, key=lambda p: p.stat().st_mtime)


def download(dest: str | Path, sources: Sequence[str] | str = DEFAULT, culane_test: bool = False,
             margin_gb: float = 5.0) -> Dict[str, List[str]]:
    """Fetch the catalog files of ``sources`` into ``dest/<source>/``; returns source -> file names present.
    A source whose files fail is reported and skipped (the others continue); the error list is raised at the end."""
    names = [s.strip() for s in (sources.split(",") if isinstance(sources, str) else sources) if s.strip()]
    unknown = [n for n in names if n not in CATALOG]
    if unknown:
        hint = "; ".join(f"{n}: {MANUAL[n]}" for n in unknown if n in MANUAL)
        raise ValueError(f"cannot download {unknown} automatically (choose from {sorted(CATALOG)}). {hint}")
    flags = {"culane_test": culane_test}
    done: Dict[str, List[str]] = {}
    errors: List[str] = []
    for name in names:
        folder = Path(dest) / name
        folder.mkdir(parents=True, exist_ok=True)
        rec_path = folder / ".complete.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
        items = [it for it in CATALOG[name] if not it.optional or flags.get(it.optional)]
        todo = [it for it in items if not (it.key in rec and (folder / rec[it.key]).exists())]
        need = sum(it.gb for it in todo)
        LOG.info("%s: %d of %d files to fetch (~%.0f GB)", name, len(todo), len(items), need)
        if todo and free_bytes(folder) < (need + margin_gb) * GB:
            errors.append(f"{name}: needs ~{need:.0f} GB + {margin_gb:.0f} GB margin, {free_bytes(folder) / GB:.0f} GB "
                          f"free at {folder}")
            LOG.error(errors[-1])
            continue
        for it in todo:
            t0 = time.time()
            try:
                path = _http(it, folder) if it.kind == "http" else _gdrive(it, folder)
            except Exception as e:  # keep going with the other files / sources; reported at the end
                errors.append(f"{name}/{it.key}: {e}")
                LOG.error(errors[-1])
                continue
            rec[it.key] = path.name
            rec_path.write_text(json.dumps(rec, indent=1))
            mb = path.stat().st_size / 2 ** 20
            LOG.info("%s: %s %.0f MB in %.0f s (%.0f MB/s)", name, path.name, mb, time.time() - t0,
                     mb / max(time.time() - t0, 1e-3))
        done[name] = sorted(rec[it.key] for it in items if it.key in rec)
    if errors:
        raise RuntimeError("some downloads failed:\n  " + "\n  ".join(errors))
    return done


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sparc download", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True, help="datasets folder (one sub-folder per source, read by prepare)")
    p.add_argument("--sources", default=",".join(DEFAULT), help=f"comma list from {sorted(CATALOG)}")
    p.add_argument("--culane-test", action="store_true", help="also the CULane test archives (~10 GB)")
    p.add_argument("--margin-gb", type=float, default=5.0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        res = download(a.dest, a.sources, a.culane_test, a.margin_gb)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(res, indent=1))
    print("needs an account (put the archives into <dest>/<source>/ by hand):")
    for k, v in MANUAL.items():
        print(f"  {k:11s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
