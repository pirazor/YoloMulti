"""Builds colab/train_sparc.ipynb (SPARC stage A on a Colab A100). Run: python colab/build_train_sparc.py

Each cell is a (kind, source) pair. The data / training logic lives in the package (``sparc.tools.prepare``,
``sparc.tools.viz``, ``python -m sparc ...``) where it is unit-tested; the notebook only wires it together.
"""
import ast
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("train_sparc.ipynb"))
COLAB_URL = "https://colab.research.google.com/github/pirazor/YoloMulti/blob/multitask/colab/train_sparc.ipynb"
cells = []


def md(s):
    cells.append(("markdown", s.strip("\n")))


def code(s, tags=None):
    cells.append(("code", s.strip("\n"), tags or []))


# ======================================================================================================== intro
md(f"""
# SPARC stage A on a Colab A100

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

Trains the SPARC network (`docs/sparc/plan.md`, stage A) on public datasets: road users, traffic lights (colour) and
signs (fine class), lanes as 3D ego-frame polylines with order / direction / continuity / type, and the drivable
area. Everything that matters is written to Google Drive, so a disconnected session resumes where it stopped.

**Before you start**
1. *Runtime -> Change runtime type -> A100 GPU* (Colab Pro / Pro+). High-RAM is on by default for A100.
2. Datasets. The notebook **downloads the public ones itself** (`DOWNLOAD`, default `bdd100k,culane,tt100k`) into
   `DATASETS_DIR` on Google Drive (default `MyDrive/datasets`, one folder per source), once: later sessions find them
   complete and skip the download. That set alone trains every stage A output:

| source | downloaded from | size | trains |
|---|---|---|---|
| BDD100K | ETH Zurich mirror (100k train + val images, det_20, drivable, lane labels) | ~6 GB | road users, light colour, drivable area, lanes |
| CULane | official Google Drive files (train/val drivers, list, annotations_new; `CULANE_TEST` adds the test set) | ~40 GB | lanes with ego order, the CULane F1 benchmark |
| TT100K | Tsinghua (`data.zip`, 2016 release) | ~20 GB | fine sign classes |

   Google Drive needs about 70 GB free for these (a 100 GB / 2 TB Google One plan); with less, set `DATASETS_DIR` to
   `/content/datasets` (local disk, downloaded again in every session). If Google Drive answers "too many users have
   downloaded this file", its quota resets within a day: the cell names the file to fetch in the browser instead.

   The sources below need an account, so they cannot be downloaded by a script; add them by hand when you have them
   (same folder layout), they are picked up automatically:

| folder | what to put in it (official file names) |
|---|---|
| `mtsd/` | `mtsd_v2_fully_annotated_annotation.zip`, `mtsd_v2_fully_annotated_images.{{train.0,train.1,train.2,val}}.zip` |
| `openlane/` | `images_training_*.tar` / `images_validation_*.tar` (a subset of the parts is fine), `lane3d_1000_v1.2.tar` |
| `zod/` | the ZOD Frames download (archives or the unpacked `single_frames/` folder) |
| `lisa_tl/`, `bosch_tl/`, `tt100k/`, `nuimages/`, `curvelanes/` | the respective downloads (split `x.zip.001 ...` parts are joined) |

Archives are unpacked to the A100's local disk (about 160 GB usable; sources are added in priority order and skipped
with a message when they would not fit). Converted samples are cached on Drive, so later sessions only re-unpack.

**Presets** (run all cells top to bottom, *Runtime -> Run all*):

| preset | data | training samples | use |
|---|---|---|---|
| `smoke` | synthetic scenes, no downloads | 4 x 4 000 | about 15 min: proves the whole GPU path before you spend hours |
| `quick` | up to 20 000 frames per source | 16 x 30 000 | first real numbers, catches conversion problems |
| `full` | every frame | 50 x 100 000 | the stage A run; spans several sessions, resumed automatically |

The throughput check (section 4) measures images / s on your data and prints the expected training time before the
long run starts. Batch size (48 on an 80 GB A100, 32 on 40 GB) and learning rate are set from the GPU; if the batch
does not fit, the check halves it.
""")

# ======================================================================================================== params
md("## 1 · Settings")
code('''
PRESET = "smoke"                                   # @param ["smoke", "quick", "full"]
DATASETS_DIR = "/content/drive/MyDrive/datasets"   # @param {type:"string"}
DOWNLOAD = "bdd100k,culane,tt100k"                 # @param {type:"string"}
CULANE_TEST = False                                # @param {type:"boolean"}
SOURCES = "auto"                                   # @param {type:"string"}
DRIVE_DIR = "/content/drive/MyDrive/sparc"         # @param {type:"string"}
RUN_NAME = ""                                      # @param {type:"string"}
RESUME = True                                      # @param {type:"boolean"}
BACKBONE = "vit_t16_dinov3"                        # @param ["vit_t16_dinov3", "vit_s16_dinov3", "convnext_t_dinov3", "yolo26s"]
DISTILL = True                                     # @param {type:"boolean"}
TEACHER = "dinov3_b"                               # @param ["dinov3_b", "dinov3_l", "dinov3_s"]
EPOCHS = 0                                         # @param {type:"integer"}
EPOCH_SIZE = 0                                     # @param {type:"integer"}
BATCH = 0                                          # @param {type:"integer"}
LR = 0.0                                           # @param {type:"number"}
TRAIN_FRAMES_PER_SOURCE = 0                        # @param {type:"integer"}
VAL_FRAMES_PER_SOURCE = 0                          # @param {type:"integer"}
FULL_VALIDATION = True                             # @param {type:"boolean"}
CACHE_CONVERTED = True                             # @param {type:"boolean"}
REPO_URL = "https://github.com/pirazor/YoloMulti.git"  # @param {type:"string"}
REPO_REF = "multitask"                             # @param {type:"string"}

# advanced (testing / other machines)
WORK_DIR = "/content"        # local SSD: code, unpacked datasets, converted samples
REPO_DIR = ""                # use an existing checkout instead of cloning
SKIP_INSTALL = False
ALLOW_CPU = False            # run without a GPU (tests only)
IMGSZ = [384, 640]
PRETRAINED = True
SMOKE_FRAMES = 1200          # synthetic training frames of the smoke preset
SKIP_BENCH = False
LOG_EVERY = 100
DISK_MARGIN_GB = 20          # local disk kept free while unpacking (a source that does not fit is skipped)
''', tags=["parameters"])

code('''
# preset defaults (a non-zero setting above wins)
PRESETS = {
    #           epochs  epoch_size  train frames/source  val frames/source  val_every
    "smoke": dict(epochs=4, epoch_size=4000, train_limit=None, val_limit=200, val_every=2),
    "quick": dict(epochs=16, epoch_size=30000, train_limit=20000, val_limit=800, val_every=2),
    "full": dict(epochs=50, epoch_size=100000, train_limit=None, val_limit=2000, val_every=2),
}
P = dict(PRESETS[PRESET])
EPOCHS = EPOCHS or P["epochs"]
EPOCH_SIZE = EPOCH_SIZE or P["epoch_size"]
TRAIN_LIMIT = TRAIN_FRAMES_PER_SOURCE or P["train_limit"]
VAL_LIMIT = VAL_FRAMES_PER_SOURCE or P["val_limit"]
VAL_EVERY = P["val_every"]
RUN_NAME = RUN_NAME or f"stage_a_{PRESET}"
print(f"preset {PRESET}: {EPOCHS} epochs x {EPOCH_SIZE} samples, train frames/source {TRAIN_LIMIT or 'all'}, "
      f"val frames/source {VAL_LIMIT}, validate every {VAL_EVERY} epochs, run '{RUN_NAME}'")
''')

# ======================================================================================================== setup
md("## 2 · Code, dependencies, GPU")
code('''
import json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path

q = shlex.quote
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def sh(cmd, check=True):
    """Run a shell command, output streamed into the cell; raise when it fails."""
    print(f"$ {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True)
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc


def secret(name):
    """Environment variable, else a Colab secret (left sidebar, key icon), else None."""
    if os.environ.get(name):
        return os.environ[name]
    if IN_COLAB:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            return None
    return None


WORK = Path(WORK_DIR)
WORK.mkdir(parents=True, exist_ok=True)
if IN_COLAB and any(str(p).startswith("/content/drive") for p in (DATASETS_DIR, DRIVE_DIR)):
    from google.colab import drive
    drive.mount("/content/drive")

if REPO_DIR:
    REPO = Path(REPO_DIR).resolve()
else:
    REPO = WORK / "YoloMulti"
    url, token = REPO_URL, secret("GITHUB_TOKEN")
    if token and url.startswith("https://github.com/"):
        url = url.replace("https://", f"https://x-access-token:{token}@")
    if (REPO / ".git").exists():
        sh(f"git -C {q(str(REPO))} fetch -q --depth 1 {q(url)} {q(REPO_REF)} && git -C {q(str(REPO))} checkout -q FETCH_HEAD")
    else:
        sh(f"git clone -q --depth 1 --branch {q(REPO_REF)} {q(url)} {q(str(REPO))}")
NEEDS = ["sparc/engine/trainer.py", "sparc/tools/prepare.py", "sparc/tools/viz.py", "sparc/data/converters/bdd100k.py"]
missing = [f for f in NEEDS if not (REPO / f).exists()]
if missing:
    raise SystemExit(f"the code at REPO_REF={REPO_REF!r} is older than this notebook (missing {missing}); "
                     "set REPO_REF to the branch this notebook came from ('multitask'), restart and Run all")
sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.chdir(REPO)
if not SKIP_INSTALL:
    sh(f'{q(sys.executable)} -m pip install -q "ultralytics==8.4.171" "timm>=1.0.20" scipy onnx onnxruntime onnxslim gdown')
os.environ.setdefault("YOLO_OFFLINE", "1")  # no Ultralytics telemetry / update checks
hf = secret("HF_TOKEN")
if hf:
    os.environ["HF_TOKEN"] = hf
print("code", REPO, "commit", subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip() or "?",
      "| Hugging Face token", "set" if hf else "not set")
''')

code('''
import torch

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    GPU_GB = props.total_memory / 2**30
    DEVICE = "cuda:0"
    print(f"GPU {props.name}, {GPU_GB:.0f} GB")
    if "A100" not in props.name and "H100" not in props.name:
        print("note: this notebook is tuned for an A100; the batch size below follows the GPU memory")
else:
    GPU_GB, DEVICE = 0.0, "cpu"
    if not ALLOW_CPU:
        raise SystemExit("no GPU: Runtime -> Change runtime type -> A100 GPU, then Run all")
try:
    BF16 = DEVICE != "cpu" and torch.cuda.is_bf16_supported(including_emulation=False)
except TypeError:  # older torch
    BF16 = DEVICE != "cpu" and torch.cuda.is_bf16_supported()
AMP = "bf16" if BF16 else ("fp16" if DEVICE != "cpu" else "none")
BATCH_USER = BATCH  # 0 = choose from the GPU memory (the throughput check halves it if it does not fit)
BATCH_AUTO = 48 if GPU_GB >= 70 else 32 if GPU_GB >= 38 else 16 if GPU_GB >= 20 else 8 if DEVICE != "cpu" else 4
BATCH = BATCH_USER or BATCH_AUTO
WORKERS = max(2, min(12, (os.cpu_count() or 4) - 2))
print(f"device {DEVICE} | amp {AMP} | batch {BATCH} | {WORKERS} data-loader workers | {os.cpu_count()} CPUs | "
      f"local disk free {shutil.disk_usage(WORK).free / 2**30:.0f} GB")
''')

# ======================================================================================================== data
md("""
## 3 · Data
`sparc.tools.download` fetches the public datasets listed in `DOWNLOAD` into `DATASETS_DIR` (resumable, MD5-checked,
skipped when already complete). `sparc.tools.prepare` then unpacks every source found in `DATASETS_DIR` to local disk, finds its root, converts it to SPARC
samples (sources in parallel) or restores the cached conversion from Drive, and writes the training manifests. The
`smoke` preset generates synthetic road scenes instead. Read the conversion summaries: every skipped record is
counted by reason.
""")
code('''
import logging

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S", force=True)
DATA = WORK / "data"
DRIVE = Path(DRIVE_DIR)
RUN_DIR = DRIVE / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

if PRESET == "smoke":
    from sparc.data.synthetic import make_dataset

    syn = DATA / "converted" / "synthetic"
    if not (syn / "manifest.json").exists():
        make_dataset(syn, n_train=SMOKE_FRAMES, n_val=max(16, SMOKE_FRAMES // 10), seed=0, overwrite=True)
    m = {"train": [{"root": str(syn), "weight": 1.0}], "val": [{"root": str(syn), "limit": VAL_LIMIT}]}
    TRAIN_MANIFEST, VAL_FULL_MANIFEST = DATA / "data_train.yaml", DATA / "data_val_full.yaml"
    TRAIN_MANIFEST.write_text(yaml.safe_dump(m, sort_keys=False))
    VAL_FULL_MANIFEST.write_text(yaml.safe_dump({"train": m["train"], "val": [{"root": str(syn)}]}, sort_keys=False))
else:
    from sparc.tools.download import MANUAL, download
    from sparc.tools.prepare import discover, prepare, write_manifests

    if DOWNLOAD.strip():
        t0 = time.time()
        try:
            got = download(DATASETS_DIR, DOWNLOAD, culane_test=CULANE_TEST)
        except RuntimeError as e:  # a failed source is named; the rest is on disk and the next run resumes
            print(f"\\n{e}\\n\\nre-run this cell to resume; or remove the failing source from DOWNLOAD to go on without it")
            raise SystemExit("download incomplete") from None
        print(f"downloads complete in {(time.time() - t0) / 60:.0f} min: " + ", ".join(f"{k} ({len(v)} files)" for k, v in got.items()))
    missing = [k for k in MANUAL if not (Path(DATASETS_DIR) / k).is_dir()]
    if missing:
        print("not present (need an account; optional):\\n" + "\\n".join(f"  {k:11s} {MANUAL[k]}" for k in missing))
    found = discover(DATASETS_DIR, SOURCES)
    if not found:
        raise SystemExit(f"no dataset folders in {DATASETS_DIR}: see the table at the top of the notebook")
    for name, f in sorted(found.items()):
        print(f"  {name:11s} {f.bytes / 2**30:7.1f} GB  {'unpacked folder' if f.unpacked else str(len(f.archives)) + ' archive(s)'}")
    t0 = time.time()
    PREPARED = prepare(DATASETS_DIR, DATA, DRIVE / "cache" if CACHE_CONVERTED else None, SOURCES,
                       jobs=min(4, max(1, WORKERS // 3)), margin_gb=DISK_MARGIN_GB)
    if not PREPARED:
        raise SystemExit("no source could be prepared (see the log above)")
    TRAIN_MANIFEST, VAL_FULL_MANIFEST = write_manifests(PREPARED, DATA, TRAIN_LIMIT, VAL_LIMIT)
    print(f"\\nprepared in {(time.time() - t0) / 60:.0f} min")
    for s, r in PREPARED.items():
        print(f"  {s:11s} {r.frames}  signs {r.signs}  {'(cache)' if r.from_cache else ''}")
    for s in sorted(set(found) - set(PREPARED)):
        print(f"  {s:11s} NOT prepared (disk space or conversion error above)")
print("\\ntraining manifest:\\n" + TRAIN_MANIFEST.read_text())
''')

md("""
### Data check
Two frames per source, warped into the virtual camera exactly as the network sees them (the second one augmented):
drivable area (blue = direct, orange = alternative), lane markings (yellow mask), lanes from the 3D targets coloured by
ego order (violet left, orange right), road users (white) and traffic elements (light colour or sign class). Boxes
off their objects or lanes off their markings mean a conversion problem: fix it before training.
""")
code('''
import matplotlib.pyplot as plt
import numpy as np
import torch

from sparc.data.dataset import AugCfg, SparcDataset
from sparc.data.virtual_camera import VirtualCamera
from sparc.data.vocab import SignVocab
from sparc.engine.config import DataCfg
from sparc.tools.viz import draw_item

data_cfg = DataCfg.load(TRAIN_MANIFEST)
CAMERA = VirtualCamera(tuple(IMGSZ), 60.0)
VOCAB = SignVocab.from_manifests([s.root for s in data_cfg.train] + [s.root for s in data_cfg.val])
print(f"{len(VOCAB)} fine sign classes")
rows = []
torch.manual_seed(0)
for s in data_cfg.train:
    plain = SparcDataset(s.root, "train", VOCAB, CAMERA, None, limit=64)
    aug = SparcDataset(s.root, "train", VOCAB, CAMERA, AugCfg(), limit=64)
    i = len(plain) // 2
    rows.append((plain.source, draw_item(plain[i], VOCAB.names), draw_item(aug[i], VOCAB.names)))
fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.6 * len(rows)), squeeze=False)
for r, (name, a, b) in enumerate(rows):
    for c, img in enumerate((a, b)):
        axes[r, c].imshow(img)
        axes[r, c].set_axis_off()
        axes[r, c].set_title(f"{name} ({'augmented' if c else 'as converted'})", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(RUN_DIR / "data_check.jpg", dpi=80)
plt.show()
''')

# ======================================================================================================== weights / config
md("""
## 4 · Pretrained weights and configuration
The default backbone is the DINOv3-distilled ViT-Tiny and the distillation teacher DINOv3 ViT-B. Their Hugging Face
repositories are under the DINOv3 licence: if the download is refused, the notebook stops (it never trains with
substitute weights). Accept the licence on the model page (link printed below), add a Colab secret `HF_TOKEN` with a
read token, and re-run. On a resumed run the saved configuration is reused.
""")
code('''
import math

import timm

from adas_mt.distill.teacher import ALIASES
from sparc.engine.config import TrainCfg
from sparc.nn.backbones_spec import BACKBONES

_PROBED = {}


def downloadable(timm_name):
    """True when the pretrained weights load (downloaded once into the Hugging Face cache, reused by training)."""
    if timm_name in _PROBED:
        return _PROBED[timm_name]
    try:
        timm.create_model(timm_name, pretrained=True, num_classes=0)
        _PROBED[timm_name] = True
        return True
    except Exception as e:  # gated repository, no network, ...
        repo = "timm/" + timm_name
        print(f"  cannot load {timm_name}: {type(e).__name__}: {str(e)[:160]}\\n  -> accept the licence at "
              f"https://huggingface.co/{repo} and add an HF_TOKEN secret")
        _PROBED[timm_name] = False
        return False


def timm_name(key):
    spec = BACKBONES[key]
    return spec["timm"] + "." + spec["tag"] if spec["kind"] != "yolo" else None


BASE = TrainCfg.load(REPO / "sparc" / "cfg" / "stage_a.yaml")  # reference recipe: lr at its batch, warm-up


def configure_batch(cfg, batch):
    """Batch size and what follows from it: lr by square-root scaling from the reference batch (AdamW), warm-up at
    most a tenth of the run (short presets would otherwise never leave it)."""
    cfg.batch = int(batch)
    cfg.lr = LR or BASE.lr * math.sqrt(cfg.batch / BASE.batch)
    cfg.warmup_iters = min(BASE.warmup_iters, max(20, cfg.epochs * max(1, cfg.epoch_size // cfg.batch) // 10))


CFG_FILE = RUN_DIR / "train_cfg.yaml"
LAST = RUN_DIR / "weights" / "last.pt"
RESUMING = RESUME and LAST.exists() and CFG_FILE.exists()
if RESUMING:
    cfg = TrainCfg.load(CFG_FILE)
    if EPOCHS > cfg.epochs:
        print(f"extending the run from {cfg.epochs} to {EPOCHS} epochs")
        cfg.epochs = EPOCHS
    if BATCH_USER and BATCH_USER != cfg.batch:
        print(f"batch {cfg.batch} -> {BATCH_USER} (set in the settings); the learning rate of the run is kept")
        cfg.batch = BATCH_USER
    elif cfg.batch > BATCH_AUTO:
        print(f"note: this run uses batch {cfg.batch}, more than this GPU would get ({BATCH_AUTO}); if training stops "
              f"with 'CUDA out of memory', set BATCH = {BATCH_AUTO} and re-run")
    print(f"resuming {RUN_DIR} with its saved configuration (backbone {cfg.model.backbone})")
else:
    backbone, teacher = BACKBONE, TEACHER
    if PRETRAINED and timm_name(backbone) and not downloadable(timm_name(backbone)):
        raise SystemExit(f"the {backbone} weights cannot be downloaded (see above): check the internet connection and "
                         "the HF_TOKEN secret, then re-run this cell")
    distill = DISTILL and PRETRAINED
    if distill and not downloadable(ALIASES.get(teacher, teacher)):
        raise SystemExit(f"the teacher {teacher} cannot be downloaded (see above): accept its licence, add the "
                         "HF_TOKEN secret and re-run this cell, or set DISTILL = False to train without a teacher")
    overrides = {"epochs": EPOCHS, "epoch_size": EPOCH_SIZE, "workers": WORKERS, "amp": AMP, "val_every": VAL_EVERY,
                 "log_every": LOG_EVERY, "model.backbone": backbone, "model.pretrained": PRETRAINED,
                 "model.imgsz": list(IMGSZ), "distill.enabled": distill, "distill.teacher": teacher}
    cfg = TrainCfg.load(REPO / "sparc" / "cfg" / "stage_a.yaml", overrides)
    configure_batch(cfg, BATCH)
CFG_FILE.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
print(yaml.safe_dump({k: cfg.to_dict()[k] for k in ("model", "distill", "epochs", "epoch_size", "batch", "lr",
                                                   "warmup_iters", "amp", "val_every")}, sort_keys=False))
''')

md("""
### Throughput check
A short training run (40 iterations, the real model and data) before the long one: it proves the GPU path works and
measures images / s, the share of time spent waiting for data, and GPU memory, from which the run time is estimated.
A data wait above ~30 % means the CPU decoding is the bottleneck (fewer, larger epochs or a smaller `TRAIN_FRAMES` do
not change that; a faster runtime does).
""")
code('''
import csv
from collections import deque


def run(cmd, log=None):
    """Run a command with its output streamed into the cell (and appended to `log`); Ctrl-C / stop -> terminate.
    The last output lines are kept in ``run.tail``."""
    print("$ " + " ".join(q(str(c)) for c in cmd), flush=True)
    fh = open(log, "a", encoding="utf-8") if log else None
    run.tail = deque(maxlen=200)
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            run.tail.append(line)
            if fh:
                fh.write(line)
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        p.wait()
        print("\\ninterrupted: the last finished epoch is in weights/last.pt; re-run this cell to resume")
        rc = -1
    finally:
        if fh:
            fh.close()
    return rc


def last_row(results_csv):
    rows = list(csv.DictReader(open(results_csv)))
    return {k: float(v) for k, v in rows[-1].items() if v not in ("", None)} if rows else {}


THROUGHPUT = None
if not SKIP_BENCH and not RESUMING:
    bench = WORK / "bench"
    bm = yaml.safe_load(TRAIN_MANIFEST.read_text())
    bm["val"] = [dict(v, limit=8) for v in bm["val"]][:1]
    (WORK / "bench_data.yaml").write_text(yaml.safe_dump(bm, sort_keys=False))
    while True:
        shutil.rmtree(bench, ignore_errors=True)
        bcfg = TrainCfg.from_dict(cfg.to_dict())
        bcfg.epochs, bcfg.epoch_size, bcfg.log_every, bcfg.val_every = 1, cfg.batch * (40 if DEVICE != "cpu" else 3), 10, 0
        (WORK / "bench_cfg.yaml").write_text(yaml.safe_dump(bcfg.to_dict(), sort_keys=False))
        rc = run([sys.executable, "-m", "sparc", "train", "--cfg", WORK / "bench_cfg.yaml", "--data",
                  WORK / "bench_data.yaml", "--out", bench, "--device", DEVICE])
        if rc == 0:
            break
        if "out of memory" in "".join(run.tail).lower() and cfg.batch > 4:
            print(f"\\nbatch {cfg.batch} does not fit in GPU memory: retrying with {cfg.batch // 2}\\n")
            configure_batch(cfg, cfg.batch // 2)
            CFG_FILE.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
            continue
        raise SystemExit("the throughput check failed: read the error above before starting the long run")
    r = last_row(bench / "results.csv")
    THROUGHPUT = r.get("train/img_per_s", 0.0)
    eta_h = cfg.epochs * cfg.epoch_size / max(THROUGHPUT, 1e-9) / 3600
    print(f"\\n{THROUGHPUT:.0f} img/s (first iterations included, so a lower bound), data wait "
          f"{100 * r.get('train/data_wait_frac', 0):.0f} %, GPU memory {r.get('train/gpu_mem_gb', 0):.1f} GB")
    print(f"estimated training time: {eta_h:.1f} h for {cfg.epochs} x {cfg.epoch_size} samples (+ validation)")
''')

# ======================================================================================================== train
md("""
## 5 · Train
Checkpoints (`weights/last.pt` every epoch, `weights/best.pt` by fitness), `results.csv`, `train.log` and the
configuration go to `DRIVE_DIR/runs/RUN_NAME`. If the session disconnects: reconnect, keep the settings, *Run all*:
data is re-unpacked (conversions come from the cache) and training resumes from the last finished epoch.
""")
code('''
cmd = [sys.executable, "-m", "sparc", "train", "--cfg", CFG_FILE, "--data", TRAIN_MANIFEST, "--out", RUN_DIR,
       "--device", DEVICE]
if RESUMING:
    cmd += ["--resume", LAST, "--signs", RUN_DIR / "signs.json"]
t0 = time.time()
rc = run(cmd, log=RUN_DIR / "train.log")
print(f"\\ntraining {'finished' if rc == 0 else 'stopped'} after {(time.time() - t0) / 3600:.2f} h (exit code {rc})")
''')

# ======================================================================================================== results
md("## 6 · Results")
code('''
import pandas as pd

res = pd.read_csv(RUN_DIR / "results.csv")
val = res.dropna(subset=["fitness"]) if "fitness" in res else res.iloc[0:0]
SOURCE_ORDER = ["bdd100k", "culane", "openlane", "mtsd", "zod", "lisa_tl", "bosch_tl", "tt100k", "nuimages", "curvelanes",
                "dtld", "synthetic"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
srcs = [s for s in SOURCE_ORDER if any(c.startswith(s + "/") for c in res.columns)]
INK, MUTED, GRID, OTHER = "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"
# colour follows the source (fixed order, never cycled); a 9th+ source is drawn in neutral grey and named by its label
COLOR = {s: PALETTE[i] if i < len(PALETTE) else OTHER for i, s in enumerate(srcs)}


def end_labels(ax, ends):
    """Name the series at their last point (<= 4 series; the legend covers more), nudged apart so they never overlap."""
    if not 0 < len(ends) <= 4:
        return
    lo, hi = ax.get_ylim()
    ys = []
    for _, _, y in sorted(ends, key=lambda e: e[2]):
        ys.append(max(y, ys[-1] + 0.08 * (hi - lo)) if ys else y)
    for (name, x, _), y in zip(sorted(ends, key=lambda e: e[2]), ys):
        ax.annotate(name, (x, y), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK, va="center",
                    annotation_clip=False)


panels = [("train/total", "training loss", None), ("agents_mAP50-95", "road users mAP50-95", "src"),
          ("elements_mAP50", "lights + signs mAP50", "src"), ("da_mIoU", "drivable area mIoU", "src"),
          ("lane_F1", "lane F1 (CULane 30 px)", "src"), ("sign_top1", "sign top-1 (matched)", "src"),
          ("light_color_acc", "light colour accuracy", "src"), ("train/img_per_s", "throughput (img/s)", None)]
panels = [p for p in panels if (p[2] is None and p[0] in res) or (p[2] and any(f"{s}/{p[0]}" in res for s in srcs))]
cols = 4
fig, axes = plt.subplots((len(panels) + cols - 1) // cols, cols, figsize=(16, 3.4 * ((len(panels) + cols - 1) // cols)),
                         squeeze=False, facecolor="#fcfcfb")
for ax, (key, title, kind) in zip(axes.flat, panels):
    ax.set_facecolor("#fcfcfb")
    series = [(None, res["epoch"], res[key])] if kind is None else [
        (s, val["epoch"], val[f"{s}/{key}"]) for s in srcs if f"{s}/{key}" in val and val[f"{s}/{key}"].notna().any()]
    for s, x, y in series:
        ax.plot(x, y, color=COLOR.get(s, PALETTE[0]), lw=2, marker="o" if len(x) < 15 else None, ms=4)
    end_labels(ax, [(s, x.iloc[-1], y.iloc[-1]) for s, x, y in series if s and len(x)])
    ax.set_xlim(res["epoch"].min() - 0.3, res["epoch"].max() + 0.3)  # every panel on the run's epoch range
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("epoch", fontsize=8, color=MUTED)
for ax in list(axes.flat)[len(panels):]:
    ax.set_visible(False)
if srcs and len(val):
    handles = [plt.Line2D([], [], color=COLOR[s], lw=2) for s in srcs]
    fig.legend(handles, srcs, loc="upper center", ncol=len(srcs), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
fig.savefig(RUN_DIR / "results.png", dpi=90, bbox_inches="tight")
plt.show()
if len(val):
    best = val.loc[val["fitness"].idxmax()]
    keys = [c for c in val.columns if "/" in c and not c.startswith("train/") and any(
        t in c for t in ("mAP50-95", "mAP50", "da_mIoU", "lane_F1", "sign_top1", "light_color_acc", "lane_order_acc"))
            and "AP50_" not in c]
    print(f"best epoch {int(best['epoch'])} (fitness {best['fitness']:.4f})")
    display(pd.DataFrame({"best": best[keys], "last": val.iloc[-1][keys]}).round(3))
''')

md("### Predictions on validation frames (best checkpoint, network view)")
code('''
from sparc.data.dataset import collate
from sparc.engine.trainer import load_for_eval
from sparc.tools.viz import draw_prediction

BEST = RUN_DIR / "weights" / "best.pt"
if not BEST.exists():
    BEST = LAST
model, vocab, mcfg = load_for_eval(BEST, DEVICE)
val_cfg = DataCfg.load(TRAIN_MANIFEST)
panels = []
for s in val_cfg.val:
    ds = SparcDataset(s.root, "val", vocab, VirtualCamera(tuple(mcfg.model.imgsz), mcfg.model.hfov), None, limit=3)
    for i in range(min(2, len(ds))):
        it = ds[i]
        batch = collate([it])
        with torch.no_grad(), torch.autocast(DEVICE.split(":")[0], dtype=torch.bfloat16, enabled=AMP == "bf16"):
            out = model(batch["image"].to(DEVICE).float() / 255, K=batch["K"].to(DEVICE), cam2ego=batch["cam2ego"].to(DEVICE))
        out = {k: v.float() for k, v in out.items() if torch.is_tensor(v)}
        panels.append((ds.source, draw_prediction(it, out, 0, vocab.names)))
fig, axes = plt.subplots((len(panels) + 1) // 2, 2, figsize=(14, 3.6 * ((len(panels) + 1) // 2)), squeeze=False)
for ax, (name, img) in zip(axes.flat, panels):
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(name, fontsize=10, loc="left")
for ax in list(axes.flat)[len(panels):]:
    ax.set_visible(False)
fig.tight_layout()
fig.savefig(RUN_DIR / "predictions.jpg", dpi=80)
plt.show()
''')

md("""
## 7 · Full validation and stage A acceptance
Every validation frame of every prepared source (in-training validation used a subset). Targets from
`docs/sparc/plan.md`: CULane F1 >= 0.79 and drivable-area IoU >= 0.92 are absolute; the detection target is relative
(DEIMv2-S fine-tuned on BDD100K, minus 0.5 mAP), so the BDD100K number below is to be compared with that baseline.
""")
code('''
if FULL_VALIDATION:
    rc = run([sys.executable, "-m", "sparc", "val", "--weights", BEST, "--data", VAL_FULL_MANIFEST, "--out",
              RUN_DIR / "val_full", "--device", DEVICE, "--workers", WORKERS])
    full = json.loads((RUN_DIR / "val_full" / "val.json").read_text()) if rc == 0 else {}
    checks = [("bdd100k/bdd10_mAP50-95", "BDD100K 10-class mAP50-95", None),
              ("bdd100k/agents_mAP50-95", "BDD100K road users mAP50-95", None),
              ("bdd100k/da_IoU_fg", "BDD100K drivable IoU (all drivable pixels)", 0.92),
              ("bdd100k/da_mIoU", "BDD100K drivable mIoU (direct, alternative)", None),
              ("culane/lane_F1", "CULane F1", 0.79), ("mtsd/sign_top1", "MTSD sign top-1 (matched boxes)", None),
              ("lisa_tl/light_color_acc", "LISA light colour accuracy", None),
              ("openlane/lane_F1", "OpenLane lane F1 (image)", None), ("openlane/lane_order_acc", "OpenLane ego-order accuracy", None)]
    table = [(name, full[k], ("pass" if full[k] >= t else "below") if t is not None else "compare with baseline")
             for k, name, t in checks if k in full]
    display(pd.DataFrame(table, columns=["metric", "value", "target"]).round(4))
    print(f"all metrics: {RUN_DIR / 'val_full' / 'val.json'}")
''')

md("## 8 · Export for the Jetson (ONNX)")
code('''
EXPORT = RUN_DIR / "export"
EXPORT.mkdir(exist_ok=True)
probe = next((p for p in sorted((Path(DataCfg.load(TRAIN_MANIFEST).val[0].root) / "val" / "images").iterdir())), None)
cmd = [sys.executable, "-m", "sparc", "export", "--weights", BEST, "--out", EXPORT / "sparc_stage_a.onnx"]
if probe is not None:
    cmd += ["--image", probe]
rc = run(cmd)
print("\\nnext: copy sparc_stage_a.onnx to the Jetson and build the engine with trtexec --fp16 (docs/sparc/phase0.md)")
''')

md("## 9 · Files")
code('''
for p in sorted(RUN_DIR.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 2**20:9.1f} MB  {p.relative_to(RUN_DIR.parent)}")
''')

# ======================================================================================================== write
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm", "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "cells": []}
for i, c in enumerate(cells):
    if c[0] == "code":
        ast.parse(c[1], filename=f"cell {i}")  # escaping mistakes in the templates above fail here, not in Colab
    src = c[1].splitlines(keepends=True)
    if c[0] == "markdown":
        nb["cells"].append({"cell_type": "markdown", "id": f"c{i:02d}", "metadata": {}, "source": src})
    else:
        meta = {"tags": c[2]} if c[2] else {}
        nb["cells"].append({"cell_type": "code", "id": f"c{i:02d}", "metadata": meta, "execution_count": None,
                            "outputs": [], "source": src})
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {OUT} ({len(cells)} cells)")
