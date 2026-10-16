"""Dataset archives -> converted SPARC sources + training manifests (the data step of a training machine / Colab).

    python -m sparc prepare --datasets /content/drive/MyDrive/datasets --work /content/data \\
                            --cache /content/drive/MyDrive/sparc/cache --sources auto

``--datasets`` holds one folder per source, named as the converter (``bdd100k``, ``culane``, ``openlane``, ``mtsd``,
``tt100k``, ``zod``, ``nuimages``, ``curvelanes``, ``bosch_tl``, ``lisa_tl``, ``dtld``), with the original download
archives inside (``.zip``, ``.tar``, ``.tar.gz``, ``.tgz``, split archives ``x.zip.001 ...``) or the already unpacked
dataset. Per source:

1. the archives are unpacked to ``<work>/raw/<source>`` on local disk (tar streams straight from the folder; zip is
   copied first), in priority order, skipping a source whose estimated size does not fit on the disk;
2. the dataset root is located by a marker path (e.g. ``images/100k`` for BDD100K) and missing sibling folders that
   ended up elsewhere in the extraction are linked next to it;
3. the converter runs (``python -m sparc convert``; sources in parallel) into ``<work>/converted/<source>``, or a
   cached conversion is restored from ``--cache`` (keyed by the archives, the converter arguments and the
   converter's source code; image links point into ``<work>/raw``, which is why the raw path must stay fixed);
4. ``<work>/data_train.yaml`` (training sources with sampler weights and the per-source frame limit, validation
   sources with the in-training validation limit) and ``<work>/data_val_full.yaml`` (all validation frames) are
   written.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

LOG = logging.getLogger("sparc.prepare")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")
SPLIT_SUFFIX = ".001"
LATE = ("annotations_new", "update")  # archives that overwrite files of the others (CULane's corrected lanes)
GB = 2 ** 30


@dataclass(frozen=True)
class Spec:
    markers: Tuple[str, ...]          # relative path(s) whose presence identifies the dataset root
    weight: float                     # sampler weight (renormalised over the sources present)
    priority: int                     # extraction order when disk space is short (lower first)
    siblings: Tuple[str, ...] = ()    # folders that must sit next to the root's marker
    val: bool = True                  # has a validation split worth evaluating during training
    expand: float = 1.05              # extracted size / archive size (JPEG / PNG archives barely compress)
    args: Tuple[str, ...] = ()        # extra converter arguments
    alias: Tuple[str, str] = ("", "")  # (dir, path): a download rooted at <dir> instead of the marker is linked as <path>


SOURCES: Dict[str, Spec] = {
    "bdd100k": Spec(("images/100k",), 0.35, 0, siblings=("labels",), alias=("100k", "images/100k")),
    "culane": Spec(("list/train_gt.txt",), 0.10, 1),
    "mtsd": Spec(("splits/train.txt",), 0.06, 2, siblings=("annotations", "images")),
    "openlane": Spec(("lane3d_1000", "lane3d_300"), 0.15, 3, siblings=("images",)),
    "zod": Spec(("single_frames",), 0.20, 4),
    "lisa_tl": Spec(("Annotations",), 0.01, 5),
    "bosch_tl": Spec(("train.yaml",), 0.02, 5),
    "tt100k": Spec(("annotations_all.json", "annotations.json"), 0.04, 6),
    "nuimages": Spec(("v1.0-train",), 0.05, 7, siblings=("samples",)),
    "curvelanes": Spec(("train/labels", "valid/labels"), 0.05, 8),
    "dtld": Spec((), 0.02, 9, val=False, expand=1.0),
}


@dataclass
class Found:
    source: str
    folder: Path
    archives: List[List[Path]] = field(default_factory=list)  # each entry: one archive (several paths = split parts)
    unpacked: bool = False            # the folder holds the dataset itself (no archives)

    @property
    def bytes(self) -> int:
        if self.unpacked:
            return sum(f.stat().st_size for f in self.folder.rglob("*") if f.is_file())
        return sum(p.stat().st_size for parts in self.archives for p in parts)


def _is_archive(p: Path) -> bool:
    n = p.name.lower()
    return p.is_file() and any(n.endswith(s) for s in ARCHIVE_SUFFIXES)


def discover(datasets_dir: str | Path, sources: Sequence[str] | str = "auto") -> Dict[str, Found]:
    """Folders of ``datasets_dir`` named like a source, with their archives (split parts grouped)."""
    root = Path(datasets_dir)
    names = list(SOURCES) if sources in ("auto", None, "") else [s.strip() for s in (sources.split(",") if isinstance(sources, str) else sources) if s.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise ValueError(f"unknown sources {unknown}; choose from {sorted(SOURCES)}")
    out = {}
    for name in names:
        folder = root / name
        if not folder.is_dir():
            if sources not in ("auto", None, ""):
                LOG.warning("%s: no folder %s", name, folder)
            continue
        files = sorted(p for p in folder.iterdir() if p.is_file())
        archives = [[p] for p in files if _is_archive(p)]
        parts: Dict[str, List[Path]] = {}
        for p in files:
            stem, dot, num = p.name.rpartition(".")
            if dot and num.isdigit() and len(num) == 3:
                parts.setdefault(stem, []).append(p)
        archives += [sorted(v) for _, v in sorted(parts.items())]
        archives.sort(key=lambda a: (any(t in a[0].name.lower() for t in LATE), a[0].name))
        has_dirs = any(p.is_dir() for p in folder.iterdir())
        if archives:
            out[name] = Found(name, folder, archives)
        elif has_dirs:
            out[name] = Found(name, folder, unpacked=True)
    return out


def free_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def _run(cmd: List[str], log: Optional[Path] = None) -> None:
    LOG.info("$ %s", " ".join(cmd))
    if log is None:
        subprocess.run(cmd, check=True)
        return
    with open(log, "a", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        tail = log.read_text(errors="ignore").splitlines()[-20:]
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n" + "\n".join(tail))


def unpack(parts: List[Path], dst: Path, tmp: Path) -> None:
    """Unpack one archive (or its concatenated split parts) into ``dst`` with the system tools."""
    dst.mkdir(parents=True, exist_ok=True)
    name = parts[0].name.lower()
    if len(parts) > 1:  # x.zip.001, x.zip.002 ...: plain byte splits
        joined = tmp / parts[0].name.rsplit(".", 1)[0]
        tmp.mkdir(parents=True, exist_ok=True)
        with open(joined, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 64 << 20)
        try:
            unpack([joined], dst, tmp)
        finally:
            joined.unlink(missing_ok=True)
        return
    src = parts[0]
    if name.endswith(".zip"):
        local = src
        if not str(src.resolve()).startswith(str(tmp.resolve().parent)):
            tmp.mkdir(parents=True, exist_ok=True)
            local = tmp / src.name
            shutil.copyfile(src, local)  # zip needs random access: one sequential copy beats seeking through a network mount
        try:
            if shutil.which("unzip"):
                _run(["unzip", "-q", "-o", str(local), "-d", str(dst)])
            else:
                shutil.unpack_archive(str(local), str(dst))
        finally:
            if local != src:
                local.unlink(missing_ok=True)
        return
    if shutil.which("tar"):
        cmd = ["tar", "-xf", str(src), "-C", str(dst)]
        if (name.endswith(".gz") or name.endswith(".tgz")) and shutil.which("pigz"):
            cmd = ["tar", "-I", "pigz", "-xf", str(src), "-C", str(dst)]
        _run(cmd)
    else:
        shutil.unpack_archive(str(src), str(dst))


def find_root(base: Path, spec: Spec) -> Path:
    """Shallowest directory under ``base`` holding one of the spec's markers; link misplaced sibling folders."""
    if not spec.markers:
        return base
    best: Optional[Tuple[int, Path]] = None
    for marker in spec.markers:
        parts = Path(marker).parts
        for hit in base.rglob(parts[-1]):
            if tuple(hit.parts[-len(parts):]) != parts:
                continue
            root = Path(*hit.parts[:-len(parts)])
            depth = len(root.relative_to(base).parts)
            if best is None or depth < best[0]:
                best = (depth, root)
    if best is None and spec.alias[0]:  # e.g. BDD100K's per-split image zips unpacked without bdd100k/images/
        hits = sorted((p for p in base.rglob(spec.alias[0]) if p.is_dir() and any(c.is_dir() for c in p.iterdir())),
                      key=lambda p: len(p.parts))
        if hits:
            link = base / spec.alias[1]
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                os.symlink(hits[0].resolve(), link)
            LOG.info("linked %s -> %s", link, hits[0])
            best = (0, base)
    if best is None:
        raise FileNotFoundError(f"none of {list(spec.markers)} found under {base}: is this the right download?")
    root = best[1]
    for sib in spec.siblings:
        if (root / sib).exists():
            continue
        cands = sorted((p for p in base.rglob(sib) if p.is_dir() and p.parent != root), key=lambda p: len(p.parts))
        if cands:
            os.symlink(cands[0].resolve(), root / sib)
            LOG.info("linked %s -> %s", root / sib, cands[0])
    return root


def _converter_hash(source: str) -> str:
    from sparc.data import converters

    mod = converters.get(source)
    h = hashlib.sha1(Path(mod.__file__).read_bytes())
    for dep in ("base.py",):
        h.update((Path(mod.__file__).parent / dep).read_bytes())
    h.update((Path(mod.__file__).parents[1] / "schema.py").read_bytes())
    return h.hexdigest()[:12]


def cache_key(found: Found, args: Sequence[str]) -> str:
    items = [(p.name, p.stat().st_size) for parts in found.archives for p in parts] if not found.unpacked else [str(found.folder)]
    blob = json.dumps({"items": items, "args": list(args), "code": _converter_hash(found.source)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


@dataclass
class Prepared:
    source: str
    converted: Path
    frames: Dict[str, int]
    signs: int
    from_cache: bool
    seconds: float


def prepare(datasets_dir: str | Path, work: str | Path, cache_dir: Optional[str | Path] = None,
            sources: Sequence[str] | str = "auto", jobs: int = 4, margin_gb: float = 20.0,
            extra_args: Optional[Dict[str, List[str]]] = None) -> Dict[str, Prepared]:
    work = Path(work)
    raw_dir, conv_dir, tmp = work / "raw", work / "converted", work / "tmp"
    for d in (raw_dir, conv_dir, tmp):
        d.mkdir(parents=True, exist_ok=True)
    found = discover(datasets_dir, sources)
    if not found:
        raise FileNotFoundError(f"no dataset folder found in {datasets_dir} (expected e.g. {datasets_dir}/bdd100k/*.zip)")
    order = sorted(found.values(), key=lambda f: SOURCES[f.source].priority)
    extra_args = extra_args or {}
    # ---- 1-2: unpack (sequential: disk accounting) and locate roots
    roots: Dict[str, Path] = {}
    for f in order:
        spec = SOURCES[f.source]
        dst = raw_dir / f.source
        marker = dst / ".unpacked.json"
        recipe = {"items": [[p.name for p in parts] for parts in f.archives], "unpacked": f.unpacked}
        if f.unpacked:
            if not dst.exists():
                os.symlink(f.folder.resolve(), dst)  # reading through the mount: slower, but no copy
        elif not (marker.exists() and json.loads(marker.read_text()) == recipe):
            need = f.bytes * (spec.expand + 0.2)
            have = free_bytes(dst) - margin_gb * GB
            if need > have:
                LOG.warning("%s: skipped, needs ~%.0f GB, %.0f GB free (after a %.0f GB margin)", f.source, need / GB,
                            max(have, 0) / GB, margin_gb)
                continue
            shutil.rmtree(dst, ignore_errors=True)
            t0 = time.time()
            for parts in f.archives:
                LOG.info("%s: unpacking %s (%.1f GB)", f.source, parts[0].name, sum(p.stat().st_size for p in parts) / GB)
                unpack(parts, dst, tmp / f.source)
            marker.write_text(json.dumps(recipe))
            LOG.info("%s: unpacked in %.0f s", f.source, time.time() - t0)
        try:
            roots[f.source] = find_root(dst, spec)
        except FileNotFoundError as e:
            LOG.error("%s: %s", f.source, e)
    # ---- 3: convert (parallel) or restore from the cache
    results: Dict[str, Prepared] = {}

    def one(source: str) -> Prepared:
        t0 = time.time()
        args = list(SOURCES[source].args) + list(extra_args.get(source, []))
        key = cache_key(found[source], args)
        dst = conv_dir / source
        done = dst / ".converted.json"
        cached = Path(cache_dir) / f"{source}-{key}.tar" if cache_dir else None
        from_cache = False
        if done.exists() and json.loads(done.read_text()).get("key") == key:
            from_cache = True
        elif cached is not None and cached.exists():
            shutil.rmtree(dst, ignore_errors=True)
            conv_dir.mkdir(parents=True, exist_ok=True)
            _run(["tar", "-xf", str(cached), "-C", str(conv_dir)])
            from_cache = True
        else:
            log = work / f"convert_{source}.log"
            log.unlink(missing_ok=True)
            _run([sys.executable, "-m", "sparc", "convert", source, "--", "--root", str(roots[source]), "--dst", str(dst),
                  "--overwrite", *args], log=log)
            (dst / ".converted.json").write_text(json.dumps({"key": key}))
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                part = cached.with_suffix(".tar.part")
                _run(["tar", "-cf", str(part), "-C", str(conv_dir), source])
                part.replace(cached)
        (dst / ".converted.json").write_text(json.dumps({"key": key}))
        m = json.loads((dst / "manifest.json").read_text())
        return Prepared(source, dst, dict(m.get("splits", {})), len(m.get("signs", [])), from_cache, time.time() - t0)

    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(one, s): s for s in roots}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            try:
                results[s] = fut.result()
                r = results[s]
                LOG.info("%s: %s %s in %.0f s", s, "restored" if r.from_cache else "converted", r.frames, r.seconds)
            except Exception as e:  # one broken source must not stop the others
                LOG.error("%s: conversion failed: %s", s, e)
    return results


def write_manifests(prepared: Dict[str, Prepared], out_dir: str | Path, train_limit: Optional[int] = None,
                    val_limit: Optional[int] = None, weights: Optional[Dict[str, float]] = None) -> Tuple[Path, Path]:
    """``data_train.yaml`` (training sources + limited validation) and ``data_val_full.yaml`` (all val frames)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, val_full = [], [], []
    w = {s: (weights or {}).get(s, SOURCES[s].weight) for s in prepared}
    total = sum(w[s] for s, p in prepared.items() if p.frames.get("train"))
    for s, p in sorted(prepared.items(), key=lambda kv: SOURCES[kv[0]].priority):
        if p.frames.get("train"):
            e = {"root": str(p.converted), "weight": round(w[s] / total, 4)}
            if train_limit:
                e["limit"] = int(train_limit)
            train.append(e)
        if p.frames.get("val") and SOURCES[s].val:
            e = {"root": str(p.converted)}
            val_full.append(dict(e))
            if val_limit:
                e["limit"] = int(val_limit)
            val.append(e)
    if not train:
        raise ValueError("no training frames in any prepared source")
    a, b = out_dir / "data_train.yaml", out_dir / "data_val_full.yaml"
    a.write_text(yaml.safe_dump({"train": train, "val": val}, sort_keys=False))
    b.write_text(yaml.safe_dump({"train": train, "val": val_full}, sort_keys=False))
    return a, b


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sparc prepare", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", required=True, help="folder with one sub-folder of archives per source")
    p.add_argument("--work", required=True, help="local working directory (raw/, converted/, manifests)")
    p.add_argument("--cache", default=None, help="folder for cached conversions (e.g. on Google Drive)")
    p.add_argument("--sources", default="auto", help="'auto' or a comma list")
    p.add_argument("--jobs", type=int, default=4, help="conversions in parallel")
    p.add_argument("--train-limit", type=int, default=None, help="frames per training source (evenly spaced)")
    p.add_argument("--val-limit", type=int, default=None, help="frames per source for in-training validation")
    p.add_argument("--margin-gb", type=float, default=20.0, help="disk space to keep free")
    a = p.parse_args(argv)
    res = prepare(a.datasets, a.work, a.cache, a.sources, a.jobs, a.margin_gb)
    t, v = write_manifests(res, a.work, a.train_limit, a.val_limit)
    print(json.dumps({s: {"frames": r.frames, "signs": r.signs, "cached": r.from_cache} for s, r in res.items()}, indent=1))
    print(f"manifests: {t} {v}")
    return 0
