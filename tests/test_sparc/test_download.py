"""download: resumable HTTP with MD5, Google Drive via gdown, skip when complete, disk check; prepare's BDD alias."""

import hashlib
import http.server
import json
import os
import stat
import threading
import zipfile
from functools import partial
from pathlib import Path

import pytest

from sparc.tools import download as D
from sparc.tools import prepare as P


@pytest.fixture()
def server(tmp_path, monkeypatch):
    root = tmp_path / "www"
    root.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


def _catalog(url, md5=True):
    return {"bdd100k": [D.Item("a.zip", url + "a.zip", 0.001, md5_url=(url + "a.zip.md5") if md5 else None),
                        D.Item("b.zip", url + "b.zip", 0.001)]}


def test_http_md5_and_skip_when_complete(server, tmp_path, monkeypatch):
    root, url = server
    (root / "a.zip").write_bytes(b"A" * 5000)
    (root / "a.zip.md5").write_text(hashlib.md5(b"A" * 5000).hexdigest() + "  a.zip\n")
    (root / "b.zip").write_bytes(b"B" * 300)
    monkeypatch.setattr(D, "CATALOG", _catalog(url))
    dest = tmp_path / "datasets"
    res = D.download(dest, "bdd100k", margin_gb=0)
    assert res == {"bdd100k": ["a.zip", "b.zip"]}
    assert (dest / "bdd100k" / "a.zip").read_bytes() == b"A" * 5000
    assert not list((dest / "bdd100k").glob("*.part")) and not list((dest / "bdd100k").glob("*.md5"))
    calls = []
    monkeypatch.setattr(D, "_run", lambda cmd: calls.append(cmd) or 1)
    assert D.download(dest, "bdd100k", margin_gb=0) == res and calls == []  # second session: nothing fetched
    # prepare sees the source with its archives
    assert [p[0].name for p in P.discover(dest)["bdd100k"].archives] == ["a.zip", "b.zip"]


def test_md5_mismatch_removes_partial_and_reports(server, tmp_path, monkeypatch):
    root, url = server
    (root / "a.zip").write_bytes(b"corrupt")
    (root / "a.zip.md5").write_text("0" * 32 + "\n")
    (root / "b.zip").write_bytes(b"B")
    monkeypatch.setattr(D, "CATALOG", _catalog(url))
    with pytest.raises(RuntimeError, match="MD5"):
        D.download(tmp_path / "d", "bdd100k", margin_gb=0)
    folder = tmp_path / "d" / "bdd100k"
    assert not (folder / "a.zip").exists() and not (folder / "a.zip.part").exists()
    assert (folder / "b.zip").exists()  # the other file of the source still came
    assert json.loads((folder / ".complete.json").read_text()) == {"b.zip": "b.zip"}


def test_gdrive_uses_original_name_and_reports_quota(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gdown"
    fake.write_text("#!/bin/sh\n# args: --continue -O <dir>/ <id>\n"
                    "if [ \"$4\" = BAD ]; then echo 'Too many users have viewed or downloaded this file' >&2; exit 1; fi\n"
                    "echo data > \"$3/driver_$4.tar.gz\"\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(D, "CATALOG", {"culane": [D.Item("t1", "ID1", 0.001, kind="gdrive"),
                                                   D.Item("t2", "BAD", 0.001, kind="gdrive"),
                                                   D.Item("x", "ID3", 0.001, kind="gdrive", optional="culane_test")]})
    with pytest.raises(RuntimeError, match=r"too many users.*drive.google.com/file/d/BAD"):
        D.download(tmp_path / "d", "culane", margin_gb=0)
    folder = tmp_path / "d" / "culane"
    assert sorted(p.name for p in folder.glob("*.tar.gz")) == ["driver_ID1.tar.gz"]  # the test archive is opt-in
    assert json.loads((folder / ".complete.json").read_text()) == {"t1": "driver_ID1.tar.gz"}


def test_disk_check_and_unknown_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "free_bytes", lambda p: 1)
    with pytest.raises(RuntimeError, match="needs ~"):
        D.download(tmp_path, "tt100k")
    with pytest.raises(ValueError, match="mapillary"):
        D.download(tmp_path, "mtsd")


def test_prepare_links_bdd_images_unpacked_without_their_prefix(tmp_path):
    """ETH's per-split image zips may unpack as 100k/<split>/ next to bdd100k/labels/: the root is still found."""
    base = tmp_path / "raw"
    (base / "100k" / "train").mkdir(parents=True)
    (base / "100k" / "train" / "x.jpg").write_bytes(b"j")
    (base / "bdd100k" / "labels" / "det_20").mkdir(parents=True)
    root = P.find_root(base, P.SOURCES["bdd100k"])
    assert (root / "images" / "100k" / "train" / "x.jpg").exists() and (root / "labels" / "det_20").is_dir()
