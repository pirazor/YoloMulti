"""Sidecar metadata of exported artifacts (light module: no torch / ultralytics, so a Jetson runner can import it)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def meta_path(model_path: str | Path) -> Path:
    """``model.onnx`` -> ``model.json``; any other artifact (``model.engine``) -> ``model.engine.json``."""
    p = Path(model_path)
    return p.with_suffix(".json") if p.suffix == ".onnx" else p.with_name(p.name + ".json")


def read_meta(model_path: str | Path) -> Dict[str, Any]:
    """Sidecar JSON; an ``.onnx`` without one falls back to the copy embedded in the graph, an engine to the JSON of
    the ONNX it was built from (same stem)."""
    p = Path(model_path)
    sidecar = meta_path(p)
    for cand in (sidecar, p.with_suffix(".json")):
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    if p.suffix == ".onnx" and p.is_file():
        try:
            import onnx
        except ImportError as e:
            raise FileNotFoundError(f"no metadata for {p}: expected {sidecar}. Copy the .json next to the model (the copy "
                                    "embedded in the ONNX can only be read with the `onnx` package)") from e
        for prop in onnx.load(str(p), load_external_data=False).metadata_props:
            if prop.key == "adas_mt":
                return json.loads(prop.value)
    raise FileNotFoundError(f"no metadata for {p}: expected {sidecar} (written by `adas_mt export` / `adas_mt trt-build`)")
