"""The fine traffic-sign vocabulary: the union of the sign names written by the converters.

Names are prefixed with their source (``mtsd:regulatory--stop--g1``), so classes of different datasets are distinct
unless an alias table merges them (:meth:`SignVocab.with_aliases`); each source is masked to its own names by the
label mask, so an un-merged duplicate only costs a little data, never a wrong negative.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_LEFT_RIGHT = re.compile(r"(left|right)", re.IGNORECASE)


class SignVocab:
    def __init__(self, names: Iterable[str] = (), aliases: Optional[Dict[str, str]] = None):
        self.aliases = dict(aliases or {})
        self.names: List[str] = sorted({self.aliases.get(n, n) for n in names if n})
        self.index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def id(self, name: str) -> int:
        """-1 for "" (unlabelled) and for names the vocabulary does not know."""
        if not name:
            return -1
        return self.index.get(self.aliases.get(name, name), -1)

    def flip(self, name: str) -> str:
        """Name of the mirrored sign: left <-> right when that sign exists, the same name when it has no chirality,
        "" (unlabelled) when the mirrored class is unknown."""
        if not name or not _LEFT_RIGHT.search(name):
            return name
        swapped = _LEFT_RIGHT.sub(lambda m: {"left": "right", "right": "left", "Left": "Right", "Right": "Left",
                                             "LEFT": "RIGHT", "RIGHT": "LEFT"}[m.group(1)], name)
        return swapped if self.id(swapped) >= 0 else ""

    @classmethod
    def from_manifests(cls, roots: Iterable[str | Path], aliases: Optional[Dict[str, str]] = None) -> "SignVocab":
        names = set()
        for r in roots:
            m = Path(r) / "manifest.json"
            if m.exists():
                names.update(json.loads(m.read_text()).get("signs", []))
        return cls(names, aliases)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"names": self.names, "aliases": self.aliases}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "SignVocab":
        d = json.loads(Path(path).read_text())
        return cls(d["names"], d.get("aliases"))
