"""Dataset converters: native formats -> the SPARC sample schema (``sparc/data/schema.py``).

Each converter is a module with ``add_args(parser)`` and ``convert(args) -> ConvertStats``; ``REGISTRY`` maps the
``python -m sparc convert <source>`` name to the module. They are validated on format-faithful mini dumps
(``tests/test_sparc/test_converters*.py``): on a real download, run with ``--limit 200`` first and read the summary
(skipped records are counted by reason, never dropped silently).
"""

from __future__ import annotations

import importlib

REGISTRY = {
    "bdd100k": "sparc.data.converters.bdd100k",
    "culane": "sparc.data.converters.culane",
    "curvelanes": "sparc.data.converters.curvelanes",
    "openlane": "sparc.data.converters.openlane",
    "mtsd": "sparc.data.converters.mtsd",
    "tt100k": "sparc.data.converters.tt100k",
    "zod": "sparc.data.converters.zod",
    "dtld": "sparc.data.converters.dtld",
    "bosch_tl": "sparc.data.converters.bosch_tl",
    "lisa_tl": "sparc.data.converters.lisa_tl",
    "nuimages": "sparc.data.converters.nuimages",
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown converter {name!r}; choose from {sorted(REGISTRY)}")
    return importlib.import_module(REGISTRY[name])
