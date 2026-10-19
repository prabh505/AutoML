"""Column-name normalization (PRD §8.4, FR-ING-03; PI-45).

`normalize_name` is deterministic and idempotent. `normalize_columns` additionally dedupes
(case-insensitively, since names are lowercased) with suffixes `__2`, `__3`, ….
Names are lowercased (PI-45), matching every example in the PRD.
"""

from __future__ import annotations

import keyword
import re
import unicodedata

MAX_LEN = 64
RESERVED = frozenset(k.lower() for k in keyword.kwlist) | {"target", "__index__"}
_WS = re.compile(r"\s+")
_BAD = re.compile(r"[^a-z0-9_]")
_UNDERSCORES = re.compile(r"_+")


def normalize_name(name: str, index: int) -> str:
    """Normalize one column name (§8.4). `index` is used only for empty names."""
    s = unicodedata.normalize("NFKC", str(name)).strip().lower()
    s = _WS.sub("_", s)
    s = _BAD.sub("_", s)
    s = _UNDERSCORES.sub("_", s).strip("_")
    if not s:
        s = f"col_{index}"
    if s[0].isdigit():
        s = "c_" + s
    s = s[:MAX_LEN].rstrip("_")
    if s in RESERVED:
        s += "_"
    return s


def normalize_columns(names: list[str]) -> list[str]:
    """Normalized, unique names in input order (§8.4). Works by position, so duplicate
    original names (e.g. two "Amount" headers) are kept apart.
    """
    out: list[str] = []
    seen: set[str] = set()
    for index, original in enumerate(names):
        base = normalize_name(original, index)
        candidate, n = base, 1
        while candidate in seen:
            n += 1
            candidate = f"{base}__{n}"
        seen.add(candidate)
        out.append(candidate)
    return out
