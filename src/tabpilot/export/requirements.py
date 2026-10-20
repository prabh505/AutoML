"""Pinned requirements for a bundle (PRD §22.2; FR-ART-01).

Exact versions are read from the installed distributions (`importlib.metadata`), never guessed.
The list covers what `model.joblib` and `pipeline.py` import: numpy, pandas, scikit-learn,
SciPy and joblib always, plus the model libraries the pipeline actually uses.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

BASE = ("numpy", "pandas", "scikit-learn", "scipy", "joblib")
LIBRARY_DISTRIBUTIONS = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}


def model_libraries(model: Any) -> set[str]:
    """Top-level library modules of every object in the estimator tree."""
    found: set[str] = set()
    stack = [model]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        top = (type(obj).__module__ or "").split(".")[0]
        if top in LIBRARY_DISTRIBUTIONS:
            found.add(top)
        if isinstance(obj, list | tuple):
            stack.extend(obj)
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif callable(getattr(obj, "get_params", None)) and not isinstance(obj, type):
            stack.extend(obj.get_params(deep=False).values())
    return found


def pinned(distributions: Iterable[str]) -> list[str]:
    """`name==version` lines, sorted; a missing distribution is an error (never unpinned)."""
    out = []
    for name in sorted(set(distributions)):
        try:
            out.append(f"{name}=={version(name)}")
        except PackageNotFoundError as exc:
            raise RuntimeError(f"{name} is not installed; cannot pin it") from exc
    return out


def requirements_for(model: Any) -> list[str]:
    """Pinned lines for `model` and `pipeline.py`."""
    libraries = {LIBRARY_DISTRIBUTIONS[m] for m in model_libraries(model)}
    return pinned([*BASE, *libraries])
