"""Python source for an unfitted copy of a fitted pipeline (PRD §22.4; FR-ART-02).

`estimator_source(model)` walks the estimator tree through `get_params(deep=False)` and emits
constructor calls with every parameter: the column map, type rules, DSL specs, preprocessing
`ColumnTransformer`, the model constructor with its final hyperparameters, the target transformer
and the ensemble wrapper. Every literal is written with `repr()`, so user-derived strings such
as column names are always inert Python string literals (never code). The generated code is a
fixpoint: running it and generating again gives the same text (tested).

Nothing here comes from an LLM; it is deterministic code generation from fitted objects.
"""

from __future__ import annotations

import inspect
import math
import sys
import types
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from tabpilot_runtime import VotingEnsemble

INDENT = "    "
IMPORT_WIDTH = 88  # ruff/isort default line length
RUNTIME = "tabpilot_runtime"
NUMPY_ALIAS = "np"


class CodegenError(ValueError):
    """An object in the pipeline cannot be written as source code."""


@dataclass
class Imports:
    """`from module import name` lines needed by the generated code."""

    names: dict[str, set[str]] = field(default_factory=dict)
    numpy: bool = False

    def add(self, module: str, name: str) -> None:
        self.names.setdefault(module, set()).add(name)

    def lines(self, indent: str = "", *, modules: Iterable[str] | None = None) -> list[str]:
        """isort-style `from` lines: sorted, wrapped one name per line past 88 columns. `indent`
        is the block's indentation (counted for wrapping; continuation lines carry it).
        """
        out = []
        for module, names in sorted(self.names.items()):
            if modules is not None and module not in modules:
                continue
            line = f"from {module} import {', '.join(sorted(names))}"
            if len(indent) + len(line) > IMPORT_WIDTH:
                body = "".join(f"{indent}{INDENT}{n},\n" for n in sorted(names))
                line = f"from {module} import (\n{body}{indent})"
            out.append(line)
        return out


def _public_module(obj: Any) -> str:
    """The shortest importable module path that exposes `obj` under its own name."""
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", getattr(obj, "__name__", None))
    if not module or not name or "." in name:
        raise CodegenError(f"cannot import {obj!r}")
    # Only modules that are already imported are inspected (no dynamic import, T-SEC-01): the
    # parents of an imported module are always in `sys.modules`.
    parts = module.split(".")
    for i in range(1, len(parts) + 1):
        candidate = sys.modules.get(".".join(parts[:i]))
        if candidate is not None and vars(candidate).get(name) is obj:
            return ".".join(parts[:i])
    raise CodegenError(f"{module}.{name} is not importable")


def _float(value: float) -> str:
    if math.isnan(value):
        return 'float("nan")'
    if math.isinf(value):
        return 'float("inf")' if value > 0 else 'float("-inf")'
    return repr(value)


def _ctor_params(obj: Any) -> dict[str, Any]:
    if isinstance(obj, BaseEstimator) or callable(getattr(obj, "get_params", None)):
        params: dict[str, Any] = dict(obj.get_params(deep=False))  # CatBoost is not sklearn's
        return params
    signature = inspect.signature(type(obj).__init__)
    out = {}
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        stored = vars(obj)
        if name not in stored:
            raise CodegenError(f"{type(obj).__name__}.{name} is not stored on the object")
        out[name] = stored[name]
    return out


class _Writer:
    def __init__(self) -> None:
        self.imports = Imports()

    def value(self, v: Any, depth: int) -> str:  # noqa: PLR0911 - one branch per kind
        if v is None or isinstance(v, bool | str | bytes):
            return repr(v)
        if isinstance(v, np.generic):
            self.imports.numpy = True
            inner = _float(float(v)) if isinstance(v, np.floating) else repr(v.item())
            return f"{NUMPY_ALIAS}.{type(v).__name__}({inner})"
        if isinstance(v, int):
            return repr(v)
        if isinstance(v, float):
            return _float(v)
        if isinstance(v, list | tuple):
            return self.sequence(v, depth)
        if isinstance(v, dict):
            return self.mapping(v, depth)
        if isinstance(v, np.ufunc) or (isinstance(v, type) and v.__module__ == "numpy"):
            self.imports.numpy = True
            return f"{NUMPY_ALIAS}.{v.__name__}"
        if isinstance(v, type | types.FunctionType | types.BuiltinFunctionType):
            self.imports.add(_public_module(v), v.__name__)
            return v.__name__
        if isinstance(v, np.ndarray):
            self.imports.numpy = True
            return f"{NUMPY_ALIAS}.asarray({self.sequence(v.tolist(), depth)})"
        return self.constructor(v, depth)

    def sequence(self, v: list[Any] | tuple[Any, ...], depth: int) -> str:
        open_, close = ("[", "]") if isinstance(v, list) else ("(", ")")
        if not v:
            return "[]" if isinstance(v, list) else "()"
        pad = INDENT * (depth + 1)
        items = [self.value(x, depth + 1) for x in v]
        trailing = "," if isinstance(v, tuple) and len(v) == 1 else ""
        flat = f"{open_}{', '.join(items)}{trailing}{close}"
        if len(flat) <= 80 and "\n" not in flat:
            return flat
        body = "".join(f"{pad}{item},\n" for item in items)
        return f"{open_}\n{body}{INDENT * depth}{close}"

    def mapping(self, v: dict[Any, Any], depth: int) -> str:
        if not v:
            return "{}"
        pad = INDENT * (depth + 1)
        items = [f"{self.value(k, depth + 1)}: {self.value(x, depth + 1)}" for k, x in v.items()]
        flat = "{" + ", ".join(items) + "}"
        if len(flat) <= 80 and "\n" not in flat:
            return flat
        body = "".join(f"{pad}{item},\n" for item in items)
        return "{\n" + body + INDENT * depth + "}"

    def constructor(self, obj: Any, depth: int) -> str:
        cls = type(obj)
        self.imports.add(_public_module(cls), cls.__name__)
        params = _ctor_params(obj)
        if isinstance(obj, VotingEnsemble):
            params = {**params, "prefit": False}  # retraining fits the members (pipeline.py train)
        pad = INDENT * (depth + 1)
        if params:
            body = "".join(f"{pad}{k}={self.value(v, depth + 1)},\n" for k, v in params.items())
            call = f"{cls.__name__}(\n{body}{INDENT * depth})"
        else:
            call = f"{cls.__name__}()"
        config = getattr(obj, "_sklearn_output_config", {}) or {}
        if "transform" in config:
            call += f".set_output(transform={config['transform']!r})"
        return call


@dataclass(frozen=True)
class GeneratedSource:
    """`expression` builds an unfitted copy of the estimator."""

    expression: str
    imports: list[str]
    runtime_imports: list[str]
    numpy: bool

    @property
    def all_imports(self) -> list[str]:
        """Every import line, for running the expression on its own."""
        return (["import numpy as np"] if self.numpy else []) + self.imports + self.runtime_imports


def estimator_source(model: Any, depth: int = 1, runtime_indent: str = "") -> GeneratedSource:
    """Source code for an unfitted copy of `model` (fitted attributes are not written).

    `imports` are third-party `from` lines; `runtime_imports` are the `tabpilot_runtime` ones,
    wrapped for a block indented by `runtime_indent`; `numpy` says whether `np` is used.
    """
    writer = _Writer()
    expression = writer.value(model, depth)
    third = [m for m in writer.imports.names if m.split(".")[0] != RUNTIME]
    first = [m for m in writer.imports.names if m.split(".")[0] == RUNTIME]
    return GeneratedSource(expression, writer.imports.lines(modules=third),
                           writer.imports.lines(runtime_indent, modules=first),
                           writer.imports.numpy)  # fmt: skip
