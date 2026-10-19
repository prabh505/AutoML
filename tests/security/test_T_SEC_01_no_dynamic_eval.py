"""T-SEC-01 (NFR-SEC-01, FR-FE-02, §24.6): no dynamic code evaluation anywhere in `src/`.

Fails on calls to the builtins `eval`, `exec`, `compile`, `__import__`; on any `.query(` or
`.eval(` method call (covers `pd.eval`, `DataFrame.query`, `DataFrame.eval`); and on
`importlib.import_module` / `getattr` with a non-literal name.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
BANNED_BUILTINS = {"eval", "exec", "compile", "__import__"}
BANNED_METHODS = {"eval", "query"}
LITERAL_ONLY = {"import_module", "getattr"}


def findings(root: Path = SRC) -> list[str]:
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_BUILTINS:
                found.append(f"{rel}:{node.lineno} {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in BANNED_METHODS:
                found.append(f"{rel}:{node.lineno} .{func.attr}()")
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else "")  # fmt: skip
            if name in LITERAL_ONLY:
                arg = node.args[1] if name == "getattr" and len(node.args) > 1 else (
                    node.args[0] if node.args else None)  # fmt: skip
                if arg is not None and not isinstance(arg, ast.Constant):
                    found.append(f"{rel}:{node.lineno} {name}() with a non-literal name")
    return found


@pytest.mark.security
def test_T_SEC_01_no_dynamic_evaluation_in_src() -> None:
    assert findings() == []


@pytest.mark.security
@pytest.mark.parametrize("code", [
    "eval(spec)", "exec(text)", "compile(s, 'x', 'exec')", "__import__(name)",
    "import pandas as pd\npd.eval(expr)", "df.query(spec['expr'])", "df.eval(user)",
    "import importlib\nimportlib.import_module(name)", "getattr(obj, spec['op'])",
])  # fmt: skip
def test_scanner_catches_each_pattern(code: str, tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(code + "\n", encoding="utf-8")
    assert len(findings(tmp_path)) == 1


@pytest.mark.security
def test_scanner_allows_safe_lookalikes(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text(
        "import re, importlib\nre.compile('x')\nimportlib.import_module('json')\n"
        "getattr(obj, 'name')\n", encoding="utf-8")  # fmt: skip
    assert findings(tmp_path) == []
