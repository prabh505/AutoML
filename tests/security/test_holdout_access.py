"""L-2 / I-1 static guard: only the finalizer path may read `SplitSpec.holdout_indices`.

`SplitSpec` carries the holdout indices (FR-SPL-01) for persistence and `--resplit`. Import
contract R2 protects `splitting.guard`, but not this attribute, so this test does: an AST scan
of `src/` fails if any module outside the allowed set touches `.holdout_indices`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
ALLOWED = {
    "tabpilot/schemas/split.py",  # the contract itself
    "tabpilot/splitting/strategy.py",  # builds the split
    "tabpilot/splitting/guard.py",
    "tabpilot/evaluation/holdout.py",  # P09: one-time holdout evaluation
}
ALLOWED_PREFIXES = ("tabpilot/orchestration/",)  # P15: run context constructs the guard


def offenders(root: Path = SRC) -> list[str]:
    found = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in ALLOWED or rel.startswith(ALLOWED_PREFIXES):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "holdout_indices":
                found.append(f"{rel}:{node.lineno}")
            if isinstance(node, ast.Constant) and node.value == "holdout_indices":
                found.append(f"{rel}:{node.lineno} (string access)")
    return found


@pytest.mark.security
def test_only_the_finalizer_path_reads_holdout_indices() -> None:
    assert offenders() == []


@pytest.mark.security
def test_scanner_detects_a_violation(tmp_path: Path) -> None:
    pkg = tmp_path / "tabpilot" / "training"
    pkg.mkdir(parents=True)
    (pkg / "leak.py").write_text(
        "def f(spec):\n    return spec.holdout_indices\n", encoding="utf-8"
    )
    assert offenders(tmp_path) == ["tabpilot/training/leak.py:2"]
