"""T-E2E-05 reliability suite: 54 runs over fixtures × seeds × stub failure modes (PRD §31.3).

Every run must either produce a bundle that reloads and passes R1 parity, or fail with an E-code.
The gate: ≥ 95% valid bundles, and 100% of the failures carry an E-code (SM-9 style).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import joblib
import pytest
from tests.helpers.bundles import unzip
from tests.helpers.run_harness import go

from tabpilot.llm.stub import MODES
from tabpilot.schemas import ModelName
from tabpilot_runtime.bundle import verify

FIXTURES = [("binary_small.csv", "target"), ("multiclass_basic.csv", "species"),
            ("regression_basic.csv", "price")]  # fmt: skip
SEEDS = (1, 2, 3)
MIN_VALID_SHARE = 0.95


@pytest.mark.slow
def test_T_E2E_05_reliability_suite(tmp_path: Path) -> None:
    runs = list(itertools.product(FIXTURES, SEEDS, MODES))
    assert len(runs) >= 50
    valid, failures, report = 0, 0, []
    for index, ((file, target), seed, mode) in enumerate(runs):
        work = tmp_path / f"r{index}"
        work.mkdir()
        outcome = go(work, file, target, llm="stub", stub_mode=mode, seed=seed, trials=1,
                     models=(ModelName.LINEAR,))  # fmt: skip
        result = outcome.result
        if result.status in ("completed", "completed_partial"):
            assert result.final is not None
            root = unzip(outcome.run_dir / str(result.final.bundle_path), work / "u")
            parity = verify(root, joblib.load(root / "model" / "model.joblib"))
            if parity.ok:
                valid += 1
                continue
            report.append((file, seed, mode, "R1 parity failed"))
        failures += 1
        report.append((file, seed, mode, str(result.error_code)))
        assert result.error_code is not None, (file, seed, mode, result)  # 100% coded
    share = valid / len(runs)
    print(f"reliability: {valid}/{len(runs)} valid bundles, {failures} coded failures")  # noqa: T201
    assert share >= MIN_VALID_SHARE, (share, report)
