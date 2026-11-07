"""P15 end-to-end: the CLI through `RunManager` in a real worker process (PRD §31.3).

T-E2E-01…03: binary, multiclass and regression runs with the stub LLM → a bundle that passes R1
in isolation. A `--llm off` run covers the LLM-free product. T-E2E-04 (live provider) is opt-in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.helpers.bundles import isolated, unzip
from tests.helpers.run_harness import FIXTURES

from tabpilot.cli import main
from tabpilot.schemas.report import ArtifactManifest, FinalReport

CASES = [
    pytest.param("binary_basic.csv", "Churn", "binary", id="T-E2E-01-binary"),
    pytest.param("multiclass_basic.csv", "species", "multiclass", id="T-E2E-02-multiclass"),
    pytest.param("regression_basic.csv", "price", "regression", id="T-E2E-03-regression"),
]


def run_cli(tmp_path: Path, file: str, target: str, *extra: str) -> tuple[int, Path]:
    out = tmp_path / "run"
    code = main(["run", str(FIXTURES / file), "--target", target, "--out", str(out),
                 "--budget-s", "240", "--tuning-mode", "trials", "--trials", "2",
                 "--models", "linear,lightgbm", *extra])  # fmt: skip
    return code, out


def verify_bundle(out: Path, tmp_path: Path) -> ArtifactManifest:
    (bundle,) = out.glob("tabpilot_*.zip")
    root = unzip(bundle, tmp_path / "clean")
    manifest = ArtifactManifest.model_validate_json((root / "manifest.json").read_text("utf-8"))
    assert manifest.reproducibility.r1_verified
    proc = isolated(root, "verify")
    assert proc.returncode == 0, proc.stderr
    assert "tabpilot not importable" in proc.stdout
    assert "R1 parity PASSED: 50 rows" in proc.stdout
    return manifest


@pytest.mark.e2e
@pytest.mark.parametrize(("file", "target", "task"), CASES)
def test_T_E2E_stub_llm_run_produces_a_verified_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], file: str, target: str, task: str
) -> None:
    code, out = run_cli(tmp_path, file, target, "--llm", "stub")
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "status: completed" in printed
    assert "bundle:" in printed
    manifest = verify_bundle(out, tmp_path)
    paths = {f.path for f in manifest.files}
    assert {"model/model.joblib", "pipeline.py", "report/report.html", "tabpilot_runtime/__init__.py",
            "metadata/plan.json", "metadata/llm_calls.jsonl"} <= paths  # fmt: skip
    report = FinalReport.model_validate_json((out / "report" / "report.json").read_text("utf-8"))
    assert report.sections.narrative.source == "llm"
    assert report.sections.narrative.fact_check == "passed"
    experiment = json.loads((out / "records" / "experiment.json").read_text("utf-8"))
    assert experiment["config"]["task"] == task
    assert experiment["plan_source"] == "llm"
    assert not (out / "input").exists()  # the raw input is deleted after export (§24.7)


@pytest.mark.e2e
def test_T_E2E_llm_off_run_makes_no_llm_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_cli(tmp_path, "binary_basic.csv", "Churn", "--llm", "off")
    assert code == 0, capsys.readouterr().out
    manifest = verify_bundle(out, tmp_path)
    assert "metadata/llm_calls.jsonl" not in {f.path for f in manifest.files}
    report = FinalReport.model_validate_json((out / "report" / "report.json").read_text("utf-8"))
    assert report.sections.narrative.source == "template"


@pytest.mark.e2e
def test_cli_rejects_an_out_of_range_budget_before_creating_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_cli(tmp_path, "binary_basic.csv", "Churn", "--budget-s", "239")
    assert code == 2
    assert "outside 240" in capsys.readouterr().err
    assert not out.exists()  # rejected before anything was created


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="live provider key not set (KI-025)")
def test_T_E2E_04_live_provider_run_completes_or_falls_back(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, "binary_basic.csv", "Churn", "--llm", "providers",
                        "--providers", "groq")  # fmt: skip
    assert code == 0
    verify_bundle(out, tmp_path)
