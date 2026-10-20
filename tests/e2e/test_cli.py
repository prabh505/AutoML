"""P09 CLI end-to-end tests on the task fixtures, `trials` mode (FR-PLN-03, FR-EVL-01/02)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tabpilot import cli
from tabpilot.cli import RunArgs, execute, main, parse
from tabpilot.schemas import (
    ExecutionProfileName,
    FeasibilityReport,
    ModelName,
    RunEvent,
    Simplification,
    Task,
)
from tabpilot.schemas.experiment import EvaluationResult, ExperimentConfig, LeaderboardEntry
from tabpilot.schemas.report import FinalReport
from tabpilot.splitting.guard import HoldoutSealed
from tabpilot.tracking import RECORD_FILES

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "datasets"
CASES = [("binary_basic.csv", "churn", Task.BINARY, True),
         ("multiclass_basic.csv", "species", Task.MULTICLASS, True),
         ("regression_basic.csv", "price", Task.REGRESSION, True),
         ("binary_small.csv", "target", Task.BINARY, False)]  # fmt: skip


def run(tmp_path: Path, name: str, target: str, **kw: Any) -> Any:
    args = RunArgs(file=FIX / name, target=target, out=tmp_path / "run", tuning_mode="trials",
                   trials=2, models=(ModelName.LINEAR, ModelName.LIGHTGBM), **kw)  # fmt: skip
    return execute(args)


def events(run_dir: Path) -> list[RunEvent]:
    lines = (run_dir / "records" / "events.jsonl").read_text().splitlines()
    return [RunEvent.model_validate_json(line) for line in lines]


def check_report_and_bundle(summary: Any, name: str, task: Task) -> None:
    """P13/P14 outputs of every run: the report and a verified bundle."""
    out = summary.run_dir
    # P13: explanations and the report are part of every run (FR-EXP-01, FR-RPT-01)
    report = FinalReport.model_validate_json((out / "report" / "report.json").read_text())
    assert report.sections.narrative.source == "template"
    assert report.sections.narrative.fact_check == "passed"
    assert report.sections.explanations.method in ("shap_tree", "shap_linear")
    assert report.sections.explanations.top_features
    html = (out / "report" / "report.html").read_text()
    assert html.count("<h2>") == 17
    assert (out / "report" / "report.md").exists()
    assert (out / "report" / "figures" / "leaderboard.png").exists()
    assert ("roc-curve.png" in html) == (task == Task.BINARY)
    assert ("residual-histogram.png" in html) == (task == Task.REGRESSION)
    stages = json.loads((out / "records" / "stage_timings.json").read_text())
    assert {"explain", "report"} <= set(stages)
    assert any(e.type.value == "report_rendered" for e in events(out))
    # P14: every run offers a verified bundle (FR-ART-01)
    assert summary.bundle is not None
    assert summary.bundle.parent == out
    assert summary.bundle.name.startswith(f"tabpilot_{Path(name).stem}_")
    exported = [e for e in events(out) if e.type.value == "bundle_exported"]
    assert len(exported) == 1
    assert exported[0].data["sha256"] == summary.bundle_sha256
    assert exported[0].data["parity_max_abs_diff"] <= 1e-9
    assert "export" in stages


@pytest.mark.e2e
@pytest.mark.parametrize(("name", "target", "task", "holdout"), CASES)
def test_cli_end_to_end(tmp_path: Path, name: str, target: str, task: Task, holdout: bool) -> None:
    summary = run(tmp_path, name, target)
    out = summary.run_dir
    assert summary.task == task
    for record in RECORD_FILES:
        assert (out / "records" / record).exists(), record
    experiment = json.loads((out / "records" / "experiment.json").read_text())
    config = ExperimentConfig.model_validate(experiment["config"])
    evaluation = EvaluationResult.model_validate(experiment["evaluation"])
    assert config.task == task
    assert config.llm.enabled is False
    assert experiment["lineage"]["plan_source"] == "rule_based"
    # T-EVL-01 on the written leaderboard: stats recompute from stored fold scores
    board = [
        LeaderboardEntry.model_validate(e)
        for e in json.loads((out / "leaderboard.json").read_text())
    ]
    assert {e.model for e in board} == {ModelName.LINEAR, ModelName.LIGHTGBM}
    for e in board:
        assert e.status == "ok"
        assert e.comparable
        assert e.split_id == config.split.split_id
        assert e.mean == pytest.approx(math.fsum(e.fold_scores) / len(e.fold_scores), abs=1e-12)
    assert evaluation.selection is not None
    assert evaluation.selection.winner == summary.winner
    # T-LEAK-03 (run level): frozen before the holdout, opened exactly once
    kinds = [(ev.type.value, ev.data) for ev in events(out)]
    frozen = next(i for i, (k, _) in enumerate(kinds) if k == "selection_frozen")
    evaluated = next(i for i, (k, _) in enumerate(kinds) if k == "holdout_evaluated")
    assert frozen < evaluated
    assert kinds[evaluated][1]["accesses"] == (1 if holdout else 0)
    hold = json.loads((out / "holdout.json").read_text())
    assert hold["present"] is holdout
    if holdout:
        assert set(hold["final"]) >= {config.primary_metric.name.value}
        assert hold["baseline"] is not None
    else:
        assert "no holdout" in hold["reason"]
        assert config.split.n_repeats == 3
        assert len(board[0].fold_scores) == 15
    # T-EVL-06: CV results carry no holdout numbers
    assert "holdout" not in (out / "leaderboard.json").read_text()
    trials = (out / "records" / "trials.jsonl").read_text().splitlines()
    assert len(trials) == 4  # 2 trials × 2 tuned models
    check_report_and_bundle(summary, name, task)


@pytest.mark.e2e
def test_cli_is_reproducible_in_trials_mode(tmp_path: Path) -> None:
    first = run(tmp_path / "a", "binary_small.csv", "target")
    second = run(tmp_path / "b", "binary_small.csv", "target")
    assert [e.fold_scores for e in first.entries] == [e.fold_scores for e in second.entries]
    assert first.winner == second.winner


@pytest.mark.e2e
def test_main_prints_summary_and_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    code = main(["run", str(FIX / "binary_small.csv"), "--target", "target", "--out", str(out),
                 "--tuning-mode", "trials", "--trials", "1", "--models", "linear"])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0
    assert "selected: 0-linear-" in text
    assert "holdout: no holdout" in text
    assert (
        main(["run", str(FIX / "binary_small.csv"), "--target", "target", "--out", str(out)]) == 2
    )
    assert "not empty" in capsys.readouterr().err
    bad = main(
        ["run", str(FIX / "binary_small.csv"), "--target", "nope", "--out", str(tmp_path / "x")]
    )
    assert bad == 2
    assert "nope" in capsys.readouterr().err
    over = main(["run", str(FIX / "binary_small.csv"), "--target", "target", "--out",
                 str(tmp_path / "y"), "--profile", "hosted", "--budget-s", "5000"])  # fmt: skip
    assert over == 2
    assert "maximum" in capsys.readouterr().err
    unavailable = main(["run", str(FIX / "binary_small.csv"), "--target", "target", "--out",
                        str(tmp_path / "z"), "--profile", "hosted", "--models", "catboost"])  # fmt: skip
    assert unavailable == 2
    assert "not available" in capsys.readouterr().err


@pytest.mark.unit
def test_parse_options() -> None:
    args = parse(["run", "f.csv", "--target", "y", "--out", "o", "--task", "regression",
                  "--budget-s", "300", "--seed", "7", "--profile", "hosted", "--tuning-mode",
                  "trials", "--trials", "3", "--models", "linear,lightgbm"])  # fmt: skip
    assert args == RunArgs(file=Path("f.csv"), target="y", out=Path("o"), task=Task.REGRESSION,
                           budget_s=300, seed=7, profile=ExecutionProfileName.HOSTED,
                           tuning_mode="trials", trials=3,
                           models=(ModelName.LINEAR, ModelName.LIGHTGBM))  # fmt: skip


@pytest.mark.e2e
def test_guard_exists_from_split_and_refuses_before_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I-1 / L-2 at run level: the guard is built at split time; an access attempt during
    selection is refused; the finalizer opens it exactly once after the freeze (T-LEAK-03).
    """
    guards: list[Any] = []
    real_make_guard, real_select = cli.make_guard, cli.select

    def spy_make_guard(*a: Any, **kw: Any) -> Any:
        guards.append(real_make_guard(*a, **kw))
        return guards[-1]

    def probing_select(entries: Any, freeze: Any) -> Any:
        assert len(guards) == 1  # built before training, at the split stage
        with pytest.raises(HoldoutSealed):
            guards[0].open("probe.before_freeze")
        return real_select(entries, freeze)

    monkeypatch.setattr(cli, "make_guard", spy_make_guard)
    monkeypatch.setattr(cli, "select", probing_select)
    summary = run(tmp_path, "binary_basic.csv", "churn")
    guard = guards[0]
    assert [e.caller for e in guard.access_log] == ["evaluation.holdout"]
    assert [e.caller for e in guard.refused_log] == ["probe.before_freeze"]
    evaluated = next(ev for ev in events(summary.run_dir) if ev.type.value == "holdout_evaluated")
    assert (evaluated.data["accesses"], evaluated.data["refused"]) == (1, 1)


@pytest.mark.e2e
def test_infeasible_job_is_rejected_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FR-ING-05: a rejected feasibility estimate stops the run before any training."""

    class TinyRam:
        total = 64 * 2**20  # 64 MB machine → local soft limit ≈ 45 MB

    monkeypatch.setattr(cli.psutil, "virtual_memory", TinyRam)
    out = tmp_path / "run"
    code = main(["run", str(FIX / "binary_small.csv"), "--target", "target", "--out", str(out)])
    assert code == 2
    err = capsys.readouterr().err
    assert "exceeds the" in err
    assert "even after simplifications" in err
    kinds = [ev.type.value for ev in events(out)]
    assert "candidate_started" not in kinds
    assert "trial_completed" not in kinds
    assert not (out / "records" / "trials.jsonl").read_text()


@pytest.mark.e2e
def test_soft_memory_limit_stops_extra_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§29.6: above the soft RSS limit, no further tuning trials run; the run still finishes."""
    real = cli.ResourceMonitor

    def over_soft(soft: float, hard: float, **kw: Any) -> Any:
        return real(soft, hard, rss_fn=lambda: (soft + hard) / 2, **kw)

    monkeypatch.setattr(cli, "ResourceMonitor", over_soft)
    args = RunArgs(file=FIX / "binary_small.csv", target="target", out=tmp_path / "run",
                   tuning_mode="trials", trials=4, models=(ModelName.LINEAR, ModelName.LIGHTGBM))  # fmt: skip
    summary = execute(args)
    trials = (summary.run_dir / "records" / "trials.jsonl").read_text().splitlines()
    assert len(trials) == 2  # trial 0 (defaults) per model, no extra trials
    assert "resource_warning" in [ev.type.value for ev in events(summary.run_dir)]


@pytest.mark.e2e
def test_winner_oof_saved_and_binary_thresholds_recorded(tmp_path: Path) -> None:
    """FR-TRN-01: the selected candidate's OOF is on disk; §10.2: binary accuracy/F1 at 0.5 and
    at the F1-optimal threshold chosen on OOF.
    """
    summary = run(tmp_path, "binary_basic.csv", "churn")
    experiment = json.loads((summary.run_dir / "records" / "experiment.json").read_text())
    oof = np.load(summary.run_dir / experiment["lineage"]["winner_oof"])
    assert oof.dtype == np.float32
    assert oof.shape == (2000, 2)
    thresholds = experiment["thresholds"]
    assert set(thresholds) == {e.entry_id for e in summary.entries}
    report = thresholds[summary.winner]
    assert report["threshold_0_5"]["threshold"] == 0.5
    assert 0.0 < report["f1_optimal"]["threshold"] < 1.0
    assert report["f1_optimal"]["f1"] >= report["threshold_0_5"]["f1"]
    assert experiment["feasibility"]["status"] in {"ok", "simplified"}
    regression = run(tmp_path / "r", "regression_basic.csv", "price")
    reg = json.loads((regression.run_dir / "records" / "experiment.json").read_text())
    assert reg["thresholds"] == {}


@pytest.mark.e2e
def test_duplicate_grouping_ignores_identifier_columns(tmp_path: Path) -> None:
    """§11.6 (KI-013): duplicated feature rows are grouped even when an ID column differs."""
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"x1": rng.normal(size=270).round(3), "x2": rng.normal(size=270).round(3),
                         "label": rng.choice(["a", "b"], 270)})  # fmt: skip
    frame = pd.concat([base, base.iloc[:30]], ignore_index=True)  # 10% duplicate rows
    frame.insert(0, "customer_id", np.arange(100_000, 100_000 + len(frame)))
    path = tmp_path / "dups.csv"
    frame.to_csv(path, index=False)
    summary = execute(RunArgs(file=path, target="label", out=tmp_path / "run", tuning_mode="trials",
                              trials=1, models=(ModelName.LINEAR,)))  # fmt: skip
    config = json.loads((summary.run_dir / "records" / "experiment.json").read_text())["config"]
    assert config["split"]["duplicate_grouping"] is True


@pytest.mark.e2e
def test_memory_always_over_hard_limit_two_strikes_then_e_no_valid_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§29.6 (KI-014): two hard-limit aborts stop training; §8.9 (KI-015): no valid model →
    E-NO-VALID-MODEL message, not a traceback.
    """
    real = cli.ResourceMonitor
    monitors: list[Any] = []

    def over_hard(soft: float, hard: float, **kw: Any) -> Any:
        monitors.append(real(soft, hard, rss_fn=lambda: hard + 1, **kw))
        return monitors[-1]

    monkeypatch.setattr(cli, "ResourceMonitor", over_hard)
    code = main(["run", str(FIX / "binary_small.csv"), "--target", "target", "--out",
                 str(tmp_path / "run"), "--tuning-mode", "trials", "--trials", "4",
                 "--models", "linear,lightgbm"])  # fmt: skip
    assert code == 2
    err = capsys.readouterr().err
    assert "No model trained successfully" in err
    assert "memory: RSS above" in err  # per-model reasons, not just statuses (KI-023)
    assert monitors[0].hard_strikes == 2


@pytest.mark.e2e
def test_run_trial_cap_is_never_exceeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§17.3 total trials per run (KI-016): later models are skipped once the cap is used."""
    real = cli.TimeBudget
    monkeypatch.setattr(cli, "TimeBudget", lambda total, _cap: real(total, 3))
    args = RunArgs(file=FIX / "binary_small.csv", target="target", out=tmp_path / "run",
                   tuning_mode="trials", trials=3,
                   models=(ModelName.LINEAR, ModelName.LIGHTGBM, ModelName.RANDOM_FOREST))  # fmt: skip
    summary = execute(args)
    trials = [
        json.loads(t)
        for t in (summary.run_dir / "records" / "trials.jsonl").read_text().splitlines()
    ]
    assert len(trials) == 3  # never above the run cap
    # trial 0 is reserved for each planned model, so every model trains (§12.8, KI-022)
    assert {t["model"] for t in trials} == {"lightgbm", "random_forest", "linear"}
    assert all(e.status == "ok" for e in summary.entries)


@pytest.mark.e2e
def test_feasibility_simplifications_are_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§29.5 (KI-017): each simplification in a `simplified` report changes the run."""
    steps = [Simplification.REDUCE_TUNING_TRIALS, Simplification.DISABLE_RANDOM_FOREST,
             Simplification.REDUCE_FOLDS, Simplification.CAP_ONEHOT]  # fmt: skip
    report = FeasibilityReport(status="simplified", estimated_peak_mb=100.0, limit_mb=200.0,
                               simplifications=steps)  # fmt: skip
    monkeypatch.setattr(cli, "estimate_feasibility", lambda *a, **k: report)
    schemas: list[Any] = []
    real_schema = cli._simplified_schema
    monkeypatch.setattr(cli, "_simplified_schema",
                        lambda run, s: schemas.append(real_schema(run, s)) or schemas[-1])  # fmt: skip
    args = RunArgs(file=FIX / "binary_basic.csv", target="churn", out=tmp_path / "run",
                   tuning_mode="trials", trials=4,
                   models=(ModelName.LINEAR, ModelName.LIGHTGBM, ModelName.RANDOM_FOREST))  # fmt: skip
    summary = execute(args)
    experiment = json.loads((summary.run_dir / "records" / "experiment.json").read_text())
    assert experiment["config"]["split"]["k"] == 3  # REDUCE_FOLDS
    assert ModelName.RANDOM_FOREST not in {
        e.model for e in summary.entries
    }  # DISABLE_RANDOM_FOREST
    assert schemas[0].onehot_max_categories == 8  # CAP_ONEHOT
    trials = [
        json.loads(t)
        for t in (summary.run_dir / "records" / "trials.jsonl").read_text().splitlines()
    ]
    per_model = {m: sum(t["model"] == m for t in trials) for m in ("linear", "lightgbm")}
    assert per_model == {"linear": 2, "lightgbm": 2}  # REDUCE_TUNING_TRIALS: 4 → 2
