"""P09 CLI end-to-end tests on the task fixtures, `trials` mode (FR-PLN-03, FR-EVL-01/02)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from tabpilot.cli import RunArgs, execute, main, parse
from tabpilot.schemas import ExecutionProfileName, ModelName, RunEvent, Task
from tabpilot.schemas.experiment import EvaluationResult, ExperimentConfig, LeaderboardEntry
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
