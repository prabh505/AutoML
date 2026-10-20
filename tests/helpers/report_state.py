"""A `ReportState` built from the PRD §26 contract examples (P13/P14 tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tabpilot.evaluation.metrics import Problem
from tabpilot.reporting import ReportState
from tabpilot.schemas import DatasetMetadata, ExperimentConfig, PlannerOutput, Task
from tabpilot.schemas.experiment import EvaluationResult

CONTRACTS = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"


def contract(name: str) -> dict[str, Any]:
    data = json.loads((CONTRACTS / f"{name}.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def report_state(problem: Problem | None = None, **overrides: Any) -> ReportState:
    """Contract-example run facts; `problem` sets the task (binary 0/1 by default)."""
    problem = problem or Problem(Task.BINARY, ("0", "1"), 1)
    ev = EvaluationResult.model_validate(contract("prd_26_9_EvaluationResult"))
    config = ExperimentConfig.model_validate(contract("prd_26_7_ExperimentConfig"))
    config = config.model_copy(update={"task": problem.task})
    base: dict[str, Any] = {
        "config": config,
        "metadata": DatasetMetadata.model_validate(contract("prd_26_1_DatasetMetadata")),
        "plan": PlannerOutput.model_validate(contract("prd_26_4_PlannerOutput")),
        "entries": ev.entries, "selection": ev.selection, "holdout": ev.holdout,
        "problem": problem,
    }  # fmt: skip
    base.update(overrides)
    return ReportState(**base)
