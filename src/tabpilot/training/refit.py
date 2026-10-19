"""Final refit on all DEV rows and the reference baseline (PRD §11.10, FR-EVL-02)."""

from __future__ import annotations

import math
from typing import Any

from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import PipelineSchema, build_pipeline
from tabpilot.schemas import CandidateSpec, ExecutionProfileName, ModelName, SplitSpec
from tabpilot.schemas.experiment import ModelConfig
from tabpilot.training.cv import DevData

REFIT_ITERATION_FACTOR = 1.1
ROUNDS_PARAM = {ModelName.LIGHTGBM: "n_estimators", ModelName.XGBOOST: "n_estimators",
                ModelName.CATBOOST: "iterations"}  # fmt: skip
BASELINE_ORDER = (ModelName.LIGHTGBM, ModelName.HIST_GB, ModelName.RANDOM_FOREST,
                  ModelName.LINEAR)  # fmt: skip


def refit_params(candidate: CandidateSpec, best_iterations: list[int]) -> dict[str, Any]:
    """Selected params; boosting rounds = round(mean CV best iteration × 1.1) (§11.10)."""
    params: dict[str, Any] = dict(candidate.model.params)
    key = ROUNDS_PARAM.get(candidate.model.name)
    if key and best_iterations:
        mean = sum(best_iterations) / len(best_iterations)
        params[key] = max(1, math.floor(mean * REFIT_ITERATION_FACTOR + 0.5))
    return params


def refit_final(
    candidate: CandidateSpec,
    best_iterations: list[int],
    dev: DevData,
    split: SplitSpec,
    *,
    schema: PipelineSchema,
    registry: ModelRegistry,
    profile: ExecutionProfileName | None = None,
) -> Any:
    """Fit the selected configuration on every DEV row, without early stopping (§11.10)."""
    model = candidate.model.model_copy(update={"params": refit_params(candidate, best_iterations)})
    final = candidate.model_copy(update={"model": model})
    rows = list(split.dev_indices)
    pipe = build_pipeline(final, schema, registry, early_stopping=False, profile=profile)
    return pipe.fit(dev.frame.iloc[rows], dev.y[rows])


def baseline_candidate(available: list[ModelName], template: ModelConfig) -> CandidateSpec:
    """Reference baseline: default LightGBM, else the best available fallback (FR-EVL-02)."""
    name = next((m for m in BASELINE_ORDER if m in available), ModelName.LINEAR)
    model = template.model_copy(update={"name": name, "params": {}})
    return CandidateSpec(candidate_id="reference_baseline", iteration=0, model=model,
                         feature_set="baseline")  # fmt: skip
