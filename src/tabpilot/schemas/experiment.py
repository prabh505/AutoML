"""Experiment contracts: model/candidate configs, metrics, training results, leaderboard,
selection, and evaluation (PRD §10, §18, §26.6–§26.9).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from tabpilot.schemas._base import (
    METRIC_DIRECTION,
    PRIMARY_METRICS,
    ClassWeight,
    ColumnName,
    Component,
    Contract,
    Direction,
    ExecutionProfileName,
    MetricName,
    ModelName,
    NonNegFloat,
    NonNegInt,
    ProviderName,
    RunId,
    TargetTransform,
    Task,
)
from tabpilot.schemas.features import MAX_ACTIVE_LLM_FEATURES, FeatureSpec
from tabpilot.schemas.plan import ColumnDecision, Encoding, Imputation

ParamValue = int | float | str | bool | None
ENTRY_ID_PATTERN = r"^[0-2]-[a-z_]+-[0-9a-f]{8}$"
FOLD_SCORE_TOLERANCE = 1e-9


def fold_statistics(scores: list[float]) -> tuple[float, float, float]:
    """Mean, sample std (ddof=1; 0 for one fold) and SEM of fold scores (PRD §18.1)."""
    n = len(scores)
    mean = math.fsum(scores) / n
    std = math.sqrt(math.fsum((s - mean) ** 2 for s in scores) / (n - 1)) if n > 1 else 0.0
    return mean, std, std / math.sqrt(n)


class MetricSpec(Contract):
    """A metric and its direction (`metric_spec/1.0`, PRD §10.2, §10.4)."""

    schema_version: Literal["metric_spec/1.0"] = "metric_spec/1.0"
    name: MetricName
    direction: Direction

    @model_validator(mode="after")
    def _direction_matches_metric(self) -> Self:
        """Direction is fixed by the metric, never chosen freely."""
        if METRIC_DIRECTION[self.name] != self.direction:
            raise ValueError(f"{self.name} must be {METRIC_DIRECTION[self.name]}")
        return self

    def selection_score(self, value: float) -> float:
        """Transform a metric value so that higher is always better (PRD §10.4)."""
        return value if self.direction == Direction.MAXIMIZE else -value


class ModelConfig(Contract):
    """A concrete model configuration (`model_config/1.0`, PRD §26.6)."""

    schema_version: Literal["model_config/1.0"] = "model_config/1.0"
    name: ModelName
    task: Task
    params: dict[str, ParamValue] = Field(default_factory=dict)
    class_weight: ClassWeight = ClassWeight.NONE
    seed: NonNegInt
    n_jobs: int = Field(ge=1)


class CandidateSpec(Contract):
    """A candidate pipeline: preprocessing + features + model (`candidate/1.0`, PRD §0.4)."""

    schema_version: Literal["candidate/1.0"] = "candidate/1.0"
    candidate_id: str = Field(min_length=1, max_length=60)
    iteration: NonNegInt = Field(le=2)
    model: ModelConfig
    feature_set: Literal["baseline", "baseline+llm"]
    features: list[FeatureSpec] = Field(default_factory=list, max_length=MAX_ACTIVE_LLM_FEATURES)
    column_decisions: list[ColumnDecision] = Field(default_factory=list)
    imputation: Imputation = Field(default_factory=Imputation)
    encoding: Encoding = Field(default_factory=Encoding)
    target_transform: TargetTransform = TargetTransform.NONE
    parent_entry_id: str | None = None
    change_set_id: str | None = None

    @model_validator(mode="after")
    def _feature_set_matches_features(self) -> Self:
        """`baseline` candidates carry no LLM features."""
        if (self.feature_set == "baseline") == bool(self.features):
            raise ValueError("feature_set 'baseline' ⇔ no LLM features")
        return self


class SplitSummary(Component):
    """Split facts recorded in `ExperimentConfig` (PRD §26.7)."""

    split_id: str = Field(min_length=1)
    strategy: str
    k: int = Field(ge=2, le=5)
    n_repeats: Literal[1, 3] = 1
    holdout_fraction: float = Field(ge=0.0, le=0.5)
    holdout_present: bool
    group_column: ColumnName | None = None
    duplicate_grouping: bool = False


class Seeds(Component):
    """Seeds derived from `root` (PRD §11.3; invariant I-13)."""

    root: NonNegInt
    split: NonNegInt
    model: NonNegInt
    optuna: NonNegInt
    profile: NonNegInt
    explain: NonNegInt

    @staticmethod
    def offsets(root: int) -> dict[str, int]:
        """Seed values derived from `root` exactly as §11.3 specifies."""
        return {"root": root, "split": root, "model": root + 1, "optuna": root + 2,
                "profile": root + 3, "explain": root + 4}  # fmt: skip

    @classmethod
    def derive(cls, root: int) -> Seeds:
        """Build the seed set for `root`."""
        return cls(**cls.offsets(root))

    @model_validator(mode="after")
    def _derived_from_root(self) -> Self:
        if self.model_dump() != Seeds.offsets(self.root):
            raise ValueError("seeds must be derived from root per PRD §11.3")
        return self


class Budgets(Component):
    """Run budgets recorded in `ExperimentConfig` (PRD §26.7)."""

    time_s: int = Field(gt=0)
    max_trials: NonNegInt
    llm_calls: int = Field(ge=0, le=25)
    reflection_rounds: int = Field(ge=0, le=2)


class LLMSettingsSummary(Component):
    """LLM settings recorded in `ExperimentConfig` (PRD §26.7)."""

    enabled: bool
    order: list[ProviderName] = Field(default_factory=list)


class ExperimentConfig(Contract):
    """Frozen experiment configuration (`experiment_config/1.0`, PRD §26.7)."""

    schema_version: Literal["experiment_config/1.0"] = "experiment_config/1.0"
    run_id: RunId
    tabpilot_version: str
    execution_profile: ExecutionProfileName
    task: Task
    target: ColumnName
    positive_label: str | None = None
    primary_metric: MetricSpec
    split: SplitSummary
    seeds: Seeds
    budgets: Budgets
    llm: LLMSettingsSummary
    library_versions: dict[str, str] = Field(default_factory=dict)
    prompt_template_versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _metric_allowed_for_task(self) -> Self:
        """The primary metric is one of the task's allowed metrics (FR-TGT-03)."""
        if self.primary_metric.name not in PRIMARY_METRICS[self.task]:
            raise ValueError(f"{self.primary_metric.name} is not a primary metric for {self.task}")
        if self.task != Task.BINARY and self.positive_label is not None:
            raise ValueError("positive_label applies to binary tasks only")
        return self


ResultStatus = Literal["ok", "failed", "skipped_resource", "skipped_budget"]


class TrainingResult(Contract):
    """Outcome of one candidate's CV / tuning (`training_result/1.0`, PRD §26.8)."""

    schema_version: Literal["training_result/1.0"] = "training_result/1.0"
    entry_id: str = Field(pattern=ENTRY_ID_PATTERN)
    status: ResultStatus
    fold_scores: list[float] = Field(default_factory=list)
    fit_seconds_mean: NonNegFloat = 0.0
    predict_seconds_mean: NonNegFloat = 0.0
    oof_ref: str | None = None
    best_params: dict[str, ParamValue] = Field(default_factory=dict)
    trials_completed: NonNegInt = 0
    trials_pruned: NonNegInt = 0
    trials_failed: NonNegInt = 0
    error: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _status_consistent(self) -> Self:
        """ok ⇒ finite fold scores; failed ⇒ a sanitized error (FR-TRN-01)."""
        if self.status == "ok" and (
            not self.fold_scores or not all(math.isfinite(s) for s in self.fold_scores)
        ):
            raise ValueError("an ok result needs finite fold scores")
        if self.status == "failed" and not self.error:
            raise ValueError("a failed result needs an error reason")
        return self


class LeaderboardEntry(Contract):
    """One leaderboard row (`leaderboard_entry/1.0`, PRD §18.1).

    `mean`, `std` (ddof=1) and `sem` must equal the values recomputed from `fold_scores`
    (G-09 / T-EVL-01).
    """

    schema_version: Literal["leaderboard_entry/1.0"] = "leaderboard_entry/1.0"
    entry_id: str = Field(pattern=ENTRY_ID_PATTERN)
    experiment_id: RunId
    iteration: NonNegInt = Field(le=2)
    model: ModelName | Literal["ensemble"]
    feature_set: Literal["baseline", "baseline+llm", "ensemble"]
    params: dict[str, ParamValue] = Field(default_factory=dict)
    metric: MetricSpec
    fold_scores: list[float] = Field(default_factory=list)
    mean: float | None = None
    std: NonNegFloat | None = None
    sem: NonNegFloat | None = None
    secondary: dict[MetricName, float] = Field(default_factory=dict)
    fit_seconds: NonNegFloat = 0.0
    predict_seconds: NonNegFloat = 0.0
    model_size_bytes: NonNegInt = 0
    n_features_out: NonNegInt = 0
    status: ResultStatus
    comparable: bool
    warnings: list[str] = Field(default_factory=list)
    split_id: str = Field(min_length=1)
    parent_entry_id: str | None = None
    change_set_id: str | None = None

    @model_validator(mode="after")
    def _statistics_match_fold_scores(self) -> Self:
        if self.status == "ok" and not self.fold_scores:
            raise ValueError("an ok entry needs fold scores")
        if not self.fold_scores:
            if any(v is not None for v in (self.mean, self.std, self.sem)):
                raise ValueError("mean/std/sem require fold scores")
            return self
        mean, std, sem = fold_statistics(self.fold_scores)
        for name, stored, expected in (("mean", self.mean, mean), ("std", self.std, std),
                                       ("sem", self.sem, sem)):  # fmt: skip
            if stored is None or not math.isclose(
                stored, expected, rel_tol=FOLD_SCORE_TOLERANCE, abs_tol=FOLD_SCORE_TOLERANCE
            ):
                raise ValueError(f"{name} does not match fold_scores (expected {expected!r})")
        return self


class RunnerUpPaired(Component):
    """Paired fold differences winner − runner-up, in selection-score units (PRD §18.3)."""

    runner_up: str | None = None
    mean_diff: float
    min_diff: float | None = None
    max_diff: float | None = None
    folds_won: NonNegInt
    n_folds: NonNegInt | None = None


class SelectionRecord(Contract):
    """Model-selection outcome (`selection/1.0`, PRD §18.2)."""

    schema_version: Literal["selection/1.0"] = "selection/1.0"
    winner: str
    rule: str = Field(min_length=1, max_length=80)
    near_ties: list[str] = Field(default_factory=list)
    tie_break: str | None = Field(default=None, max_length=80)
    runner_up_paired: RunnerUpPaired | None = None


HoldoutScores = dict[str, float | list[float]]


class HoldoutResult(Component):
    """One-time holdout scores (PRD §18.3, §26.9; L-7). `ci95` is [low, high] when present."""

    present: bool
    final: HoldoutScores | None = None
    baseline: HoldoutScores | None = None
    accessed_at: datetime | None = None
    reason: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        for scores in (self.final, self.baseline):
            for key, value in (scores or {}).items():
                if key == "ci95":
                    if not (isinstance(value, list) and len(value) == 2 and value[0] <= value[1]):
                        raise ValueError("ci95 must be [low, high]")
                elif not isinstance(value, float | int):
                    raise ValueError(f"holdout score {key} must be a number")
        if self.present and (self.final is None or self.accessed_at is None):
            raise ValueError("a present holdout records final scores and access time")
        if not self.present and (self.final is not None or self.reason is None):
            raise ValueError("an absent holdout has no scores and states a reason")
        return self


class EvaluationResult(Contract):
    """Leaderboard, selection, and holdout (`evaluation/1.0`, PRD §26.9)."""

    schema_version: Literal["evaluation/1.0"] = "evaluation/1.0"
    entries: list[LeaderboardEntry]
    selection: SelectionRecord | None = None
    holdout: HoldoutResult | None = None

    @model_validator(mode="after")
    def _winner_is_listed(self) -> Self:
        """A winner must be an eligible entry (PRD §18.2)."""
        if self.selection is not None:
            eligible = {e.entry_id for e in self.entries if e.status == "ok" and e.comparable}
            if self.selection.winner not in eligible:
                raise ValueError("selection winner must be an ok, comparable entry")
        return self
