"""Reflection and ensembling contracts (PRD §19, §20, §26.10)."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from tabpilot.schemas._base import (
    ClassWeight,
    ColumnName,
    Component,
    Contract,
    Fraction,
    ModelName,
    NonNegFloat,
    NonNegInt,
    TargetTransform,
    Task,
)
from tabpilot.schemas.experiment import MetricSpec
from tabpilot.schemas.features import MAX_FEATURES_PER_REFLECTION, FeatureProvenance, FeatureSpec
from tabpilot.schemas.plan import ALLOCATION_TOLERANCE, PlannerOutput


class ReflectionLeaderboardRow(Component):
    """A leaderboard row as shown to the reflector (PRD §19.2)."""

    entry_id: str
    model: ModelName | Literal["ensemble"]
    feature_set: Literal["baseline", "baseline+llm", "ensemble"]
    mean: float
    std: NonNegFloat
    fold_scores: list[float]
    fit_seconds: NonNegFloat


class FeatureImportance(Component):
    """Normalized importance of a **source** column (PRD §19.2; PI-16: no one-hot labels)."""

    name: ColumnName
    importance: Fraction


class RemainingBudget(Component):
    """What the reflector may still spend (PRD §19.2, §19.4)."""

    seconds: NonNegFloat
    trials: NonNegInt
    features_allowed: int = Field(ge=0, le=MAX_FEATURES_PER_REFLECTION)


class ReflectionInput(Contract):
    """Evidence given to the reflector (`reflection_input/1.0`, PRD §19.2).

    Contains no raw rows, no predictions, and no holdout information.
    """

    schema_version: Literal["reflection_input/1.0"] = "reflection_input/1.0"
    task: Task
    primary_metric: MetricSpec
    leaderboard_top: list[ReflectionLeaderboardRow] = Field(max_length=5)
    top_features: list[FeatureImportance] = Field(default_factory=list, max_length=15)
    llm_features: list[FeatureProvenance] = Field(default_factory=list)
    current_plan: PlannerOutput
    remaining_budget: RemainingBudget
    warnings: list[str] = Field(default_factory=list)

    @field_validator("current_plan")
    @classmethod
    def _plan_without_rationale(cls, v: PlannerOutput) -> PlannerOutput:
        """The plan is passed without its rationale (PRD §19.2)."""
        if v.rationale:
            raise ValueError("current_plan must not carry the rationale")
        return v


class AddFeatures(Component):
    """Add ≤ 6 new DSL features (PRD §19.3)."""

    type: Literal["add_features"]
    features: list[FeatureSpec] = Field(min_length=1, max_length=MAX_FEATURES_PER_REFLECTION)


class RemoveFeatures(Component):
    """Remove LLM features by name (PRD §19.3)."""

    type: Literal["remove_features"]
    feature_names: list[str] = Field(min_length=1)


class ChangeEncoding(Component):
    """Change encoding for named columns within the allowed encodings (PRD §19.3)."""

    type: Literal["change_encoding"]
    columns: list[ColumnName] = Field(min_length=1)
    encoding: Literal["onehot", "ordinal", "frequency", "target"]


class ToggleClassWeight(Component):
    """Set class weighting for one model (PRD §19.3)."""

    type: Literal["toggle_class_weight"]
    model: ModelName
    value: ClassWeight


class ChangeTargetTransform(Component):
    """Regression only: none ↔ log1p (PRD §19.3)."""

    type: Literal["change_target_transform"]
    value: TargetTransform


class ReallocateTuning(Component):
    """New tuning fractions over available models (PRD §19.3)."""

    type: Literal["reallocate_tuning"]
    allocation: dict[ModelName, float]

    @field_validator("allocation")
    @classmethod
    def _bounded(cls, v: dict[ModelName, float]) -> dict[ModelName, float]:
        if any(f < 0 for f in v.values()) or sum(v.values()) > 1.0 + ALLOCATION_TOLERANCE:
            raise ValueError("fractions must be non-negative and sum to at most 1.0")
        return v


class DropModel(Component):
    """Drop one model; the validator keeps ≥ 1 model (PRD §19.3)."""

    type: Literal["drop_model"]
    model: ModelName


ReflectionChange = Annotated[
    AddFeatures
    | RemoveFeatures
    | ChangeEncoding
    | ToggleClassWeight
    | ChangeTargetTransform
    | ReallocateTuning
    | DropModel,
    Field(discriminator="type"),
]


class ReflectionProposal(Contract):
    """One bounded set of changes (`reflection/1.0`, PRD §19.3, §26.10).

    Task, target, metric, folds, holdout, seeds, and budgets are not fields, so they cannot
    be changed.
    """

    schema_version: Literal["reflection/1.0"] = "reflection/1.0"
    changes: list[ReflectionChange] = Field(default_factory=list, max_length=10)
    expected_effect: str = Field(default="", max_length=300)
    rationale: str = Field(default="", max_length=800)


class AcceptanceDecision(Contract):
    """Paired-fold acceptance outcome (`acceptance/1.0`, PRD §19.5, §20.4)."""

    schema_version: Literal["acceptance/1.0"] = "acceptance/1.0"
    parent_entry_id: str
    child_entry_id: str
    fold_deltas: list[float] = Field(min_length=1)
    mean_gain: float
    rel_gain: float
    folds_better: NonNegInt
    folds_required: NonNegInt
    accepted: bool
    reason: str = Field(max_length=300)

    @model_validator(mode="after")
    def _counts(self) -> Self:
        if self.folds_better != sum(d > 0 for d in self.fold_deltas):
            raise ValueError("folds_better must count positive fold deltas")
        return self


class ReflectionRecord(Contract):
    """One reflection round (`reflection_record/1.0`, PRD §19.6–§19.7)."""

    schema_version: Literal["reflection_record/1.0"] = "reflection_record/1.0"
    round: int = Field(ge=1, le=2)
    status: Literal["accepted", "rejected", "skipped", "invalid", "failed", "empty"]
    reason: str = Field(max_length=300)
    proposal: ReflectionProposal | None = None
    issues: list[str] = Field(default_factory=list)
    decisions: list[AcceptanceDecision] = Field(default_factory=list)


class EnsembleMember(Component):
    """One ensemble member and its weight (PRD §20.3)."""

    entry_id: str
    weight: Fraction


class EnsembleResult(Contract):
    """Ensemble search outcome (`ensemble/1.0`, PRD §20)."""

    schema_version: Literal["ensemble/1.0"] = "ensemble/1.0"
    kept: bool
    reason: str = Field(max_length=300)
    members: list[EnsembleMember] = Field(default_factory=list, max_length=5)
    fold_scores: list[float] = Field(default_factory=list)
    decision: AcceptanceDecision | None = None
    predicted_inference_ratio: NonNegFloat | None = None

    @model_validator(mode="after")
    def _weights(self) -> Self:
        if self.members and abs(sum(m.weight for m in self.members) - 1.0) > 1e-9:
            raise ValueError("member weights must sum to 1")
        if self.kept and (len(self.members) < 2 or self.decision is None):
            raise ValueError("a kept ensemble has ≥ 2 members and an acceptance decision")
        return self
