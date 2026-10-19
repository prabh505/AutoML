"""Planner contracts: input, output (`plan/1.0`), and validation result (PRD §12, §26.3–§26.4)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from tabpilot.schemas._base import (
    ClassWeight,
    ColumnName,
    Component,
    Contract,
    MetricName,
    ModelName,
    Simplification,
    TargetTransform,
    Task,
)
from tabpilot.schemas.features import MAX_FEATURES_PER_PLAN, FeatureSpec
from tabpilot.schemas.profile import CompactProfile

ALLOCATION_TOLERANCE = 1e-6


class DslCatalogEntry(Component):
    """One DSL op as described to the planner (PRD §26.3)."""

    op: str = Field(min_length=1, max_length=40)
    inputs: list[str]
    params: dict[str, list[float]] = Field(default_factory=dict)


class PlannerBudget(Component):
    """Budget facts given to the planner (PRD §12.2)."""

    train_seconds: float = Field(gt=0)
    max_features: int = Field(default=MAX_FEATURES_PER_PLAN, ge=0, le=MAX_FEATURES_PER_PLAN)
    max_models: int = Field(default=5, ge=1, le=5)


class PlannerInput(Contract):
    """Exactly what the planner LLM receives (`planner_input/1.0`, PRD §12.2, §26.3)."""

    schema_version: Literal["planner_input/1.0"] = "planner_input/1.0"
    task: Task
    primary_metric: MetricName
    profile: CompactProfile
    available_models: list[ModelName] = Field(min_length=1)
    dsl_catalog: list[DslCatalogEntry]
    budget: PlannerBudget
    constraints: list[str] = Field(default_factory=list)


class ColumnDecision(Component):
    """Planner decision for one listed column (PRD §12.3)."""

    column: ColumnName
    action: Literal["keep", "drop", "treat_as_categorical", "treat_as_numeric"]
    reason: str = Field(default="", max_length=200)


class Imputation(Component):
    """Imputation choices (PRD §12.3)."""

    numeric: Literal["median", "mean", "constant_zero"] = "median"
    categorical: Literal["most_frequent", "constant_missing"] = "most_frequent"
    add_missing_indicators: bool = True


class Encoding(Component):
    """Encoding choices (PRD §12.3)."""

    low_cardinality: Literal["onehot", "ordinal"] = "onehot"
    high_cardinality: Literal["frequency", "target"] = "frequency"


class PlannedModel(Component):
    """One model in the plan (PRD §12.3)."""

    name: ModelName
    priority: int = Field(ge=1, le=5)
    class_weight: ClassWeight = ClassWeight.NONE


class PlannerOutput(Contract):
    """A modeling plan from the LLM or the rule-based planner (`plan/1.0`, PRD §12.3, §26.4).

    Structural bounds live here; dataset-dependent checks (columns, availability, DSL types,
    cost) are done by the plan validator (P11, §12.6).
    """

    schema_version: Literal["plan/1.0"] = "plan/1.0"
    column_decisions: list[ColumnDecision] = Field(default_factory=list)
    imputation: Imputation = Field(default_factory=Imputation)
    encoding: Encoding = Field(default_factory=Encoding)
    features: list[FeatureSpec] = Field(default_factory=list, max_length=MAX_FEATURES_PER_PLAN)
    models: list[PlannedModel] = Field(min_length=1, max_length=5)
    tuning_allocation: dict[ModelName, float] = Field(default_factory=dict)
    target_transform: TargetTransform = TargetTransform.NONE
    cv_folds: Literal[3, 5] = 5
    metric_recommendation: str | None = Field(default=None, max_length=40)
    rationale: str = Field(default="", max_length=800)

    @field_validator("tuning_allocation")
    @classmethod
    def _allocation_bounded(cls, v: dict[ModelName, float]) -> dict[ModelName, float]:
        """Fractions are non-negative and sum to at most 1.0 (PRD §12.3)."""
        if any(f < 0 for f in v.values()):
            raise ValueError("tuning fractions must be non-negative")
        if sum(v.values()) > 1.0 + ALLOCATION_TOLERANCE:
            raise ValueError("tuning fractions must sum to at most 1.0")
        return v

    @model_validator(mode="after")
    def _unique_models(self) -> Self:
        """Each model appears once."""
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("models must be unique")
        return self


class PlanIssue(Component):
    """One validator finding (PRD §12.6)."""

    code: str = Field(min_length=1, max_length=60)
    severity: Literal["info", "repair", "severe"]
    message: str = Field(max_length=300)
    path: str | None = Field(default=None, max_length=120)


class PlanValidation(Contract):
    """Outcome of plan validation (`plan_validation/1.0`, PRD §12.6–§12.7)."""

    schema_version: Literal["plan_validation/1.0"] = "plan_validation/1.0"
    status: Literal["valid", "repaired", "invalid"]
    issues: list[PlanIssue] = Field(default_factory=list)
    cv_folds_ignored: bool = False
    simplifications: list[Simplification] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_issues(self) -> Self:
        """valid ⇔ no issues; repaired ⇒ no severe issue (PRD §12.6)."""
        severe = any(i.severity == "severe" for i in self.issues)
        if self.status == "valid" and self.issues:
            raise ValueError("a valid plan has no issues")
        if self.status == "repaired" and (severe or not self.issues):
            raise ValueError("a repaired plan has issues, none severe")
        return self
