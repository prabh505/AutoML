"""Run contracts: config, events, ledger, warnings, errors, result, and graph state.

PRD §13.3, §26.13, §27.4, §28.2–§28.3; A-004; PI-31.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from tabpilot.schemas._base import (
    PRIMARY_METRICS,
    ColumnName,
    ColumnRole,
    Component,
    Contract,
    ExecutionProfileName,
    MetricName,
    MutableContract,
    NonNegFloat,
    NonNegInt,
    ProviderName,
    RunId,
    SemanticType,
    Sha256,
    Stage,
    Task,
)
from tabpilot.schemas.dataset import DatasetMetadata
from tabpilot.schemas.errors import ErrorCode, WarningCode
from tabpilot.schemas.experiment import CandidateSpec, HoldoutResult, LeaderboardEntry
from tabpilot.schemas.plan import PlannerOutput, PlanValidation
from tabpilot.schemas.profile import CompactProfile
from tabpilot.schemas.reflection import EnsembleResult, ReflectionRecord

MIN_TIME_BUDGET_S = 240  # A-004 (slider minimum)
MAX_TIME_BUDGET_S = 7200  # A-004 (local maximum; the active profile maximum is checked on start)


class EventType(StrEnum):
    """Closed catalog of run event types (PI-31). Prompts that add events extend this enum."""

    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_SKIPPED = "stage_skipped"
    WARNING_RAISED = "warning_raised"
    PLAN_RECEIVED = "plan_received"
    PLAN_VALIDATED = "plan_validated"
    PLANNER_FALLBACK = "planner_fallback"
    FEATURE_FAILED = "feature_failed"
    CANDIDATE_STARTED = "candidate_started"
    CANDIDATE_COMPLETED = "candidate_completed"
    CANDIDATE_FAILED = "candidate_failed"
    TRIAL_COMPLETED = "trial_completed"
    LEADERBOARD_UPDATED = "leaderboard_updated"
    REFLECTION_PROPOSED = "reflection_proposed"
    REFLECTION_DECIDED = "reflection_decided"
    REFLECTION_SKIPPED = "reflection_skipped"
    ENSEMBLE_DECIDED = "ensemble_decided"
    SELECTION_FROZEN = "selection_frozen"
    HOLDOUT_EVALUATED = "holdout_evaluated"
    EXPLANATION_FALLBACK = "explanation_fallback"
    REPORT_RENDERED = "report_rendered"
    BUNDLE_EXPORTED = "bundle_exported"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_SUCCEEDED = "llm_call_succeeded"
    LLM_CALL_FAILED = "llm_call_failed"
    LLM_FALLBACK = "llm_fallback"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_RATIONALE = "llm_rationale"
    RESOURCE_WARNING = "resource_warning"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


class RunEvent(Contract):
    """One progress event (`event/1.0`, PRD §26.13).

    Emitted only by the code that performed the action (§27.4). LLM text is carried only with
    `origin = "llm_rationale"`.
    """

    schema_version: Literal["event/1.0"] = "event/1.0"
    run_id: RunId
    seq: NonNegInt
    ts: datetime
    stage: Stage
    type: EventType
    data: dict[str, JsonValue] = Field(default_factory=dict)
    origin: Literal["system", "llm_rationale"] = "system"

    @model_validator(mode="after")
    def _rationale_origin(self) -> Self:
        """LLM rationale events, and only those, are labeled `llm_rationale`."""
        if (self.type == EventType.LLM_RATIONALE) != (self.origin == "llm_rationale"):
            raise ValueError("llm_rationale events must have origin 'llm_rationale' and v.v.")
        return self


class ColumnOverride(Component):
    """A user override from the configure page (PRD §3.2 stage 2, FR-ING-04)."""

    column: ColumnName
    semantic_type: SemanticType | None = None
    role: Literal[ColumnRole.FEATURE, ColumnRole.EXCLUDED] | None = None

    @model_validator(mode="after")
    def _changes_something(self) -> Self:
        if self.semantic_type is None and self.role is None:
            raise ValueError("an override must change the type or the role")
        return self


class RunConfig(Contract):
    """User-confirmed run configuration (`run_config/1.0`, PRD §28.2)."""

    schema_version: Literal["run_config/1.0"] = "run_config/1.0"
    run_id: RunId
    target: ColumnName
    task: Task
    positive_label: str | None = None
    primary_metric: MetricName
    group_column: ColumnName | None = None
    column_overrides: list[ColumnOverride] = Field(default_factory=list)
    time_budget_s: int = Field(ge=MIN_TIME_BUDGET_S, le=MAX_TIME_BUDGET_S)
    seed_root: NonNegInt = 42
    llm_mode: Literal["providers", "off", "stub"] = "providers"
    provider_order: list[ProviderName] = Field(default_factory=list)
    consented_providers: list[ProviderName] = Field(default_factory=list)
    profile_mode: Literal["standard", "minimal"] = "standard"
    reflection_max_rounds: int = Field(default=1, ge=0, le=2)
    enable_ensemble: bool = True
    execution_profile: ExecutionProfileName

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.primary_metric not in PRIMARY_METRICS[self.task]:
            raise ValueError(f"{self.primary_metric} is not a primary metric for {self.task}")
        if self.task != Task.BINARY and self.positive_label is not None:
            raise ValueError("positive_label applies to binary tasks only")
        if self.group_column is not None and self.group_column == self.target:
            raise ValueError("the group column cannot be the target")
        if not set(self.consented_providers) <= set(self.provider_order):
            raise ValueError("consented providers must be in provider_order")
        if len(set(self.provider_order)) != len(self.provider_order):
            raise ValueError("provider_order must not repeat providers")
        return self


class RunWarning(Contract):
    """A structured warning (`warning/1.0`; named `RunWarning` to avoid the builtin, PI-25)."""

    schema_version: Literal["warning/1.0"] = "warning/1.0"
    code: WarningCode
    message: str = Field(max_length=500)
    stage: Stage | None = None
    column: ColumnName | None = None
    count: NonNegInt | None = None


class StageError(Contract):
    """A sanitized stage error (`stage_error/1.0`, PRD §24.10)."""

    schema_version: Literal["stage_error/1.0"] = "stage_error/1.0"
    stage: Stage
    error_code: ErrorCode | None = None
    error_class: str = Field(max_length=100)
    message: str = Field(max_length=500)
    recoverable: bool


class BudgetLedger(Contract):
    """Budget use so far (`budget_ledger/1.0`, PRD §29.3; A-004). Replaced, not mutated."""

    schema_version: Literal["budget_ledger/1.0"] = "budget_ledger/1.0"
    time_budget_s: int = Field(gt=0)
    elapsed_s: NonNegFloat = 0.0
    stage_seconds: dict[Stage, NonNegFloat] = Field(default_factory=dict)
    trials_used: NonNegInt = 0
    trials_max: NonNegInt
    llm_calls_used: NonNegInt = 0
    llm_input_tokens_used: NonNegInt = 0
    llm_output_tokens_used: NonNegInt = 0
    llm_seconds_used: NonNegFloat = 0.0


class FinalResult(Contract):
    """Frozen selection and its outputs (`final_result/1.0`, PRD §13.3, L-8)."""

    schema_version: Literal["final_result/1.0"] = "final_result/1.0"
    selected_entry_id: str
    is_ensemble: bool = False
    holdout: HoldoutResult
    explanation_method: str | None = None
    report_paths: list[str] = Field(default_factory=list)
    bundle_path: str | None = None
    bundle_sha256: Sha256 | None = None


RunStatus = Literal["running", "completed", "completed_partial", "failed", "cancelled"]


class RunResult(Contract):
    """Final run outcome returned to the Run Manager (`run_result/1.0`, PRD §28.2–§28.3)."""

    schema_version: Literal["run_result/1.0"] = "run_result/1.0"
    run_id: RunId
    status: Literal["completed", "completed_partial", "failed", "cancelled"]
    duration_s: NonNegFloat
    error_code: ErrorCode | None = None
    error_message: str | None = Field(default=None, max_length=1000)
    stopped_after_stage: Stage | None = None
    final: FinalResult | None = None
    warnings: list[RunWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _failures_carry_codes(self) -> Self:
        """Every failure has an E-code (T-E2E-05); a partial result names the last stage."""
        if self.status == "failed" and self.error_code is None:
            raise ValueError("a failed run must carry an error code")
        if self.status in ("completed", "completed_partial") and self.final is None:
            raise ValueError("a completed run must carry its final result")
        if self.status == "completed_partial" and self.stopped_after_stage is None:
            raise ValueError("a partial run must name the stage it stopped after")
        return self


class RunState(MutableContract):
    """LangGraph run state (`state/1.0`, PRD §13.3). Small and serializable.

    Large objects (DataFrames, OOF arrays, fitted pipelines) live in the worker's `RunContext`.
    """

    schema_version: Literal["state/1.0"] = "state/1.0"
    run_id: RunId
    config: RunConfig
    dataset_meta: DatasetMetadata
    split_spec_ref: str
    profile_compact: CompactProfile
    plan: PlannerOutput | None = None
    plan_source: Literal["llm", "rule_based"] | None = None
    plan_validation: PlanValidation | None = None
    iteration: int = Field(default=0, ge=0, le=2)
    rounds_used: int = Field(default=0, ge=0, le=2)
    candidates: list[CandidateSpec] = Field(default_factory=list)
    leaderboard: list[LeaderboardEntry] = Field(default_factory=list)
    best_candidate_id: str | None = None
    reflection_history: list[ReflectionRecord] = Field(default_factory=list)
    ensemble: EnsembleResult | None = None
    selection_frozen: bool = False
    final: FinalResult | None = None
    budget: BudgetLedger
    warnings: list[RunWarning] = Field(default_factory=list)
    errors: list[StageError] = Field(default_factory=list)
    status: RunStatus = "running"

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.config.run_id != self.run_id or self.dataset_meta.run_id != self.run_id:
            raise ValueError("config and dataset metadata must belong to this run")
        if self.rounds_used > self.config.reflection_max_rounds:
            raise ValueError("rounds_used exceeds reflection_max_rounds")
        if self.final is not None and not self.selection_frozen:
            raise ValueError("a final result requires a frozen selection")
        return self
