"""All data contracts (PRD §26).

Pydantic v2 models; depends only on pydantic.

Must not: import any other tabpilot module.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, TypeAdapter

from tabpilot.schemas._base import (
    METRIC_DIRECTION,
    PRIMARY_METRICS,
    ClassWeight,
    ColumnRole,
    Contract,
    Direction,
    ExecutionProfileName,
    MetricName,
    ModelName,
    ProviderName,
    SemanticType,
    Simplification,
    Stage,
    TargetTransform,
    Task,
)
from tabpilot.schemas.dataset import (
    ColumnInfo,
    DatasetMetadata,
    FeasibilityReport,
    ParseReport,
    TargetReport,
    TaskDetection,
)
from tabpilot.schemas.errors import ErrorCode, TabPilotError, WarningCode
from tabpilot.schemas.experiment import (
    CandidateSpec,
    EvaluationResult,
    ExperimentConfig,
    HoldoutResult,
    LeaderboardEntry,
    MetricSpec,
    ModelConfig,
    Seeds,
    SelectionRecord,
    TrainingResult,
    fold_statistics,
)
from tabpilot.schemas.features import (
    FEATURE_SPEC_ADAPTER,
    FeatureProvenance,
    FeatureSpec,
)
from tabpilot.schemas.llm import LLMBudget, LLMRequest, LLMResponse
from tabpilot.schemas.plan import PlannerInput, PlannerOutput, PlanValidation
from tabpilot.schemas.profile import CompactProfile, ProfileSummary
from tabpilot.schemas.reflection import (
    AcceptanceDecision,
    EnsembleResult,
    ReflectionChange,
    ReflectionInput,
    ReflectionProposal,
    ReflectionRecord,
)
from tabpilot.schemas.report import ArtifactManifest, FinalReport
from tabpilot.schemas.run import (
    BudgetLedger,
    ColumnOverride,
    EventType,
    FinalResult,
    RunConfig,
    RunEvent,
    RunResult,
    RunState,
    RunWarning,
    StageError,
)
from tabpilot.schemas.split import SplitSpec

# Every top-level contract, keyed by name. Used by the JSON Schema export and contract tests.
CONTRACTS: dict[str, type[BaseModel]] = {
    model.__name__: model
    for model in (
        DatasetMetadata, ColumnInfo, ParseReport, TargetReport, TaskDetection,
        FeasibilityReport, SplitSpec, ProfileSummary, CompactProfile, PlannerInput,
        PlannerOutput, PlanValidation, FeatureProvenance, ModelConfig, CandidateSpec,
        ExperimentConfig, TrainingResult, LeaderboardEntry, EvaluationResult, SelectionRecord,
        ReflectionInput, ReflectionProposal, ReflectionRecord, AcceptanceDecision,
        EnsembleResult, FinalReport, ArtifactManifest, RunEvent, RunConfig, RunResult,
        RunState, BudgetLedger, LLMBudget, LLMRequest, LLMResponse, RunWarning, StageError,
        MetricSpec, FinalResult,
    )
}  # fmt: skip

# Discriminated unions are exported through type adapters.
UNION_CONTRACTS: dict[str, TypeAdapter[Any]] = {
    "FeatureSpec": FEATURE_SPEC_ADAPTER,
    "ReflectionChange": TypeAdapter(ReflectionChange),
}

__all__ = [
    "CONTRACTS", "METRIC_DIRECTION", "PRIMARY_METRICS", "UNION_CONTRACTS",
    "AcceptanceDecision", "ArtifactManifest", "BudgetLedger", "CandidateSpec", "ClassWeight",
    "ColumnInfo", "ColumnOverride", "ColumnRole", "CompactProfile", "Contract",
    "DatasetMetadata", "Direction", "EnsembleResult", "ErrorCode", "EvaluationResult",
    "EventType", "ExecutionProfileName", "ExperimentConfig", "FeasibilityReport",
    "FeatureProvenance", "FeatureSpec", "FinalReport", "FinalResult", "HoldoutResult",
    "LLMBudget", "LLMRequest", "LLMResponse", "LeaderboardEntry", "MetricName", "MetricSpec",
    "ModelConfig", "ModelName", "ParseReport", "PlanValidation", "PlannerInput",
    "PlannerOutput", "ProfileSummary", "ProviderName", "ReflectionChange", "ReflectionInput",
    "ReflectionProposal", "ReflectionRecord", "RunConfig", "RunEvent", "RunResult", "RunState",
    "RunWarning", "Seeds", "SelectionRecord", "SemanticType", "Simplification", "SplitSpec",
    "Stage", "StageError", "TabPilotError", "TargetReport", "TargetTransform", "Task",
    "TaskDetection", "TrainingResult", "WarningCode", "fold_statistics",
]  # fmt: skip
