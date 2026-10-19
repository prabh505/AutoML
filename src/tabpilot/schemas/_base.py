"""Base classes and shared enumerations for every TabPilot data contract (PRD §26)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class Contract(BaseModel):
    """Immutable, versioned contract base (PRD §26).

    `extra="forbid"` makes unknown keys fail validation, so LLM-output models surface them to
    the repair loop (PRD §12.5). `protected_namespaces=()` allows the PRD field names that start
    with `model_` (PI-25).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        validate_default=True,
        ser_json_inf_nan="strings",
    )


class Component(Contract):
    """A nested part of a contract. It is versioned through the contract that contains it."""


class MutableContract(BaseModel):
    """Mutable contract base, used only by `RunState` (PRD §26, §13.3)."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        protected_namespaces=(),
        validate_default=True,
    )


ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

RunId = Annotated[str, StringConstraints(pattern=ULID_PATTERN)]
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_PATTERN)]
ColumnName = Annotated[str, StringConstraints(min_length=1, max_length=80)]
ShortText = Annotated[str, StringConstraints(max_length=300)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegInt = Annotated[int, Field(ge=0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]


class Task(StrEnum):
    """Supported prediction tasks (PRD §2.2, §10.1)."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class SemanticType(StrEnum):
    """Column semantic types (FR-ING-04, §8.5)."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"


class ColumnRole(StrEnum):
    """Column roles (FR-ING-04)."""

    FEATURE = "feature"
    TARGET = "target"
    EXCLUDED = "excluded"
    GROUP = "group"


class ExecutionProfileName(StrEnum):
    """Execution profiles (PRD §29.1)."""

    HOSTED = "hosted"
    LOCAL = "local"


class ModelName(StrEnum):
    """Model zoo entries (PRD §16.1)."""

    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    HIST_GB = "hist_gb"


class ProviderName(StrEnum):
    """LLM providers (PRD §23.2)."""

    GROQ = "groq"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class ClassWeight(StrEnum):
    """Class-weight setting for classification models (PRD §12.3, §16.2)."""

    NONE = "none"
    BALANCED = "balanced"


class TargetTransform(StrEnum):
    """Regression target transform (PRD §12.3, §10.4)."""

    NONE = "none"
    LOG1P = "log1p"


class Direction(StrEnum):
    """Metric optimization direction (PRD §10.4)."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MetricName(StrEnum):
    """Metric names (PRD §10.2)."""

    ROC_AUC = "roc_auc"
    AVERAGE_PRECISION = "average_precision"
    LOG_LOSS = "log_loss"
    BRIER = "brier"
    ACCURACY = "accuracy"
    F1 = "f1"
    MACRO_F1 = "macro_f1"
    BALANCED_ACCURACY = "balanced_accuracy"
    OVR_MACRO_ROC_AUC = "ovr_macro_roc_auc"
    RMSE = "rmse"
    MAE = "mae"
    RMSLE = "rmsle"
    R2 = "r2"


METRIC_DIRECTION: dict[MetricName, Direction] = {
    MetricName.ROC_AUC: Direction.MAXIMIZE,
    MetricName.AVERAGE_PRECISION: Direction.MAXIMIZE,
    MetricName.LOG_LOSS: Direction.MINIMIZE,
    MetricName.BRIER: Direction.MINIMIZE,
    MetricName.ACCURACY: Direction.MAXIMIZE,
    MetricName.F1: Direction.MAXIMIZE,
    MetricName.MACRO_F1: Direction.MAXIMIZE,
    MetricName.BALANCED_ACCURACY: Direction.MAXIMIZE,
    MetricName.OVR_MACRO_ROC_AUC: Direction.MAXIMIZE,
    MetricName.RMSE: Direction.MINIMIZE,
    MetricName.MAE: Direction.MINIMIZE,
    MetricName.RMSLE: Direction.MINIMIZE,
    MetricName.R2: Direction.MAXIMIZE,
}

# User-selectable primary metrics per task (PRD §10.2).
PRIMARY_METRICS: dict[Task, frozenset[MetricName]] = {
    Task.BINARY: frozenset({MetricName.ROC_AUC, MetricName.AVERAGE_PRECISION, MetricName.LOG_LOSS}),
    Task.MULTICLASS: frozenset(
        {MetricName.LOG_LOSS, MetricName.MACRO_F1, MetricName.BALANCED_ACCURACY}
    ),
    Task.REGRESSION: frozenset({MetricName.RMSE, MetricName.MAE, MetricName.RMSLE}),
}


class Stage(StrEnum):
    """Run stages and graph nodes used for events, timings, and errors (PRD §13.2, §13.4)."""

    RUN = "run"
    INGEST = "ingest"
    SPLIT = "split"
    PROFILE = "profile"
    PLAN = "plan"
    VALIDATE_PLAN = "validate_plan"
    BUILD_FEATURES = "build_features"
    TRAIN_CANDIDATES = "train_candidates"
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    APPLY_CHANGES = "apply_changes"
    ENSEMBLE = "ensemble"
    FINALIZE = "finalize"
    EXPLAIN = "explain"
    REPORT = "report"
    EXPORT = "export"
    RECOVER = "recover"


class Simplification(StrEnum):
    """Resource simplifications, applied in this order (PRD §29.5)."""

    REDUCE_TUNING_TRIALS = "reduce_tuning_trials"
    WITHIN_FOLD_SUBSAMPLE = "within_fold_subsample"
    DISABLE_RANDOM_FOREST = "disable_random_forest"
    REDUCE_FOLDS = "reduce_folds"
    CAP_ONEHOT = "cap_onehot"
    DISABLE_ENSEMBLE = "disable_ensemble"
