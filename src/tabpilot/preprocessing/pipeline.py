"""Pipeline factory: one scikit-learn `Pipeline` per candidate (PRD §15, FR-PRE-01).

Layout (§15.1): normalize_columns → type_cast → baseline_features → dsl_features → select →
preprocess (`ColumnTransformer`, §15.2) → model. Regression with `log1p` wraps the whole pipeline in
`TransformedTargetRegressor`. Every learned step is inside the pipeline, so `Pipeline.fit` on a
training fold fits everything on those rows only (L-4).

Branch membership is decided by column name with anchored regex selectors, because datetime part
columns (`hour` only when times exist) and DSL output names (collision suffixes) are resolved at
fit time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from tabpilot.features import runtime_specs
from tabpilot.features.dsl.catalog import CATALOG
from tabpilot.models import ModelRegistry
from tabpilot.schemas import (
    CandidateSpec,
    DatasetMetadata,
    ExecutionProfileName,
    ModelName,
    ProfileSummary,
    SemanticType,
    TargetTransform,
    Task,
)
from tabpilot_runtime import (
    BaselineFeatureTransformer,
    BooleanEncoder,
    ColumnNameMapper,
    ColumnSelector,
    DSLFeatureTransformer,
    FrequencyEncoder,
    TabularModel,
    TypeCaster,
)
from tabpilot_runtime.transformers import LLM_PREFIX

LOW_CARD_MAX = 15
MID_CARD_MAX = 50
ONEHOT_MIN_FREQUENCY = 0.01
MID_CARD_MAX_CATEGORIES = 16
MISSING_INDICATOR_RATE = 0.05  # A-006
TARGET_ENCODER_FOLDS = 5
MISSING = "__missing__"
DATETIME_PARTS = ("year", "month", "day", "dayofweek", "hour", "is_weekend", "days_since_min")
PIPELINE_KINDS = (SemanticType.NUMERIC, SemanticType.CATEGORICAL, SemanticType.BOOLEAN,
                  SemanticType.DATETIME)  # fmt: skip
NATIVE_MISSING = frozenset({ModelName.LIGHTGBM, ModelName.XGBOOST, ModelName.CATBOOST,
                            ModelName.HIST_GB})  # fmt: skip
STEP_NAMES = (
    "normalize_columns",
    "type_cast",
    "baseline_features",
    "dsl_features",
    "select",
    "preprocess",
    "model",
)


@dataclass(frozen=True)
class PipelineSchema:
    """Feature columns the pipeline consumes (normalized names), with DEV structure facts."""

    columns: dict[str, SemanticType]
    column_map: dict[str, str] = field(default_factory=dict)  # original → normalized
    distinct: dict[str, int] = field(default_factory=dict)
    missing_rate: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0


def schema_from(metadata: DatasetMetadata, summary: ProfileSummary) -> PipelineSchema:
    """Feature columns from ingestion metadata, with distinct counts and missing rates from the
    DEV profile (§9.2). Only modelable types are kept (§8.6 exclusions are already applied).
    """
    stats = {c.name: c for c in summary.columns}
    feats = [c for c in metadata.columns
             if c.role.value == "feature" and c.semantic_type in PIPELINE_KINDS]  # fmt: skip
    return PipelineSchema(
        columns={c.name: c.semantic_type for c in feats},
        column_map={c.original: c.name for c in feats},
        distinct={c.name: stats[c.name].distinct for c in feats if c.name in stats},
        missing_rate={c.name: stats[c.name].missing_rate for c in feats if c.name in stats},
        n_rows=summary.n_rows_dev,
    )


@dataclass
class _Branches:
    numeric: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    low: list[str] = field(default_factory=list)
    mid: list[str] = field(default_factory=list)
    high: list[str] = field(default_factory=list)

    def add_categorical(self, pattern: str, distinct: int) -> None:
        if distinct <= LOW_CARD_MAX:
            self.low.append(pattern)
        elif distinct <= MID_CARD_MAX:
            self.mid.append(pattern)
        else:
            self.high.append(pattern)


def _exact(name: str) -> str:
    return re.escape(name)


def _selector(patterns: list[str]) -> Any:
    return make_column_selector(pattern=f"^(?:{'|'.join(patterns)})$")


def _effective_types(candidate: CandidateSpec, schema: PipelineSchema) -> dict[str, SemanticType]:
    types = dict(schema.columns)
    for decision in candidate.column_decisions:
        if decision.column not in types:
            continue
        if decision.action == "drop":
            del types[decision.column]
        elif decision.action == "treat_as_categorical":
            types[decision.column] = SemanticType.CATEGORICAL
        elif decision.action == "treat_as_numeric":
            types[decision.column] = SemanticType.NUMERIC
    return types


def _llm_distinct(candidate: CandidateSpec, schema: PipelineSchema, name: str) -> int:
    spec = next(s for s in candidate.features if s.name == name)
    if spec.op == "bin_quantile":
        return int(spec.params.model_dump().get("n_bins") or MID_CARD_MAX)
    counts = [schema.distinct.get(ref, MID_CARD_MAX + 1) for ref in spec.inputs]
    return min(prod(counts), max(schema.n_rows, 1))


def _branches(
    candidate: CandidateSpec, schema: PipelineSchema, types: dict[str, SemanticType]
) -> _Branches:
    b = _Branches()
    for col, kind in types.items():
        if kind == SemanticType.NUMERIC:
            b.numeric.append(_exact(col))
        elif kind == SemanticType.BOOLEAN:
            b.boolean.append(_exact(col))
        elif kind == SemanticType.DATETIME:
            b.numeric.append(f"{_exact(col)}__(?:{'|'.join(DATETIME_PARTS)})")
        else:
            b.add_categorical(_exact(col), schema.distinct.get(col, MID_CARD_MAX + 1))
    if candidate.imputation.add_missing_indicators:
        for col in types:
            if schema.missing_rate.get(col, 0.0) >= MISSING_INDICATOR_RATE:
                b.boolean.append(f"{_exact(col)}__missing")
    for spec in candidate.features:
        pattern = f"{_exact(LLM_PREFIX + spec.name)}(?:__\\d+)?"
        output = CATALOG[spec.op].output
        if output == SemanticType.NUMERIC:
            b.numeric.append(pattern)
        elif output == SemanticType.BOOLEAN:
            b.boolean.append(pattern)
        else:
            b.add_categorical(pattern, _llm_distinct(candidate, schema, spec.name))
    return b


def _numeric_imputer(candidate: CandidateSpec) -> SimpleImputer:
    strategy = candidate.imputation.numeric
    if strategy == "constant_zero":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    return SimpleImputer(strategy=strategy)


def _categorical_imputer(candidate: CandidateSpec) -> SimpleImputer:
    if candidate.imputation.categorical == "constant_missing":
        return SimpleImputer(strategy="constant", fill_value=MISSING)
    return SimpleImputer(strategy="most_frequent")


def _ordinal() -> OrdinalEncoder:
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                          encoded_missing_value=-2)  # fmt: skip


def _high_card(candidate: CandidateSpec, task: Task, seed: int) -> Any:
    """Frequency or cross-fitted target encoding (§11.8): `TargetEncoder.fit_transform` encodes
    each training row out-of-fold; `transform` on validation rows uses the full training fold.
    """
    if candidate.encoding.high_cardinality == "frequency":
        return FrequencyEncoder()
    target_type = {Task.BINARY: "binary", Task.MULTICLASS: "multiclass"}.get(task, "continuous")
    splitter = StratifiedKFold if task != Task.REGRESSION else KFold
    cv = splitter(n_splits=TARGET_ENCODER_FOLDS, shuffle=True, random_state=seed)
    return TargetEncoder(target_type=target_type, smooth="auto", cv=cv)


def _transformers(
    candidate: CandidateSpec, b: _Branches, task: Task, seed: int
) -> list[tuple[str, Any, list[str]]]:
    model = candidate.model.name
    linear = model == ModelName.LINEAR
    imputes = model in (ModelName.LINEAR, ModelName.RANDOM_FOREST)
    steps: list[tuple[str, Any, list[str]]] = []
    if linear:
        steps.append(("num", Pipeline([("impute", _numeric_imputer(candidate)),
                                       ("scale", StandardScaler())]), b.numeric))  # fmt: skip
    elif imputes:
        steps.append(("num", _numeric_imputer(candidate), b.numeric))
    else:
        steps.append(("num", "passthrough", b.numeric))
    if imputes:
        steps.append(("bool", Pipeline([("encode", BooleanEncoder()),
                                        ("impute", SimpleImputer(strategy="most_frequent"))]),
                      b.boolean))  # fmt: skip
    else:
        steps.append(("bool", BooleanEncoder(), b.boolean))
    if model == ModelName.CATBOOST:  # raw categoricals, CatBoost encodes them itself (§15.2)
        steps.append(("cat", SimpleImputer(strategy="constant", fill_value=MISSING),
                      b.low + b.mid + b.high))  # fmt: skip
        return steps
    if linear and candidate.encoding.low_cardinality == "onehot":
        for name, cols, cap in (
            ("lowcat", b.low, None),
            ("midcat", b.mid, MID_CARD_MAX_CATEGORIES),
        ):
            onehot = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
                min_frequency=ONEHOT_MIN_FREQUENCY,
                max_categories=cap,
            )
            steps.append((name, Pipeline([("impute", _categorical_imputer(candidate)),
                                          ("encode", onehot)]), cols))  # fmt: skip
    elif linear:
        for name, cols in (("lowcat", b.low), ("midcat", b.mid)):
            steps.append((name, Pipeline([("encode", _ordinal()), ("scale", StandardScaler())]),
                          cols))  # fmt: skip
    else:
        steps.extend([("lowcat", _ordinal(), b.low), ("midcat", _ordinal(), b.mid)])
    high = _high_card(candidate, task, seed)
    steps.append(("highcat", Pipeline([("encode", high), ("scale", StandardScaler())])
                  if linear else high, b.high))  # fmt: skip
    return steps


def _categorical_prefixes(model: ModelName) -> tuple[str, ...]:
    if model == ModelName.LIGHTGBM:
        return ("lowcat__", "midcat__")
    if model == ModelName.CATBOOST:
        return ("cat__",)
    return ()


def build_pipeline(
    candidate: CandidateSpec,
    schema: PipelineSchema,
    registry: ModelRegistry,
    *,
    early_stopping: bool = True,
    profile: ExecutionProfileName | None = None,
) -> Pipeline | TransformedTargetRegressor:
    """The unfitted pipeline for `candidate` (§15.1–§15.2).

    `early_stopping` enables the inner 90/10 split for boosting models (CV); the final refit
    (P09) turns it off and sets `n_estimators` from the CV best iterations (§11.10).
    """
    task = candidate.model.task
    seed = candidate.model.seed
    types = _effective_types(candidate, schema)
    branches = _branches(candidate, schema, types)
    kinds = {col: kind.value for col, kind in types.items()}
    column_map = {orig: norm for orig, norm in schema.column_map.items() if norm in types}
    column_map.update({col: col for col in types if col not in column_map.values()})
    preprocess = ColumnTransformer(
        [(name, step, _selector(cols)) for name, step, cols in
         _transformers(candidate, branches, task, seed) if cols],
        remainder="drop",
    )  # fmt: skip
    model = TabularModel(
        registry.make_estimator(candidate.model, profile),
        family=candidate.model.name.value,
        categorical_prefixes=_categorical_prefixes(candidate.model.name),
        early_stopping=early_stopping,
        balance_scale_pos_weight=(candidate.model.name == ModelName.XGBOOST and task == Task.BINARY
                                  and candidate.model.class_weight.value == "balanced"),
        seed=seed,
    )  # fmt: skip
    indicators = candidate.imputation.add_missing_indicators
    baseline = BaselineFeatureTransformer(
        datetime_columns=[c for c, k in types.items() if k == SemanticType.DATETIME],
        missing_indicator_columns=[
            c for c in types
            if indicators and schema.missing_rate.get(c, 0.0) >= MISSING_INDICATOR_RATE
        ],
        categorical_columns=[c for c, k in types.items() if k == SemanticType.CATEGORICAL],
    )  # fmt: skip
    pipe = Pipeline([
        ("normalize_columns", ColumnNameMapper(column_map)),
        ("type_cast", TypeCaster(kinds)),
        ("baseline_features", baseline),
        ("dsl_features", DSLFeatureTransformer(runtime_specs(candidate.features))),
        ("select", ColumnSelector()),
        ("preprocess", preprocess),
        ("model", model),
    ]).set_output(transform="pandas")  # fmt: skip
    if task == Task.REGRESSION and candidate.target_transform == TargetTransform.LOG1P:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1,
                                          check_inverse=False)  # fmt: skip
    return pipe


def inner_pipeline(model: Pipeline | TransformedTargetRegressor) -> Pipeline:
    """The `Pipeline` inside a (fitted or unfitted) `TransformedTargetRegressor`, or `model`."""
    if isinstance(model, TransformedTargetRegressor):
        fitted = getattr(model, "regressor_", None)
        return fitted if fitted is not None else model.regressor
    return model
