"""Feature DSL contracts (`feature_spec/1.0`) and feature provenance (PRD §14.3–§14.6, §26.5).

`FeatureSpec` is a discriminated union on `op`. Each op fixes its input arity and has its own
parameter model with the exact bounds of §14.3. Parameters use strict numeric types, so strings
(and booleans) are rejected: "no strings except where declared" (§14.4).
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, TypeAdapter, field_validator

from tabpilot.schemas._base import Component, Contract, NonNegInt

FEATURE_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,59}$"
MAX_FEATURES_PER_PLAN = 12
MAX_FEATURES_PER_REFLECTION = 6
MAX_ACTIVE_LLM_FEATURES = 12  # A-010

InputRef = Annotated[str, Field(min_length=1, max_length=80)]
# StrictFloat accepts ints but rejects booleans and strings.
StrictNumber = StrictFloat


class NoParams(Component):
    """Parameters for ops that take none (any key is rejected)."""


class RatioParams(Component):
    """`ratio`: denominator threshold (PRD §14.3: eps 1e-9 … 1)."""

    eps: StrictNumber = Field(default=1e-6, ge=1e-9, le=1.0)


class ClipParams(Component):
    """`clip`: quantile bounds learned per fold (PRD §14.3)."""

    lower_q: StrictNumber = Field(ge=0.0, le=0.2)
    upper_q: StrictNumber = Field(ge=0.8, le=1.0)


class BinQuantileParams(Component):
    """`bin_quantile`: number of bins 2–20 (PRD §14.3)."""

    n_bins: StrictInt = Field(ge=2, le=20)


class EqualsParams(Component):
    """`equals`: numeric value only; categorical labels are never sent to the LLM (§14.3)."""

    value: StrictNumber

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return v


class _SpecBase(Contract):
    schema_version: Literal["feature_spec/1.0"] = "feature_spec/1.0"
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,15}$")
    name: str = Field(pattern=FEATURE_NAME_PATTERN)
    rationale: str = Field(default="", max_length=300)


class BinaryArithmeticSpec(_SpecBase):
    """`add`, `subtract`, `multiply`: 2 numeric inputs."""

    op: Literal["add", "subtract", "multiply"]
    inputs: list[InputRef] = Field(min_length=2, max_length=2)
    params: NoParams = Field(default_factory=NoParams)


class RatioSpec(_SpecBase):
    """`ratio`: 2 numeric inputs; |denominator| < eps → NaN."""

    op: Literal["ratio"]
    inputs: list[InputRef] = Field(min_length=2, max_length=2)
    params: RatioParams = Field(default_factory=RatioParams)


class UnaryNumericSpec(_SpecBase):
    """`log1p`, `sqrt`, `square`: 1 numeric input (negatives → NaN for log1p/sqrt, A-007)."""

    op: Literal["log1p", "sqrt", "square"]
    inputs: list[InputRef] = Field(min_length=1, max_length=1)
    params: NoParams = Field(default_factory=NoParams)


class ClipSpec(_SpecBase):
    """`clip`: 1 numeric input; stateful."""

    op: Literal["clip"]
    inputs: list[InputRef] = Field(min_length=1, max_length=1)
    params: ClipParams


class BinQuantileSpec(_SpecBase):
    """`bin_quantile`: 1 numeric input → categorical; stateful."""

    op: Literal["bin_quantile"]
    inputs: list[InputRef] = Field(min_length=1, max_length=1)
    params: BinQuantileParams


class IsMissingSpec(_SpecBase):
    """`is_missing`: 1 input of any type → boolean."""

    op: Literal["is_missing"]
    inputs: list[InputRef] = Field(min_length=1, max_length=1)
    params: NoParams = Field(default_factory=NoParams)


class EqualsSpec(_SpecBase):
    """`equals`: 1 numeric/boolean input compared with a numeric value → boolean."""

    op: Literal["equals"]
    inputs: list[InputRef] = Field(min_length=1, max_length=1)
    params: EqualsParams


class RowAggregateSpec(_SpecBase):
    """`count_missing` (any), `row_sum`, `row_mean` (numeric): 2–20 inputs."""

    op: Literal["count_missing", "row_sum", "row_mean"]
    inputs: list[InputRef] = Field(min_length=2, max_length=20)
    params: NoParams = Field(default_factory=NoParams)


class ConcatCategoriesSpec(_SpecBase):
    """`concat_categories`: 2–3 categorical inputs."""

    op: Literal["concat_categories"]
    inputs: list[InputRef] = Field(min_length=2, max_length=3)
    params: NoParams = Field(default_factory=NoParams)


class GroupStatSpec(_SpecBase):
    """`group_mean`, `group_std`, `group_deviation`: [categorical key, numeric value]; stateful."""

    op: Literal["group_mean", "group_std", "group_deviation"]
    inputs: list[InputRef] = Field(min_length=2, max_length=2)
    params: NoParams = Field(default_factory=NoParams)


class GroupCountSpec(_SpecBase):
    """`group_count`: [categorical key]; stateful."""

    op: Literal["group_count"]
    inputs: list[InputRef] = Field(min_length=1, max_length=1)
    params: NoParams = Field(default_factory=NoParams)


class DateDiffSpec(_SpecBase):
    """`date_diff_days`: 2 datetime inputs."""

    op: Literal["date_diff_days"]
    inputs: list[InputRef] = Field(min_length=2, max_length=2)
    params: NoParams = Field(default_factory=NoParams)


FeatureSpec = Annotated[
    BinaryArithmeticSpec
    | RatioSpec
    | UnaryNumericSpec
    | ClipSpec
    | BinQuantileSpec
    | IsMissingSpec
    | EqualsSpec
    | RowAggregateSpec
    | ConcatCategoriesSpec
    | GroupStatSpec
    | GroupCountSpec
    | DateDiffSpec,
    Field(discriminator="op"),
]
FEATURE_SPEC_ADAPTER: TypeAdapter[FeatureSpec] = TypeAdapter(FeatureSpec)

DSL_OPS: tuple[str, ...] = (
    "add", "subtract", "multiply", "ratio", "log1p", "sqrt", "square", "clip",
    "bin_quantile", "is_missing", "equals", "count_missing", "row_sum", "row_mean",
    "concat_categories", "group_mean", "group_std", "group_count", "group_deviation",
    "date_diff_days",
)  # fmt: skip


class FeatureProvenance(Contract):
    """Provenance of one feature (`feature_provenance/1.0`, PRD §14.6)."""

    schema_version: Literal["feature_provenance/1.0"] = "feature_provenance/1.0"
    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=80)
    spec: FeatureSpec | None = None
    source: Literal["llm_plan", "llm_reflection", "baseline"]
    iteration: NonNegInt = Field(le=2)
    status: Literal["active", "failed", "degenerate", "rejected_static"]
    failure_reason: str | None = Field(default=None, max_length=300)
    cv_contribution: float | None = None
