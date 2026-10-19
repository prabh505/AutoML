"""Profiling contracts: full profile (local only) and LLM-safe compact profile.

PRD §9, §26.2; FR-PRF-01…03; A-005.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from tabpilot.schemas._base import (
    ColumnName,
    Component,
    Contract,
    Fraction,
    NonNegFloat,
    NonNegInt,
    SemanticType,
    Task,
)

AssociationKind = Literal["auc", "mi", "spearman", "r2"]


class Association(Component):
    """Feature–target association score (PRD §9.2)."""

    kind: AssociationKind
    value: float


class ColumnProfile(Component):
    """Full statistics for one DEV column (PRD §9.2). Never sent to the LLM."""

    name: ColumnName
    type: SemanticType
    missing_rate: Fraction
    distinct: NonNegInt
    # numeric
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p1: float | None = None
    p5: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    zero_rate: Fraction | None = None
    negative_rate: Fraction | None = None
    outlier_rate_iqr: Fraction | None = None
    outlier_rate_p1_p99: Fraction | None = None
    possibly_categorical: bool = False
    # categorical
    entropy_bits: NonNegFloat | None = None
    top_categories: list[str] | None = Field(default=None, max_length=10)
    top_shares: list[Fraction] | None = Field(default=None, max_length=10)
    rare_share: Fraction | None = None
    # datetime
    min_date: str | None = None
    max_date: str | None = None
    span_days: NonNegFloat | None = None
    monotonic_with_row_order: bool | None = None
    association: Association | None = None
    leakage_score: float | None = None


class TargetProfile(Component):
    """Target statistics on DEV (PRD §9.2)."""

    class_counts: dict[str, NonNegInt] | None = None
    imbalance_ratio: NonNegFloat | None = None
    positive_share: Fraction | None = None
    mean: float | None = None
    std: float | None = None
    skew: float | None = None
    min: float | None = None


class CorrelatedPair(Component):
    """A numeric feature pair with |ρ| ≥ 0.95 (PRD §9.2)."""

    a: ColumnName
    b: ColumnName
    rho: float = Field(ge=-1.0, le=1.0)


class LeakageFinding(Component):
    """Univariate leakage-scan result (PRD §11.7, A-002)."""

    column: ColumnName
    score: float
    metric: Literal["auc", "r2"]
    level: Literal["strong", "near_deterministic"]


class ProfileSummary(Contract):
    """Full DEV profile (`profile/1.0`, PRD §9.2). Local display only."""

    schema_version: Literal["profile/1.0"] = "profile/1.0"
    task: Task
    n_rows_dev: NonNegInt
    n_features: NonNegInt
    memory_mb: NonNegFloat
    duplicate_rows: NonNegInt
    conflicting_duplicate_rows: NonNegInt
    columns: list[ColumnProfile]
    target: TargetProfile
    correlated_pairs: list[CorrelatedPair] = Field(default_factory=list)
    leakage: list[LeakageFinding] = Field(default_factory=list)
    group_candidates: list[ColumnName] = Field(default_factory=list)
    temporal_columns: list[ColumnName] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    truncated_stats: list[str] = Field(default_factory=list)


class CompactTarget(Component):
    """Target aggregates in the compact profile (PRD §26.2)."""

    positive_share: Fraction | None = None
    imbalance_ratio: NonNegFloat | None = None
    n_classes: NonNegInt | None = None
    skew: float | None = None
    nonnegative: bool | None = None


class CompactColumn(Component):
    """One column in the compact profile: aggregates only, no cell labels (PRD §9.4, A-005).

    `q` = [min, p5, p25, p50, p75, p95, max]; absent in minimal mode.
    """

    name: ColumnName
    type: SemanticType
    level: Literal["full", "short"]
    missing: Fraction
    q: list[float] | None = Field(default=None, min_length=7, max_length=7)
    skew: float | None = None
    distinct: NonNegInt | None = None
    entropy_bits: NonNegFloat | None = None
    top_shares: list[Fraction] | None = Field(default=None, max_length=10)
    assoc: Association | None = None
    flags: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _short_entries_are_short(self) -> Self:
        """`short` entries hold only name, type, missing%, and association (PRD §9.4)."""
        detail = (self.q, self.skew, self.distinct, self.entropy_bits, self.top_shares)
        if self.level == "short" and any(v is not None for v in detail):
            raise ValueError("short entries carry only name, type, missing and assoc")
        return self


class RestSummary(Component):
    """Summary of columns beyond the detailed list (PRD §9.4)."""

    n_columns: NonNegInt
    counts_by_type: dict[SemanticType, NonNegInt]
    mean_missing: Fraction
    names: list[ColumnName] | None = Field(default=None, max_length=50)


class CompactProfile(Contract):
    """LLM-safe bounded profile (`compact_profile/1.0`, PRD §9.4, §26.2)."""

    schema_version: Literal["compact_profile/1.0"] = "compact_profile/1.0"
    task: Task
    n_rows_dev: NonNegInt
    n_features: NonNegInt
    target: CompactTarget
    columns: list[CompactColumn]
    rest_summary: RestSummary | None = None
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False
    mode: Literal["standard", "minimal"] = "standard"

    @model_validator(mode="after")
    def _minimal_mode_has_no_quantiles(self) -> Self:
        """Minimal mode replaces quantiles with skew and range class only (PRD §9.4)."""
        if self.mode == "minimal" and any(c.q is not None for c in self.columns):
            raise ValueError("minimal mode must not include quantiles")
        return self
