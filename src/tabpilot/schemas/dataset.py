"""Ingestion contracts: dataset metadata, parse/target reports, task detection, feasibility.

PRD §8, §10.1, §26.1, §29.4–§29.5; FR-ING-01…05, FR-TGT-01, FR-TGT-02.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from tabpilot.schemas._base import (
    ColumnName,
    ColumnRole,
    Component,
    Contract,
    Fraction,
    NonNegFloat,
    NonNegInt,
    RunId,
    SemanticType,
    ShortText,
    Simplification,
    Task,
)


class SourceInfo(Component):
    """Uploaded file facts. `display_name` is sanitized display text only (PRD §24.2)."""

    display_name: str = Field(max_length=120)
    format: Literal["csv", "tsv", "txt", "parquet"]
    size_bytes: NonNegInt
    encoding: str | None = None
    delimiter: str | None = Field(default=None, max_length=1)
    malformed_lines_skipped: NonNegInt = 0


class RowRemoval(Component):
    """A recorded row removal (no-silent-discard rule, PRD §8.6)."""

    reason: str = Field(min_length=1, max_length=80)
    count: NonNegInt


class ColumnInfo(Component):
    """One column's normalized name, type, role, and data decisions (FR-ING-03, FR-ING-04)."""

    original: str = Field(max_length=1000)
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,79}$")
    semantic_type: SemanticType
    role: ColumnRole
    dtype: str | None = None
    missing_rate: Fraction | None = None
    coerced_to_missing: NonNegInt = 0
    exclusion_reason: str | None = Field(default=None, max_length=120)
    hints: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _excluded_columns_have_a_reason(self) -> Self:
        """No column is dropped without a recorded `exclusion_reason` (FR-ING-04 AC)."""
        if (self.role == ColumnRole.EXCLUDED) != (self.exclusion_reason is not None):
            raise ValueError("exclusion_reason is required exactly when role is 'excluded'")
        return self


class TargetInfo(Component):
    """Target summary stored in `DatasetMetadata`."""

    name: ColumnName
    task: Task | None = None
    positive_label: str | None = None
    class_counts: dict[str, NonNegInt] | None = None


class DatasetMetadata(Contract):
    """Dataset metadata (`dataset/1.0`, PRD §26.1)."""

    schema_version: Literal["dataset/1.0"] = "dataset/1.0"
    run_id: RunId
    source: SourceInfo
    n_rows_raw: NonNegInt
    n_rows_valid: NonNegInt
    n_columns: NonNegInt
    row_removals: list[RowRemoval] = Field(default_factory=list)
    columns: list[ColumnInfo]
    target: TargetInfo | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Row and column counts agree with the recorded decisions."""
        if self.n_rows_valid > self.n_rows_raw:
            raise ValueError("n_rows_valid cannot exceed n_rows_raw")
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("normalized column names must be unique")
        if self.target is not None and self.target.name not in names:
            raise ValueError("target must be one of the columns")
        return self

    @property
    def column_map(self) -> dict[str, str]:
        """Original → normalized column names (FR-ING-03)."""
        return {c.original: c.name for c in self.columns}


class ParseReport(Contract):
    """Parsing outcome (`parse_report/1.0`, FR-ING-02)."""

    schema_version: Literal["parse_report/1.0"] = "parse_report/1.0"
    encoding: str
    encoding_fallback: bool = False
    delimiter: str | None = Field(default=None, max_length=1)
    quote_char: str | None = Field(default='"', max_length=1)
    decimal_comma_columns: list[str] = Field(default_factory=list)
    rows_read: NonNegInt
    malformed_lines: NonNegInt = 0
    malformed_line_numbers: list[NonNegInt] = Field(default_factory=list, max_length=5)
    warnings: list[str] = Field(default_factory=list)


class NumericSummary(Component):
    """Distribution summary for a numeric target."""

    mean: float
    std: NonNegFloat
    min: float
    max: float
    skew: float | None = None


class TargetReport(Contract):
    """Target validation outcome (`target_report/1.0`, FR-TGT-01, §8.7, A-008)."""

    schema_version: Literal["target_report/1.0"] = "target_report/1.0"
    target: ColumnName
    n_missing: NonNegInt
    n_rows_removed: NonNegInt
    n_unique: NonNegInt
    class_counts: dict[str, NonNegInt] | None = None
    distribution: NumericSummary | None = None
    warnings: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False


class TaskDetection(Contract):
    """Detected task (`task_detection/1.0`, FR-TGT-02, §10.1)."""

    schema_version: Literal["task_detection/1.0"] = "task_detection/1.0"
    task: Task
    confidence: Literal["high", "ambiguous"]
    reason: ShortText
    alternatives: list[Task] = Field(default_factory=list)
    positive_label: str | None = None

    @model_validator(mode="after")
    def _ambiguous_offers_alternatives(self) -> Self:
        """Ambiguous detections must show the other option(s) to the user (FR-TGT-02 AC)."""
        if self.confidence == "ambiguous" and not self.alternatives:
            raise ValueError("an ambiguous detection must list alternatives")
        if self.task in self.alternatives:
            raise ValueError("alternatives must differ from the detected task")
        return self


class FeasibilityReport(Contract):
    """Resource feasibility verdict (`feasibility/1.0`, FR-ING-05, §29.4–§29.5)."""

    schema_version: Literal["feasibility/1.0"] = "feasibility/1.0"
    status: Literal["ok", "simplified", "rejected"]
    estimated_peak_mb: NonNegFloat
    limit_mb: NonNegFloat
    simplifications: list[Simplification] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_simplifications(self) -> Self:
        """`simplified` lists what was applied; `ok` applied nothing."""
        if self.status == "ok" and self.simplifications:
            raise ValueError("status 'ok' cannot list simplifications")
        if self.status == "simplified" and not self.simplifications:
            raise ValueError("status 'simplified' must list simplifications")
        return self
