"""Report and artifact-manifest contracts (PRD §21, §22.3, §26.11–§26.12)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from tabpilot.schemas._base import Component, Contract, NonNegFloat, NonNegInt, Sha256


class ExplanationsSection(Component):
    """Explanation summary in the report (PRD §21.3)."""

    method: Literal[
        "shap_tree", "shap_linear", "shap_member_plus_permutation", "permutation", "gain", "none"
    ]
    top_features: list[str] = Field(default_factory=list, max_length=20)
    unavailable_reason: str | None = Field(default=None, max_length=300)


class NarrativeSection(Component):
    """Narrative and its fact-check status (PRD §21.7)."""

    source: Literal["llm", "template"]
    fact_check: Literal["passed", "failed", "not_run"]
    text: str = Field(max_length=4000)

    @model_validator(mode="after")
    def _shipped_llm_text_passed(self) -> Self:
        """Only fact-checked LLM narratives are shipped (SM-7, NFR-EXP-02)."""
        if self.source == "llm" and self.fact_check != "passed":
            raise ValueError("an LLM narrative must pass the fact-check to be shipped")
        return self


class ReportSections(Component):
    """Report sections (PRD §21.1, §26.11)."""

    summary: dict[str, JsonValue]
    data_decisions: list[dict[str, JsonValue]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    plan: dict[str, JsonValue] = Field(default_factory=dict)
    leaderboard_ref: str
    selection: dict[str, JsonValue] = Field(default_factory=dict)
    holdout: dict[str, JsonValue] = Field(default_factory=dict)
    explanations: ExplanationsSection
    narrative: NarrativeSection
    limitations: list[str] = Field(default_factory=list)


class FinalReport(Contract):
    """Structured report (`report/1.0`, PRD §26.11)."""

    schema_version: Literal["report/1.0"] = "report/1.0"
    run_id: str
    status: Literal["completed", "completed_partial", "failed", "cancelled"]
    sections: ReportSections
    facts_ref: str


ManifestRole = Literal[
    "model", "code", "requirements", "metadata", "report", "figure", "parity", "docs",
    "runtime", "oof",
]  # fmt: skip


class ManifestFile(Component):
    """One bundle file with its checksum (PRD §22.3)."""

    path: str = Field(min_length=1, max_length=200)
    sha256: Sha256
    bytes: NonNegInt
    role: ManifestRole

    @field_validator("path")
    @classmethod
    def _relative_inside_bundle(cls, v: str) -> str:
        """Paths are relative, POSIX-style, and stay inside the bundle."""
        parts = v.split("/")
        if v.startswith("/") or "\\" in v or ".." in parts or "" in parts:
            raise ValueError("manifest paths must be relative paths inside the bundle")
        return v


class Reproducibility(Component):
    """Reproducibility facts (PRD §22.6)."""

    r1_verified: bool
    parity_max_abs_diff: NonNegFloat | None = None


class ArtifactManifest(Contract):
    """Bundle manifest (`manifest/1.0`, PRD §22.3, §26.12)."""

    schema_version: Literal["manifest/1.0"] = "manifest/1.0"
    run_id: str
    tabpilot_version: str
    created_at: datetime
    python: str
    platform: str
    seed_root: NonNegInt | None = None
    files: list[ManifestFile]
    reproducibility: Reproducibility
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_paths(self) -> Self:
        paths = [f.path for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("manifest paths must be unique")
        return self
