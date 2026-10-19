"""Feature provenance records (PRD §14.6, §26.5)."""

from __future__ import annotations

from typing import Literal

from tabpilot.features.dsl.validate import Issue
from tabpilot.schemas import FeatureProvenance, FeatureSpec
from tabpilot_runtime.transformers import DSLFeatureTransformer


def provenance(
    specs: list[FeatureSpec],
    rejected: list[Issue],
    fitted: DSLFeatureTransformer | None,
    source: Literal["llm_plan", "llm_reflection"],
    iteration: int,
) -> list[FeatureProvenance]:
    """One record per proposed feature: active, failed, degenerate, or rejected_static."""
    records: list[FeatureProvenance] = []
    failed = fitted.failed_features_ if fitted is not None else {}
    for spec in specs:
        status: Literal["active", "failed", "degenerate", "rejected_static"] = "active"
        reason = failed.get(spec.name)
        if reason == "degenerate":
            status = "degenerate"
        elif reason:
            status = "failed"
        records.append(
            FeatureProvenance(
                id=spec.id,
                name=spec.name,
                spec=spec,
                source=source,
                iteration=iteration,
                status=status,
                failure_reason=reason,
            )
        )
    accepted = {s.name for s in specs}
    for issue in rejected:
        if issue.fatal and issue.feature not in accepted:
            records.append(
                FeatureProvenance(
                    id=f"rejected_{len(records)}",
                    name=issue.feature,
                    source=source,
                    iteration=iteration,
                    status="rejected_static",
                    failure_reason=f"{issue.code}: {issue.message}"[:300],
                )
            )
    return records


def baseline_provenance(columns: list[str]) -> list[FeatureProvenance]:
    """Records for deterministic baseline features (FR-FE-01)."""
    return [
        FeatureProvenance(
            id=f"b{i:03d}", name=name, source="baseline", iteration=0, status="active"
        )
        for i, name in enumerate(columns)
    ]


def runtime_specs(specs: list[FeatureSpec]) -> list[dict[str, object]]:
    """Plain-dict specs for `DSLFeatureTransformer` (the runtime package cannot import schemas)."""
    return [
        {
            "id": s.id,
            "name": s.name,
            "op": s.op,
            "inputs": list(s.inputs),
            "params": s.params.model_dump(),
        }
        for s in specs
    ]
