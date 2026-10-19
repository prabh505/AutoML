"""Static validation of Feature DSL specs before execution (PRD §14.4; A-007, A-010).

The planner (P11) and the reflector (P12) reuse this module; they must not reimplement it.
Structural bounds (op, arity, parameter ranges, name pattern) are already enforced by the
`FeatureSpec` contract. This module adds the dataset-dependent checks. Issue codes match the plan
fixtures in `fixtures/MANIFEST.yaml` (DEC-025).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tabpilot.features.dsl.catalog import CATALOG
from tabpilot.schemas import FeatureSpec, SemanticType
from tabpilot.schemas.features import MAX_FEATURES_PER_PLAN

MAX_DEPTH = 2
BYTES_PER_VALUE = 4
UNSAFE = re.compile(r"(://|/|\\|\.\.|[();]|\bimport\b|\bos\.|__\w*__)")


@dataclass(frozen=True)
class FeatureSchema:
    """What a spec may reference: feature columns with types, plus forbidden names."""

    columns: dict[str, SemanticType]
    target: str
    excluded: frozenset[str] = frozenset()
    minimums: dict[str, float] = field(default_factory=dict)  # profile min per numeric column
    n_rows: int = 0
    memory_budget_mb: float = float("inf")


@dataclass(frozen=True)
class Issue:
    """One validation finding. `fatal` issues reject the spec; others are warnings."""

    feature: str
    code: str
    message: str
    fatal: bool = True


@dataclass(frozen=True)
class ValidationResult:
    """Outcome for one spec."""

    ok: bool
    issues: list[Issue]
    output_type: SemanticType | None = None
    depth: int = 0


def canonical_form(spec: FeatureSpec) -> tuple[object, ...]:
    """Op + inputs (sorted for commutative ops) + params, for duplicate detection (§14.4)."""
    sig = CATALOG[spec.op]
    inputs = tuple(sorted(spec.inputs)) if sig.commutative else tuple(spec.inputs)
    params = tuple(sorted(spec.params.model_dump().items()))
    return (spec.op, inputs, params)


def _check_ref(
    spec: FeatureSpec, ref: str, schema: FeatureSchema, earlier: dict[str, tuple[SemanticType, int]]
) -> tuple[SemanticType | None, int, Issue | None]:
    if UNSAFE.search(ref):
        return None, 0, Issue(spec.name, "unsafe_feature", "input looks like a path, URL, or code")
    if ref == schema.target:
        return None, 0, Issue(spec.name, "target_column", "features may not use the target")
    if ref in schema.excluded:
        return None, 0, Issue(spec.name, "excluded_column", f"input {ref!r} is excluded")
    if ref in earlier:
        kind, depth = earlier[ref]
        return kind, depth, None
    if ref in schema.columns:
        return schema.columns[ref], 0, None
    return None, 0, Issue(spec.name, "unknown_column", f"input {ref!r} does not exist")


def validate_spec(
    spec: FeatureSpec,
    schema: FeatureSchema,
    earlier: dict[str, tuple[SemanticType, int]] | None = None,
) -> ValidationResult:
    """Validate one spec against the dataset schema and earlier accepted specs (§14.4)."""
    earlier = earlier or {}
    sig = CATALOG[spec.op]
    issues: list[Issue] = []
    if spec.name in schema.columns or spec.name in earlier:
        return ValidationResult(
            False,
            [
                Issue(
                    spec.name,
                    "duplicate_feature",
                    "name collides with an existing column or feature",
                )
            ],
        )
    depth = 1  # a spec on raw columns has depth 1
    for position, ref in enumerate(spec.inputs):
        kind, ref_depth, issue = _check_ref(spec, ref, schema, earlier)
        if issue:
            issues.append(issue)
            continue
        allowed = sig.inputs[min(position, len(sig.inputs) - 1)]
        if kind not in allowed:
            issues.append(
                Issue(
                    spec.name,
                    "feature_type_mismatch",
                    f"{spec.op} input {position + 1} must be {sorted(t.value for t in allowed)}",
                )
            )
        if ref in earlier:
            depth = max(depth, ref_depth + 1)
    if depth > MAX_DEPTH:
        issues.append(Issue(spec.name, "depth_exceeded", f"composition depth > {MAX_DEPTH}"))
    if spec.op in ("log1p", "sqrt"):
        ref = spec.inputs[0]
        if schema.minimums.get(ref, 0.0) < 0:
            issues.append(
                Issue(
                    spec.name,
                    "negative_inputs",
                    "input has negative values; they become missing (A-007)",
                    fatal=False,
                )
            )
    fatal = any(i.fatal for i in issues)
    return ValidationResult(not fatal, issues, None if fatal else sig.output, depth)


def validate_specs(
    specs: list[FeatureSpec], schema: FeatureSchema, max_count: int = MAX_FEATURES_PER_PLAN
) -> tuple[list[FeatureSpec], list[Issue]]:
    """Validate a list in order; return the accepted specs (deduped, capped) and all issues.

    Later specs may reference earlier accepted ones by `name` (depth ≤ 2). A spec that depends on
    a rejected spec is rejected too.
    """
    accepted: list[FeatureSpec] = []
    issues: list[Issue] = []
    earlier: dict[str, tuple[SemanticType, int]] = {}
    seen: set[tuple[object, ...]] = set()
    added_mb = 0.0
    for spec in specs:
        form = canonical_form(spec)
        if form in seen:
            issues.append(Issue(spec.name, "duplicate_feature", "same op, inputs, and params"))
            continue
        result = validate_spec(spec, schema, earlier)
        issues += result.issues
        if not result.ok or result.output_type is None:
            continue
        if len(accepted) >= max_count:
            issues.append(Issue(spec.name, "too_many_features", f"limit is {max_count}"))
            continue
        cost = schema.n_rows * BYTES_PER_VALUE / 1e6
        if added_mb + cost > schema.memory_budget_mb:
            issues.append(Issue(spec.name, "cost_exceeded", "feature exceeds the memory budget"))
            continue
        added_mb += cost
        seen.add(form)
        earlier[spec.name] = (result.output_type, result.depth)
        accepted.append(spec)
    return accepted, issues
