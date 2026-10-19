"""Baseline + DSL feature transformers (PRD §14, FR-FE-*).

Interface: DSLFeatureTransformer, validate_spec. Pickled classes live in tabpilot_runtime
(A-001). Depends on schemas.

Must not: llm; evaluating strings as code.
"""

from tabpilot.features.dsl.catalog import CATALOG, catalog_for_planner
from tabpilot.features.dsl.validate import (
    FeatureSchema,
    Issue,
    ValidationResult,
    canonical_form,
    validate_spec,
    validate_specs,
)
from tabpilot.features.provenance import baseline_provenance, provenance, runtime_specs
from tabpilot_runtime.transformers import BaselineFeatureTransformer, DSLFeatureTransformer

__all__ = [
    "CATALOG", "BaselineFeatureTransformer", "DSLFeatureTransformer", "FeatureSchema", "Issue",
    "ValidationResult", "baseline_provenance", "canonical_form", "catalog_for_planner",
    "provenance", "runtime_specs", "validate_spec", "validate_specs",
]  # fmt: skip
