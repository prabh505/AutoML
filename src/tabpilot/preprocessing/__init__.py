"""Pipeline factory (PRD §15, FR-PRE-01).

Interface: build_pipeline(candidate, schema, registry) -> Pipeline. Depends on features, models.

Must not: llm.
"""

from tabpilot.preprocessing.pipeline import (
    STEP_NAMES,
    PipelineSchema,
    build_pipeline,
    inner_pipeline,
    schema_from,
)

__all__ = ["STEP_NAMES", "PipelineSchema", "build_pipeline", "inner_pipeline", "schema_from"]
