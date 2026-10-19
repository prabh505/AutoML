"""Baseline + DSL feature transformers (PRD §14, FR-FE-*).

Interface: DSLFeatureTransformer, validate_spec. Pickled classes live in tabpilot_runtime
(A-001). Depends on schemas.

Must not: llm; evaluating strings as code.
"""
