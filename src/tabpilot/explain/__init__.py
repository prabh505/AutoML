"""SHAP, permutation importance, error analysis (PRD §21.3-§21.5, FR-EXP-01).

Interface: explain(final, data, settings), error_analysis(oof, y, dev_frame, problem).

Must not: llm; running without a time cap.
"""

from tabpilot.explain.errors import (
    ClassMetrics,
    DecileError,
    ErrorAnalysis,
    Segment,
    error_analysis,
    worst_segments,
)
from tabpilot.explain.explainer import (
    EXPLAIN_TIMEOUT_S,
    ExplainData,
    ExplainSettings,
    ExplainTimeoutError,
    ExplanationResult,
    Importance,
    ShapOutput,
    ShapUnsupportedError,
    aggregate,
    explain,
    gain,
    permutation,
    sample_rows,
    shap_values,
)

__all__ = [
    "EXPLAIN_TIMEOUT_S",
    "ClassMetrics",
    "DecileError",
    "ErrorAnalysis",
    "ExplainData",
    "ExplainSettings",
    "ExplainTimeoutError",
    "ExplanationResult",
    "Importance",
    "Segment",
    "ShapOutput",
    "ShapUnsupportedError",
    "aggregate",
    "error_analysis",
    "explain",
    "gain",
    "permutation",
    "sample_rows",
    "shap_values",
    "worst_segments",
]
