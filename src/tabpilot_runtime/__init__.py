"""Standalone runtime package for every custom class pickled into model.joblib (amendment A-001,
ADR-0002).

Shipped inside export bundles; imports only numpy, pandas, scikit-learn and model libraries.

Must not: Importing tabpilot (import contract R5).
"""

from tabpilot_runtime.columns import (
    BooleanEncoder,
    ColumnNameMapper,
    ColumnSelector,
    PredictionSchemaError,
    TypeCaster,
)
from tabpilot_runtime.estimators import TabularModel
from tabpilot_runtime.transformers import (
    BaselineFeatureTransformer,
    DSLFeatureTransformer,
    FrequencyEncoder,
    RareCategoryGrouper,
)

__all__ = [
    "BaselineFeatureTransformer",
    "BooleanEncoder",
    "ColumnNameMapper",
    "ColumnSelector",
    "DSLFeatureTransformer",
    "FrequencyEncoder",
    "PredictionSchemaError",
    "RareCategoryGrouper",
    "TabularModel",
    "TypeCaster",
]
