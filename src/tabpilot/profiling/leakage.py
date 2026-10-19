"""Univariate leakage scan (PRD §11.7; A-002).

For each feature on a ≤ 20K-row DEV sample, a depth-3 decision tree is fitted on that feature
alone with 3-fold CV. Categoricals are frequency-encoded inside each fold. The score is ROC-AUC
(binary), macro one-vs-rest AUC (multiclass), or R² (regression).

Thresholds (A-002): classification near-deterministic ≥ 0.95, strong ≥ 0.90; regression R²
near-deterministic ≥ 0.90, strong ≥ 0.80. A strong association is evidence for review, not
proof of leakage. Leakage through feature combinations or timing is invisible to this test (§11.7).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tabpilot.profiling.budget import GroupTimer
from tabpilot.schemas import ColumnRole, DatasetMetadata, SemanticType, Task
from tabpilot.schemas.profile import LeakageFinding
from tabpilot.schemas.run import ColumnOverride

SAMPLE_ROWS = 20_000
MAX_FEATURES = 200
FOLDS = 3
THRESHOLDS = {  # (strong, near_deterministic), A-002
    "auc": (0.90, 0.95),
    "r2": (0.80, 0.90),
}
EXCLUSION_REASON = "leakage_near_deterministic"


class FoldFrequencyEncoder(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Maps category → training-fold frequency. Unseen categories → 0. Fitted per fold."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> FoldFrequencyEncoder:
        col = X.iloc[:, 0].astype("string")
        self.freq_ = col.value_counts(normalize=True, dropna=False).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        col = X.iloc[:, 0].astype("string")
        return np.asarray(col.map(self.freq_).fillna(0.0), dtype=np.float64).reshape(-1, 1)


def _as_numeric(values: pd.Series, semantic: SemanticType) -> pd.DataFrame:
    if semantic == SemanticType.DATETIME:
        days = values.astype("datetime64[ns]").astype("int64") / 86_400e9
        return days.where(values.notna()).to_frame()
    return values.astype("float64").to_frame()


def _score(x: pd.DataFrame, y: np.ndarray, task: Task, categorical: bool, seed: int) -> float:
    if task == Task.REGRESSION:
        tree: Any = DecisionTreeRegressor(max_depth=3, random_state=seed)
        cv: Any = KFold(FOLDS, shuffle=True, random_state=seed)
        scoring = "r2"
    else:
        tree = DecisionTreeClassifier(max_depth=3, random_state=seed)
        cv = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
        scoring = "roc_auc" if task == Task.BINARY else "roc_auc_ovr"
    model = make_pipeline(FoldFrequencyEncoder(), tree) if categorical else tree
    return float(np.mean(cross_val_score(model, x, y, cv=cv, scoring=scoring)))


def scan(  # noqa: PLR0917 - the scan's inputs are all required
    dev: pd.DataFrame,
    target: str,
    metadata: DatasetMetadata,
    task: Task,
    seed: int,
    timer: GroupTimer,
) -> tuple[list[LeakageFinding], dict[str, float]]:
    """Scan feature columns of `dev`. Returns findings at/above the strong level and all scores."""
    rng = np.random.default_rng(seed)
    rows = dev.index.to_numpy()
    if len(rows) > SAMPLE_ROWS:
        rows = np.sort(rng.choice(rows, SAMPLE_ROWS, replace=False))
    sample = dev.loc[rows]
    y_raw = sample[target]
    y = y_raw.astype("float64").to_numpy() if task == Task.REGRESSION else pd.factorize(
        y_raw.astype("string"))[0]  # fmt: skip
    metric = "r2" if task == Task.REGRESSION else "auc"
    strong, near = THRESHOLDS[metric]
    features = [c for c in metadata.columns if c.role == ColumnRole.FEATURE and c.name != target]
    findings: list[LeakageFinding] = []
    scores: dict[str, float] = {}
    for column in features[:MAX_FEATURES]:
        if timer.expired():
            break
        categorical = column.semantic_type in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN)
        x = (sample[[column.name]] if categorical
             else _as_numeric(sample[column.name], column.semantic_type))  # fmt: skip
        value = _score(x, y, task, categorical, seed)
        scores[column.name] = value
        if value >= strong:
            level = "near_deterministic" if value >= near else "strong"
            findings.append(LeakageFinding(column=column.name, score=round(value, 6), metric=metric,
                                           level=level))  # fmt: skip
    return findings, scores


def apply_leakage_exclusions(
    metadata: DatasetMetadata, findings: list[LeakageFinding], overrides: list[ColumnOverride]
) -> DatasetMetadata:
    """Exclude near-deterministic columns by default; a user override re-includes them (§11.7)."""
    keep = {o.column for o in overrides if o.role == ColumnRole.FEATURE}
    exclude = {f.column for f in findings if f.level == "near_deterministic"} - keep
    columns = [
        c.model_copy(update={"role": ColumnRole.EXCLUDED, "exclusion_reason": EXCLUSION_REASON})
        if c.name in exclude and c.role == ColumnRole.FEATURE
        else c
        for c in metadata.columns
    ]
    return metadata.model_copy(update={"columns": columns})
