"""Model explanations under a time cap (PRD §21.3, §21.5; FR-EXP-01).

Method selection (§21.3):

```
tree model (LightGBM, XGBoost, CatBoost, RF)  → shap.TreeExplainer on the model step
linear model                                   → shap.LinearExplainer on the model step
ensemble       → SHAP of the highest-weight member + permutation importance of the ensemble
any, on SHAP failure or timeout → permutation importance of the full pipeline
                                  (one DEV validation fold, 5 repeats, primary metric)
still failing or out of time    → native gain importance (tree models), else none
```

Native gain importance is always computed for tree models.

SHAP and permutation calls cannot be interrupted safely, so the time cap is enforced before
each call. A timed batch of 50 rows (one scorer call for permutation) is extrapolated to the
full work, with a 1.5× margin. If the estimate does not fit in the time left, the next method
is used instead (DEC-037). The worker's hard wall-clock kill (P15) is the final backstop.

SHAP runs on ≤ 1,000 DEV rows sampled with `seed_explain` (≤ 300 for random forest on hosted).
Preprocessed columns are folded back into their source columns for the global chart (one-hot
columns are summed per row before taking |SHAP|). Multiclass global importance is the mean
|SHAP| across all classes; per-class values are kept for the 3 most frequent classes.
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import Problem, score, selection_score
from tabpilot.preprocessing import inner_pipeline
from tabpilot.reflection.input import raw_importances, source_column
from tabpilot.schemas import ExecutionProfileName, MetricName, Task
from tabpilot.schemas.report import ExplanationsSection
from tabpilot_runtime import VotingEnsemble

Method = Literal[
    "shap_tree", "shap_linear", "shap_member_plus_permutation", "permutation", "gain", "none"
]  # fmt: skip
TREE_FAMILIES = frozenset({"lightgbm", "xgboost", "catboost", "random_forest"})
MAX_ROWS = 1000
MAX_ROWS_RF_HOSTED = 300
PERMUTATION_REPEATS = 5
TOP_FEATURES = 20
TOP_CLASSES = 3
TIMING_BATCH = 50
SAFETY = 1.5
REASON_CHARS = 300
# §13.4 explain node timeout, doubled by A-004 (ADR-0003): 120 s hosted, 600 s local.
EXPLAIN_TIMEOUT_S = {ExecutionProfileName.HOSTED: 120.0, ExecutionProfileName.LOCAL: 600.0}


class ExplainTimeoutError(Exception):
    """The pre-estimate says a method does not fit in the time left."""


class ShapUnsupportedError(Exception):
    """The model family has no SHAP explainer here (e.g. histogram gradient boosting)."""


@dataclass(frozen=True)
class Importance:
    """Importance of one source column (`std` over permutation repeats; 0 otherwise)."""

    name: str
    value: float
    std: float = 0.0


@dataclass(frozen=True)
class ShapOutput:
    """Per-row SHAP values of one model output on the preprocessed columns.

    `label` is "" for a single output and the class label for a multiclass output. Class
    labels stay local: they are rendered in the report and never sent to an LLM (PI-16).
    `data` holds the feature values used to colour the beeswarm (NaN for non-numeric columns).
    """

    label: str
    columns: list[str]
    values: NDArray[np.float64]
    data: NDArray[np.float64]


@dataclass(frozen=True)
class ExplainSettings:
    """Run facts the explainer needs (§21.3)."""

    problem: Problem
    metric: MetricName
    sources: list[str]
    seed: int
    profile: ExecutionProfileName
    time_cap_s: float
    clock: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class ExplainData:
    """DEV rows (raw feature columns), the encoded target, and one validation fold's positions."""

    frame: pd.DataFrame
    y: NDArray[Any]
    valid: NDArray[np.int64]


@dataclass
class ExplanationResult:
    """What the explainer produced; failed methods leave their lists empty (§21.5)."""

    method: Method
    global_importance: list[Importance] = field(default_factory=list)
    gain: list[Importance] = field(default_factory=list)
    permutation: list[Importance] = field(default_factory=list)
    shap: list[ShapOutput] = field(default_factory=list)
    n_rows: int = 0
    unavailable_reason: str | None = None
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def top_features(self) -> list[str]:
        """Source columns by importance: SHAP/permutation when available, else gain."""
        ranked = self.global_importance or self.permutation or self.gain
        return [i.name for i in ranked[:TOP_FEATURES]]

    def section(self) -> ExplanationsSection:
        """The `report/1.0` explanations section."""
        reason = self.unavailable_reason[:REASON_CHARS] if self.unavailable_reason else None
        return ExplanationsSection(method=self.method, top_features=self.top_features(),
                                   unavailable_reason=reason)  # fmt: skip


def sample_rows(n: int, *, seed: int, family: str, profile: ExecutionProfileName) -> NDArray[Any]:
    """Sorted positions of ≤ 1,000 rows (≤ 300 for random forest on hosted), seeded (§21.3)."""
    hosted_rf = family == "random_forest" and profile == ExecutionProfileName.HOSTED
    limit = MAX_ROWS_RF_HOSTED if hosted_rf else MAX_ROWS
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=limit, replace=False))


def aggregate(values: dict[str, float], sources: Sequence[str]) -> list[Importance]:
    """Sum preprocessed-column importances into source columns; positive ones, descending."""
    totals: dict[str, float] = defaultdict(float)
    for column, value in values.items():
        src = source_column(column, sources)
        if src is not None:
            totals[src] += float(value)
    ranked = sorted(((n, v) for n, v in totals.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))
    return [Importance(n, v) for n, v in ranked[:TOP_FEATURES]]


def _family(pipe: Any) -> str:
    return str(getattr(inner_pipeline(pipe).steps[-1][1], "family", ""))


def _check_time(estimate: float, deadline: float, clock: Callable[[], float], what: str) -> None:
    left = deadline - clock()
    if estimate * SAFETY > left:
        raise ExplainTimeoutError(f"{what} pre-estimate {estimate * SAFETY:.0f} s exceeds the "
                             f"{max(left, 0.0):.0f} s left")  # fmt: skip


def _explainer(family: str, est: Any, X: pd.DataFrame) -> tuple[Any, Method]:
    import shap  # noqa: PLC0415 - heavy import, only when explaining

    if family in TREE_FAMILIES:
        return shap.TreeExplainer(est), "shap_tree"
    if family == "linear":
        return shap.LinearExplainer(est, X), "shap_linear"
    raise ShapUnsupportedError(f"no SHAP explainer for the {family} family")


def _outputs(raw: Any, problem: Problem) -> list[NDArray[np.float64]]:
    """SHAP output → one (n, p) array per model output, in `problem.classes` order.

    Binary: one array for the positive class (2-D outputs explain class code 1, so they are
    negated when the positive class has code 0).
    """
    arr = np.stack(raw, axis=-1) if isinstance(raw, list) else np.asarray(raw)
    arr = arr.astype(np.float64)
    if problem.task == Task.REGRESSION:
        return [arr.reshape(arr.shape[0], -1)]
    if problem.task == Task.BINARY:
        if arr.ndim == 3:
            return [arr[:, :, problem.positive_index]]
        return [arr if problem.positive_index == 1 else -arr]
    if arr.ndim != 3 or arr.shape[2] != problem.n_classes:
        raise ValueError("multiclass SHAP output does not have one slice per class")
    return [arr[:, :, k] for k in range(problem.n_classes)]


def _by_source(values: NDArray[np.float64], columns: Sequence[str],
               sources: Sequence[str]) -> dict[str, float]:  # fmt: skip
    """Mean over rows of |sum of a source's SHAP columns| (one-hot columns summed per row)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for j, column in enumerate(columns):
        src = source_column(column, sources)
        if src is not None:
            groups[src].append(j)
    return {s: float(np.abs(values[:, idx].sum(axis=1)).mean()) for s, idx in groups.items()}


def _numeric(frame: pd.DataFrame) -> NDArray[np.float64]:
    out = np.full(frame.shape, np.nan)
    for j, column in enumerate(frame.columns):
        values = pd.to_numeric(frame[column], errors="coerce")
        out[:, j] = values.to_numpy(dtype=np.float64, na_value=np.nan)
    return out


def shap_values(
    pipe: Any, data: ExplainData, settings: ExplainSettings, deadline: float
) -> tuple[Method, list[ShapOutput], list[Importance], list[str]]:
    """SHAP for one fitted pipeline: (method, kept outputs, global source importance, notes)."""
    inner = inner_pipeline(pipe)
    tab = inner.steps[-1][1]
    rows = sample_rows(len(data.frame), seed=settings.seed, family=tab.family,
                       profile=settings.profile)  # fmt: skip
    X = tab.prepared(inner[:-1].transform(data.frame.iloc[rows]))
    explainer, method = _explainer(tab.family, tab.estimator_, X)
    clock = settings.clock
    batch = min(TIMING_BATCH, len(X))
    start = clock()
    with warnings.catch_warnings():  # shap announces output-format changes as UserWarnings
        warnings.simplefilter("ignore", UserWarning)
        explainer.shap_values(X.iloc[:batch])
        _check_time((clock() - start) * len(X) / batch, deadline, clock, "SHAP")
        raw = explainer.shap_values(X)
    outputs = _outputs(raw, settings.problem)
    columns = [str(c) for c in X.columns]
    per_output = [_by_source(v, columns, settings.sources) for v in outputs]
    totals = {s: float(np.mean([p.get(s, 0.0) for p in per_output])) for s in per_output[0]}
    ranked = sorted(((n, v) for n, v in totals.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))
    importance = [Importance(n, v) for n, v in ranked[:TOP_FEATURES]]
    data_matrix = _numeric(X)
    if settings.problem.task == Task.MULTICLASS:
        counts = np.bincount(np.asarray(data.y[rows], dtype=np.int64),
                             minlength=settings.problem.n_classes)  # fmt: skip
        top = sorted(range(len(counts)), key=lambda k: (-counts[k], k))[:TOP_CLASSES]
        kept = [ShapOutput(settings.problem.classes[k], columns, outputs[k], data_matrix)
                for k in top]  # fmt: skip
    else:
        kept = [ShapOutput("", columns, outputs[0], data_matrix)]
    notes = []
    if inner is not pipe:
        notes.append("SHAP values are on the log1p scale of the target.")
    return method, kept, importance, notes


def _scorer(settings: ExplainSettings) -> Callable[[Any, pd.DataFrame, NDArray[Any]], float]:
    problem, metric = settings.problem, settings.metric

    def scorer(model: Any, X: pd.DataFrame, y: NDArray[Any]) -> float:
        pred = model.predict(X) if problem.task == Task.REGRESSION else model.predict_proba(X)
        return selection_score(metric, score(metric, y, np.asarray(pred), problem))

    return scorer


def permutation(
    model: Any, data: ExplainData, settings: ExplainSettings, deadline: float
) -> list[Importance]:
    """Permutation importance of the full pipeline on one DEV validation fold, 5 repeats."""
    from sklearn.inspection import permutation_importance  # noqa: PLC0415

    X = data.frame.iloc[data.valid]
    y = np.asarray(data.y)[data.valid]
    scorer = _scorer(settings)
    clock = settings.clock
    start = clock()
    scorer(model, X, y)
    per_call = clock() - start
    columns = [c for c in X.columns if c in set(settings.sources)]
    _check_time(per_call * len(X.columns) * PERMUTATION_REPEATS, deadline, clock, "permutation")
    result = permutation_importance(model, X, y, scoring=scorer, n_repeats=PERMUTATION_REPEATS,
                                    random_state=settings.seed, n_jobs=1)  # fmt: skip
    found = [Importance(str(c), float(result.importances_mean[j]),
                        float(result.importances_std[j]))
             for j, c in enumerate(X.columns) if c in columns]  # fmt: skip
    found.sort(key=lambda i: (-i.value, i.name))
    return found[:TOP_FEATURES]


def gain(model: Any, sources: Sequence[str]) -> list[Importance]:
    """Native gain (boosting) or impurity (forest) importance folded into source columns."""
    tab = inner_pipeline(model).steps[-1][1]
    if tab.family not in TREE_FAMILIES:
        return []
    return aggregate(raw_importances(tab, [str(c) for c in tab.feature_names_in_]), sources)


def _why(exc: Exception) -> str:
    """Exception class only: messages can contain data values (I-11)."""
    if isinstance(exc, ExplainTimeoutError | ShapUnsupportedError):
        return str(exc)
    return f"{type(exc).__name__}"


def explain(final_model: Any, data: ExplainData, settings: ExplainSettings) -> ExplanationResult:
    """§21.3 with the §21.5 fallbacks. Never raises for a failed method: the reasons are kept
    in `unavailable_reason` and the methods that succeeded are reported.
    """
    clock = settings.clock
    start = clock()
    deadline = start + max(0.0, settings.time_cap_s)
    ensemble = isinstance(final_model, VotingEnsemble)
    if ensemble:
        weights = np.asarray(final_model.weights, dtype=float)
        target = final_model.estimators[int(np.argmax(weights))]  # lowest index on ties
    else:
        target = final_model
    result = ExplanationResult(method="none", n_rows=0)
    reasons: list[str] = []
    try:
        result.gain = gain(target, settings.sources)
    except Exception as exc:
        reasons.append(f"gain importance failed: {_why(exc)}")
    shap_method: Method | None = None
    try:
        shap_method, result.shap, result.global_importance, notes = shap_values(
            target, data, settings, deadline)  # fmt: skip
        result.notes.extend(notes)
        result.n_rows = len(result.shap[0].values)
    except Exception as exc:
        reasons.append(f"SHAP unavailable: {_why(exc)}")
    if ensemble:
        result.notes.append("SHAP values are for the highest-weight ensemble member.")
    if shap_method is None or ensemble:
        try:
            result.permutation = permutation(final_model, data, settings, deadline)
        except Exception as exc:
            reasons.append(f"permutation importance unavailable: {_why(exc)}")
    if ensemble and shap_method and result.permutation:
        result.method = "shap_member_plus_permutation"
    elif shap_method:
        result.method = shap_method
    elif result.permutation:
        result.method = "permutation"
    elif result.gain:
        result.method = "gain"
    if reasons:
        result.unavailable_reason = "; ".join(reasons)[:REASON_CHARS]
    result.seconds = round(clock() - start, 3)
    return result
