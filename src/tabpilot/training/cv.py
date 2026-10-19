"""Cross-validation runner and OOF store (PRD §11.4, §16.3, §14.6, FR-TRN-01).

Every fold builds a fresh, unfitted pipeline, fits it on the training rows only, predicts the
validation rows, and discards the fitted pipeline after scoring (§29.6). All candidates share the
`SplitSpec` fold assignment (I-3).

Failure semantics:
- A DSL feature that fails (raises, is degenerate, or exceeds the time cap at fit or transform) in
  any fold is dropped for the candidate, together with features depending on it, and the candidate
  restarts from fold 1 — folds never mix feature sets (§14.6, T-FE-08).
- A failure on fold 1 with non-default parameters is retried once with the default parameters;
  any other failure marks the candidate `failed` with the exception class and a sanitized message
  (§16.3, FR-TRN-01). Failures never propagate to other candidates (NFR-REL-03).
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import Problem, score
from tabpilot.logging import get_logger, scrub
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import PipelineSchema, build_pipeline, inner_pipeline
from tabpilot.schemas import CandidateSpec, ExecutionProfileName, MetricName, SplitSpec, Task
from tabpilot.schemas.experiment import TrainingResult
from tabpilot.splitting import average_repeats, iter_folds

QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
MAX_ERROR_CHARS = 200

log = get_logger("train")


class BudgetExceeded(Exception):  # noqa: N818 - name used by PRD §11.4
    """Raised by a budget hook to stop the current candidate (§11.4, §29.3)."""


class Cancelled(Exception):  # noqa: N818 - control-flow signal, not an error
    """The run was cancelled; propagates to the orchestrator."""


class CancelToken(Protocol):
    """Cooperative cancellation flag (`threading.Event` satisfies it)."""

    def is_set(self) -> bool: ...


Builder = Callable[..., Any]


@dataclass(frozen=True)
class DevData:
    """The validated feature frame (positional rows), the encoded target, and the problem.

    Only rows listed in the split's folds are ever indexed, so holdout rows in `frame` are never
    read here (I-1).
    """

    frame: pd.DataFrame
    y: NDArray[Any]
    problem: Problem

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, target: str, problem: Problem) -> DevData:
        """Split off `target`; classification labels become codes in `problem.classes` order."""
        labels = frame[target]
        if problem.task == Task.REGRESSION:
            y: NDArray[Any] = labels.to_numpy(dtype=np.float64)
        else:
            codes = {label: i for i, label in enumerate(problem.classes)}
            y = labels.astype("string").map(codes).to_numpy(dtype=np.float64)
            if np.isnan(y).any():
                raise ValueError("target has labels outside problem.classes")
            y = y.astype(np.int64)
        return cls(frame.drop(columns=[target]), y, problem)


class OOFStore:
    """OOF predictions by entry id: in memory, and as float32 `.npy` files when `directory` is set
    (FR-TRN-01: on disk for the selected candidates).
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory
        self._memory: dict[str, NDArray[np.float32]] = {}

    def put(self, entry_id: str, oof: NDArray[np.float32]) -> str:
        self._memory[entry_id] = oof.astype(np.float32, copy=False)
        return f"memory://{entry_id}"

    def persist(self, entry_id: str) -> str:
        if self.directory is None:
            raise ValueError("OOFStore has no directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"oof_{entry_id}.npy"
        np.save(path, self._memory[entry_id])
        return str(path)

    def get(self, ref: str) -> NDArray[np.float32]:
        if ref.startswith("memory://"):
            return self._memory[ref.removeprefix("memory://")]
        return np.asarray(np.load(ref), dtype=np.float32)


@dataclass
class CVRun:
    """A candidate's CV outcome: the `TrainingResult` contract plus in-process details."""

    result: TrainingResult
    candidate: CandidateSpec  # as finally run (features dropped, fallback params applied)
    oof: NDArray[np.float32] | None = None
    best_iterations: list[int] = field(default_factory=list)
    dropped_features: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Attempt:
    fold_scores: list[float] = field(default_factory=list)
    oof: list[NDArray[np.float32]] = field(default_factory=list)
    fit_seconds: list[float] = field(default_factory=list)
    predict_seconds: list[float] = field(default_factory=list)
    best_iterations: list[int] = field(default_factory=list)
    failed_features: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    failed_fold: int | None = None
    warnings: list[str] = field(default_factory=list)


def entry_id_for(candidate: CandidateSpec) -> str:
    """`<iteration>-<model>-<8 hex>` (ENTRY_ID_PATTERN), stable for a given candidate."""
    digest = hashlib.sha256(candidate.model_dump_json().encode()).hexdigest()[:8]
    return f"{candidate.iteration}-{candidate.model.name.value}-{digest}"


def sanitize_error(exc: BaseException, fold: int) -> str:
    """Exception class + first message line with quoted fragments (possible cell values) removed
    and key patterns scrubbed (I-10, I-11).
    """
    first = (str(exc).splitlines() or [""])[0]
    message = scrub(QUOTED.sub("<…>", first)).strip()
    text = f"{type(exc).__name__} on fold {fold}" + (f": {message}" if message else "")
    return text[:MAX_ERROR_CHARS]


def _predict(model: Any, X: pd.DataFrame, problem: Problem) -> tuple[NDArray[np.float64], bool]:
    """(n, n_outputs) predictions; probability columns aligned to `problem.classes` (§10.3).
    The flag is True when a class was missing from the training fold.
    """
    if problem.task == Task.REGRESSION:
        return np.asarray(model.predict(X), dtype=np.float64).reshape(len(X), 1), False
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(model.classes_).astype(np.int64)
    full = np.zeros((len(X), problem.n_classes), dtype=np.float64)
    full[:, classes] = proba
    return full, len(classes) < problem.n_classes


class _NonFiniteOutputError(ValueError):
    """Predictions or the fold score are NaN/inf, or the prediction shape is wrong (§16.3)."""


def _fold_failures(model: Any, at_fit: bool) -> dict[str, str]:
    dsl = inner_pipeline(model).named_steps["dsl_features"]
    if at_fit:
        return dict(dsl.failed_features_)
    return dict.fromkeys(dsl.slow_features_, "failed: exceeded time cap at transform")


def _run_folds(
    candidate: CandidateSpec,
    *,
    dev: DevData,
    split: SplitSpec,
    schema: PipelineSchema,
    registry: ModelRegistry,
    metric: MetricName,
    cancel: CancelToken | None,
    budget_hook: Callable[[], None] | None,
    build: Builder,
    profile: ExecutionProfileName | None,
) -> _Attempt:
    problem = dev.problem
    n_out = 1 if problem.task == Task.REGRESSION else problem.n_classes
    dev_rows = np.asarray(split.dev_indices, dtype=np.int64)
    position = np.full(len(dev.frame), -1, dtype=np.int64)
    position[dev_rows] = np.arange(len(dev_rows))
    attempt = _Attempt(oof=[np.full((len(dev_rows), n_out), np.nan, dtype=np.float32)
                            for _ in range(split.n_repeats)])  # fmt: skip
    for index, (repeat, _fold, train, valid) in enumerate(iter_folds(split)):
        if cancel is not None and cancel.is_set():
            raise Cancelled
        try:
            model = build(candidate, schema, registry, profile=profile)
            start = time.perf_counter()
            model.fit(dev.frame.iloc[train], dev.y[train])
            attempt.fit_seconds.append(time.perf_counter() - start)
            attempt.failed_features = _fold_failures(model, at_fit=True)
            if attempt.failed_features:
                return attempt
            start = time.perf_counter()
            pred, class_missing = _predict(model, dev.frame.iloc[valid], problem)
            attempt.predict_seconds.append(time.perf_counter() - start)
            attempt.failed_features = _fold_failures(model, at_fit=False)
            if attempt.failed_features:
                return attempt
            if pred.shape != (len(valid), n_out) or not np.all(np.isfinite(pred)):
                raise _NonFiniteOutputError("non-finite or misshaped predictions")
            scored = pred[:, 0] if problem.task == Task.REGRESSION else pred
            value = score(metric, dev.y[valid], scored, problem)
            if not np.isfinite(value):
                raise _NonFiniteOutputError("non-finite fold score")
        except (Cancelled, BudgetExceeded):
            raise
        except Exception as exc:  # model failures are isolated (NFR-REL-03)
            attempt.error = sanitize_error(exc, index + 1)
            attempt.failed_fold = index
            return attempt
        if class_missing and "class_missing_in_fold" not in attempt.warnings:
            attempt.warnings.append("class_missing_in_fold")
        attempt.oof[repeat][position[valid]] = pred.astype(np.float32)
        attempt.fold_scores.append(float(value))
        best = inner_pipeline(model).named_steps["model"].best_iteration_
        if best is not None:
            attempt.best_iterations.append(int(best))
        del model  # fitted fold pipelines are discarded after scoring (§29.6)
        if budget_hook is not None:
            budget_hook()
    return attempt


def run_cv(
    candidate: CandidateSpec,
    dev: DevData,
    split: SplitSpec,
    *,
    schema: PipelineSchema,
    registry: ModelRegistry,
    metric: MetricName,
    cancel: CancelToken | None = None,
    budget_hook: Callable[[], None] | None = None,
    resource_guard: Callable[[CandidateSpec], str | None] | None = None,
    store: OOFStore | None = None,
    build: Builder = build_pipeline,
    profile: ExecutionProfileName | None = None,
) -> CVRun:
    """Train `candidate` on every fold of `split` and return fold scores and OOF (§11.4).

    `budget_hook` runs after each fold and may raise `BudgetExceeded` (→ `skipped_budget`);
    `resource_guard` may veto the candidate before fitting (→ `skipped_resource`, §16.3);
    `cancel` is checked before each fold (→ `Cancelled` propagates).
    """
    entry_id = entry_id_for(candidate)
    reason = resource_guard(candidate) if resource_guard is not None else None
    if reason:
        log.warning("candidate_skipped", entry_id=entry_id, status="skipped_resource")
        result = TrainingResult(entry_id=entry_id, status="skipped_resource", error=reason)
        return CVRun(result, candidate)
    current = candidate
    dropped: dict[str, str] = {}
    warnings: list[str] = []
    fallback_used = False
    while True:
        try:
            attempt = _run_folds(current, dev=dev, split=split, schema=schema, registry=registry,
                                 metric=metric, cancel=cancel, budget_hook=budget_hook,
                                 build=build, profile=profile)  # fmt: skip
        except BudgetExceeded:
            log.warning("candidate_skipped", entry_id=entry_id, status="skipped_budget")
            result = TrainingResult(entry_id=entry_id, status="skipped_budget")
            return CVRun(result, current, dropped_features=dropped, warnings=warnings)
        if attempt.failed_features:
            dropped.update(attempt.failed_features)
            kept = [f for f in current.features if f.name not in attempt.failed_features]
            log.warning("features_dropped", entry_id=entry_id, n=len(attempt.failed_features))
            if not kept:
                error = "all LLM features failed (§14.6)"
                return _failed(CVRun(_failure(entry_id, error), current, dropped_features=dropped,
                                     warnings=warnings))  # fmt: skip
            current = current.model_copy(update={"features": kept})
            continue
        retry = attempt.failed_fold == 0 and not fallback_used and bool(current.model.params)
        if attempt.error and retry:
            fallback_used = True
            warnings.append("fallback_params")
            log.warning("fallback_params", entry_id=entry_id)
            model = current.model.model_copy(update={"params": {}})
            current = current.model_copy(update={"model": model})
            continue
        if attempt.error:
            return _failed(CVRun(_failure(entry_id, attempt.error), current,
                                 dropped_features=dropped, warnings=warnings))  # fmt: skip
        run = CVRun(_success(entry_id, attempt, current, split, store), current,
                    dropped_features=dropped, warnings=warnings + attempt.warnings)  # fmt: skip
        run.best_iterations = attempt.best_iterations
        run.oof = _oof(attempt, split)
        log.info("candidate_completed", entry_id=entry_id, folds=len(attempt.fold_scores))
        return run


def _failure(entry_id: str, error: str) -> TrainingResult:
    return TrainingResult(entry_id=entry_id, status="failed", error=error)


def _failed(run: CVRun) -> CVRun:
    error = run.result.error or ""
    log.warning("candidate_failed", entry_id=run.result.entry_id, error_class=error.split(" ")[0])
    return run


def _oof(attempt: _Attempt, split: SplitSpec) -> NDArray[np.float32]:
    return average_repeats(attempt.oof) if split.n_repeats > 1 else attempt.oof[0]


def _success(
    entry_id: str,
    attempt: _Attempt,
    candidate: CandidateSpec,
    split: SplitSpec,
    store: OOFStore | None,
) -> TrainingResult:
    return TrainingResult(
        entry_id=entry_id,
        status="ok",
        fold_scores=attempt.fold_scores,
        fit_seconds_mean=float(np.mean(attempt.fit_seconds)),
        predict_seconds_mean=float(np.mean(attempt.predict_seconds)),
        oof_ref=store.put(entry_id, _oof(attempt, split)) if store is not None else None,
        best_params=dict(candidate.model.params),
    )
