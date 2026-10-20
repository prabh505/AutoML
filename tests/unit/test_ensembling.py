"""P12 ensembling tests: T-ENS-01…04, eligibility, cost limits, `VotingEnsemble` (PRD §20)."""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.helpers.synthetic import synthetic

from tabpilot.config import load_settings
from tabpilot.ensembling import (
    Candidate,
    blend,
    build_ensemble,
    cross_fitted_ensemble,
    eligible,
    fold_partition,
    hill_climb,
    oof_fold_scores,
)
from tabpilot.evaluation.metrics import Problem, score
from tabpilot.evaluation.metrics import spec as metric_spec
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import build_pipeline
from tabpilot.schemas import CandidateSpec, MetricName, ModelName, Task
from tabpilot.schemas.experiment import LeaderboardEntry, ModelConfig, fold_statistics
from tabpilot.splitting import build_split
from tabpilot_runtime import VotingEnsemble

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
SRC = Path(__file__).resolve().parents[2] / "src"
BIN = Problem(Task.BINARY, ("0", "1"), 1)
LIMIT = 10**9


def data(n: int = 2500, seed: int = 0) -> tuple[np.ndarray, Any, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    split = build_split(pd.Series(y), Task.BINARY, seed_root=seed).spec
    return y, split, y[np.asarray(split.dev_indices)]


def noisy_model(y: np.ndarray, noise: float, seed: int) -> np.ndarray:
    """Probabilities: the signal plus independent noise (independent errors blend well)."""
    rng = np.random.default_rng(seed)
    logit = (2 * y - 1) * 1.0 + rng.normal(0, noise, len(y))
    p = 1 / (1 + np.exp(-logit))
    return np.column_stack([1 - p, p]).astype(np.float32)


def cand(eid: str, oof: np.ndarray | None, y_dev: np.ndarray, split: Any, *, model: str = "lightgbm",
         iteration: int = 0, status: str = "ok", predict: float = 0.01, size: int = 1000) -> Candidate:  # fmt: skip
    scores = oof_fold_scores(oof, y_dev, split, BIN, MetricName.ROC_AUC) if oof is not None else []
    stats: dict[str, Any] = (
        dict(zip(("mean", "std", "sem"), fold_statistics(scores), strict=True)) if scores else {}
    )
    entry = LeaderboardEntry(entry_id=eid, experiment_id=RID, iteration=iteration, model=model,
                             feature_set="baseline", metric=metric_spec(MetricName.ROC_AUC),
                             fold_scores=scores, status=status if scores else "failed",
                             comparable=True, split_id=split.split_id, predict_seconds=predict,
                             model_size_bytes=size, **stats)  # fmt: skip
    return Candidate(entry, oof)


# ---------------------------------------------------------------- T-ENS-01 cross-fitting
@pytest.mark.unit
def test_T_ENS_01_fold_weights_never_see_their_fold() -> None:
    _y, split, y_dev = data()
    preds = [noisy_model(y_dev, 2.5, s) for s in range(3)]
    fold_scores, final = cross_fitted_ensemble(preds, y_dev, split, BIN, MetricName.ROC_AUC)
    parts = fold_partition(split)
    assert len(fold_scores) == len(parts) == split.k * split.n_repeats
    for (tr, va), got in zip(parts, fold_scores, strict=True):
        w = hill_climb([p[tr] for p in preds], y_dev[tr], BIN, MetricName.ROC_AUC)
        assert got == pytest.approx(
            score(MetricName.ROC_AUC, y_dev[va], blend([p[va] for p in preds], w, BIN), BIN)
        )
        assert set(tr).isdisjoint(va)
    assert final.sum() == pytest.approx(1.0)
    assert np.array_equal(final, hill_climb(preds, y_dev, BIN, MetricName.ROC_AUC))


# ---------------------------------------------------------------- T-ENS-02 keep / discard
@pytest.mark.unit
def test_T_ENS_02_complementary_models_kept() -> None:
    _y, split, y_dev = data()
    cands = [cand(f"0-{m}-{i:08x}", noisy_model(y_dev, 2.0, i), y_dev, split, model=m)
             for i, m in enumerate(["lightgbm", "random_forest", "linear"])]  # fmt: skip
    result = build_ensemble(
        cands, y_dev, split, BIN, MetricName.ROC_AUC, artifact_limit_bytes=LIMIT
    )
    assert result.kept
    assert len(result.members) >= 2
    assert result.decision is not None
    assert result.decision.accepted
    assert sum(m.weight for m in result.members) == pytest.approx(1.0)
    assert result.decision.parent_entry_id == eligible(cands, MetricName.ROC_AUC)[0].entry.entry_id


@pytest.mark.unit
def test_T_ENS_02_no_gain_or_single_member_is_discarded() -> None:
    _y, split, y_dev = data()
    strong = noisy_model(y_dev, 0.5, 1)
    weak = noisy_model(y_dev, 6.0, 2)
    result = build_ensemble([cand("0-lightgbm-00000001", strong, y_dev, split),
                             cand("0-linear-00000002", weak, y_dev, split, model="linear")],
                            y_dev, split, BIN, MetricName.ROC_AUC, artifact_limit_bytes=LIMIT)  # fmt: skip
    assert not result.kept
    one = build_ensemble([cand("0-lightgbm-00000001", strong, y_dev, split)], y_dev, split, BIN,
                         MetricName.ROC_AUC, artifact_limit_bytes=LIMIT)  # fmt: skip
    assert not one.kept
    assert one.reason == "fewer than 2 eligible candidates"


@pytest.mark.unit
def test_cost_limits_reject_slow_or_large_ensembles() -> None:
    _y, split, y_dev = data()
    make = lambda predict, size: [cand(f"0-{m}-{i:08x}", noisy_model(y_dev, 2.0, i), y_dev, split,  # noqa: E731
                                      model=m, predict=predict if i else 0.01, size=size)
                                  for i, m in enumerate(["lightgbm", "random_forest", "linear"])]  # fmt: skip
    slow = build_ensemble(
        make(1.0, 10), y_dev, split, BIN, MetricName.ROC_AUC, artifact_limit_bytes=LIMIT
    )
    assert not slow.kept
    assert "limit 5×" in slow.reason
    assert (slow.predicted_inference_ratio or 0) > 5
    big = build_ensemble(
        make(0.01, 10**6), y_dev, split, BIN, MetricName.ROC_AUC, artifact_limit_bytes=10**6
    )
    assert not big.kept
    assert big.reason == "members exceed the artifact size limit"


# ---------------------------------------------------------------- T-ENS-03 determinism
@pytest.mark.unit
def test_T_ENS_03_deterministic_with_entry_id_tie_breaks() -> None:
    _y, split, y_dev = data()
    cands = [cand(f"0-{m}-{i:08x}", noisy_model(y_dev, 2.0, i), y_dev, split, model=m)
             for i, m in enumerate(["lightgbm", "random_forest", "linear"])]  # fmt: skip
    first = build_ensemble(cands, y_dev, split, BIN, MetricName.ROC_AUC, artifact_limit_bytes=LIMIT)
    second = build_ensemble(list(reversed(cands)), y_dev, split, BIN, MetricName.ROC_AUC,
                            artifact_limit_bytes=LIMIT)  # fmt: skip
    assert first == second
    twin = noisy_model(y_dev, 2.0, 7)
    weights = hill_climb([twin, twin.copy()], y_dev, BIN, MetricName.ROC_AUC)
    assert weights.tolist() == [1.0, 0.0]  # identical members: the lower index wins
    partner = noisy_model(y_dev, 2.0, 8)  # a later step ties between two copies of the partner
    weights = hill_climb([twin, partner, partner.copy()], y_dev, BIN, MetricName.ROC_AUC)
    assert weights[1] > 0
    assert weights[2] == 0


@pytest.mark.unit
def test_hill_climb_member_cap_and_regression_blend() -> None:
    _y, _split, y_dev = data()
    preds = [noisy_model(y_dev, 3.0, s) for s in range(8)]
    w = hill_climb(preds, y_dev, BIN, MetricName.ROC_AUC, max_members=2, max_steps=50)
    assert np.count_nonzero(w) <= 2
    reg = Problem(Task.REGRESSION)
    out = blend([np.array([[1.0], [3.0]]), np.array([[3.0], [5.0]])], np.array([0.25, 0.75]), reg)
    assert out[:, 0].tolist() == [2.5, 4.5]
    rows = blend([np.array([[0.2, 0.2]]), np.array([[0.0, 0.0]])], np.array([0.5, 0.5]), BIN)
    assert rows.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_eligibility_rules() -> None:
    _y, split, y_dev = data()

    def c(eid: str, noise: float, **kw: Any) -> Candidate:  # one noise draw: more noise ranks lower
        return cand(eid, noisy_model(y_dev, noise, 0), y_dev, split, **kw)

    cands = [c("0-lightgbm-00000001", 1.0), c("1-lightgbm-00000002", 1.1, iteration=1),
             c("1-lightgbm-00000003", 1.2, iteration=1),  # second variant of lightgbm iteration 1
             c("0-linear-00000004", 1.5, model="linear"), c("1-linear-00000005", 1.6, model="linear", iteration=1),
             c("2-linear-00000006", 1.7, model="linear", iteration=2),  # third linear: over the cap
             cand("0-xgboost-00000007", None, y_dev, split, model="xgboost")]  # fmt: skip
    pool = [x.entry.entry_id for x in eligible(cands, MetricName.ROC_AUC)]
    assert pool == [
        "0-lightgbm-00000001",
        "1-lightgbm-00000002",
        "0-linear-00000004",
        "1-linear-00000005",
    ]
    many = [c(f"0-{m}-{i:08x}", 1.0 + i / 10, model=m) for i, m in
            enumerate(["linear", "lightgbm", "xgboost", "catboost", "random_forest", "hist_gb"] * 2)]  # fmt: skip
    assert len(eligible(many, MetricName.ROC_AUC)) <= 8


# ---------------------------------------------------------------- T-ENS-04 VotingEnsemble
def fitted(model: ModelName, frame: pd.DataFrame, y: np.ndarray, schema: Any) -> Any:
    spec = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=model, task=Task.BINARY, seed=1, n_jobs=1))  # fmt: skip
    return build_pipeline(spec, schema, ModelRegistry(load_settings({}))).fit(frame, y)


@pytest.mark.unit
def test_T_ENS_04_voting_ensemble_matches_the_blend(tmp_path: Path) -> None:
    frame, y, schema = synthetic()
    members = [
        fitted(ModelName.LIGHTGBM, frame, y, schema),
        fitted(ModelName.LINEAR, frame, y, schema),
    ]
    weights = [0.7, 0.3]
    ens = VotingEnsemble(members, weights).fit(frame, y)
    expected = blend([m.predict_proba(frame) for m in members], np.asarray(weights), BIN)
    np.testing.assert_allclose(ens.predict_proba(frame), expected, atol=1e-12)
    assert ens.predict(frame).tolist() == ens.classes_[expected.argmax(axis=1)].tolist()
    lazy = VotingEnsemble(members, weights)  # prefit members: usable without fit
    np.testing.assert_allclose(lazy.predict_proba(frame), expected, atol=1e-12)
    (tmp_path / "ens.pkl").write_bytes(pickle.dumps(ens))
    frame.to_pickle(tmp_path / "frame.pkl")
    np.save(tmp_path / "expected.npy", expected)
    shutil.copytree(SRC / "tabpilot_runtime", tmp_path / "tabpilot_runtime")
    code = ("import importlib.util, pickle, numpy as np, pandas as pd\n"
            "assert importlib.util.find_spec('tabpilot') is None\n"
            "ens = pickle.load(open('ens.pkl', 'rb'))\n"
            "np.testing.assert_allclose(ens.predict_proba(pd.read_pickle('frame.pkl')), "
            "np.load('expected.npy'), atol=1e-12)\nprint('ok', type(ens).__module__)\n")  # fmt: skip
    env = {
        "PYTHONPATH": os.pathsep.join([str(tmp_path), sysconfig.get_paths()["purelib"]]),
        "PATH": "/usr/bin:/bin",
    }
    out = subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path, env=env, capture_output=True,
                         text=True, check=False)  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok tabpilot_runtime.ensemble"


class Fixed:
    """A fitted stand-in classifier with a given class order and constant probabilities."""

    def __init__(self, classes: list[str], proba: list[float]) -> None:
        self.classes_ = np.asarray(classes)
        self._proba = np.asarray(proba)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.tile(self._proba, (len(X), 1))

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), float(self._proba[0]))


@pytest.mark.unit
def test_voting_ensemble_aligns_class_order_by_label() -> None:
    a = Fixed(["no", "yes"], [0.8, 0.2])
    b = Fixed(["yes", "no"], [0.6, 0.4])  # reversed order
    ens = VotingEnsemble([a, b], [1, 1]).fit(None, None)
    assert ens.classes_.tolist() == ["no", "yes"]
    np.testing.assert_allclose(ens.predict_proba(np.zeros((2, 1))), [[0.6, 0.4]] * 2)
    three = VotingEnsemble([a, Fixed(["a", "no", "yes"], [1.0, 0.0, 0.0])], [1, 1]).fit(None, None)
    proba = three.predict_proba(np.zeros((1, 1)))
    assert three.classes_.tolist() == ["a", "no", "yes"]
    assert proba.sum() == pytest.approx(1.0)
    reg = VotingEnsemble([Fixed(["x"], [2.0]), Fixed(["x"], [4.0])], [3, 1], task="regression").fit(
        None, None
    )
    assert reg.predict(np.zeros((1, 1))).tolist() == [2.5]
    assert not hasattr(reg, "predict_proba")


@pytest.mark.unit
@pytest.mark.parametrize(("weights", "task"), [([1], "classification"), ([-1, 2], "classification"),
                                               ([0, 0], "classification"), ([1, 1], "ranking")])  # fmt: skip
def test_voting_ensemble_rejects_bad_configuration(weights: list[float], task: str) -> None:
    with pytest.raises(ValueError, match=r"weight|unknown task"):
        VotingEnsemble([Fixed(["a", "b"], [0.5, 0.5])] * 2, weights, task=task).fit(None, None)


@pytest.mark.unit
def test_voting_ensemble_refits_clones_when_not_prefit() -> None:
    frame, y, schema = synthetic()
    member = fitted(ModelName.LINEAR, frame, y, schema)
    ens = VotingEnsemble([member], [1.0], prefit=False).fit(frame, y)
    assert ens.estimators_[0] is not member
    with pytest.raises(ValueError, match="not fitted"):
        VotingEnsemble([member], [1.0], prefit=False).predict(frame)
