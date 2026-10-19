"""P07 tests: T-FE-01…07, T-LEAK-07 (DSL stateful ops), pickling without `tabpilot`, clone."""

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
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from tests.helpers.fit_spy import FitSpy, fitted_rows

from tabpilot.features import (
    FeatureSchema,
    canonical_form,
    catalog_for_planner,
    provenance,
    runtime_specs,
    validate_spec,
    validate_specs,
)
from tabpilot.features.provenance import baseline_provenance
from tabpilot.ingestion import inference
from tabpilot.schemas import FEATURE_SPEC_ADAPTER, FeatureSpec, SemanticType
from tabpilot_runtime import (
    BaselineFeatureTransformer,
    DSLFeatureTransformer,
    FrequencyEncoder,
    RareCategoryGrouper,
)
from tabpilot_runtime.dsl_ops import FALSE_TOKENS, OPS, STATEFUL, TRUE_TOKENS

SRC = Path(__file__).resolve().parents[2] / "src"


def spec(
    name: str, op: str, inputs: list[str], params: dict[str, Any] | None = None, i: int = 1
) -> FeatureSpec:
    data: dict[str, Any] = {"id": f"f{i:02d}", "name": name, "op": op, "inputs": inputs}
    if params is not None:
        data["params"] = params
    return FEATURE_SPEC_ADAPTER.validate_python(data)


def frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(10, 3, n)
    b = rng.normal(5, 2, n)
    b[:3] = 0.0
    a[3:6] = np.nan
    return pd.DataFrame({
        "a": a, "b": b, "neg": rng.normal(0, 1, n),
        "cat": pd.Categorical(rng.choice(["x", "y", "z"], n)),
        "cat2": pd.Categorical(rng.choice(["p", "q"], n)),
        "d1": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
        "d2": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
        "flag": rng.integers(0, 2, n).astype(float),
    })  # fmt: skip


SCHEMA = FeatureSchema(
    columns={"a": SemanticType.NUMERIC, "b": SemanticType.NUMERIC, "neg": SemanticType.NUMERIC,
             "cat": SemanticType.CATEGORICAL, "cat2": SemanticType.CATEGORICAL,
             "d1": SemanticType.DATETIME, "d2": SemanticType.DATETIME, "flag": SemanticType.BOOLEAN},
    target="y", excluded=frozenset({"customer_id"}), minimums={"neg": -2.5, "a": 1.0},
    n_rows=200,
)  # fmt: skip


def fit_dsl(
    specs: list[FeatureSpec], df: pd.DataFrame, **kw: Any
) -> tuple[DSLFeatureTransformer, pd.DataFrame]:
    t = DSLFeatureTransformer(runtime_specs(specs), **kw).fit(df)
    return t, t.transform(df)


# ---------------------------------------------------------------- T-FE-01 baseline
@pytest.mark.unit
def test_T_FE_01_baseline_features() -> None:
    df = pd.DataFrame({
        "when": pd.to_datetime(["2024-01-06 00:00", "2024-01-07 13:30", None, "2024-03-01 08:00"] * 25),
        "x": [1.0, np.nan, 3.0, 4.0] * 25,
        "c": pd.Categorical((["common"] * 3 + ["rare_one"]) * 24 + ["common"] * 4),
    })  # fmt: skip
    df.loc[99, "c"] = "rare_one"
    t = BaselineFeatureTransformer(
        ["when"], ["x"], ["c"], rare_min_share=0.01, rare_min_count=30
    ).fit(df)
    out = t.transform(df)
    assert "when" not in out.columns
    expected = {f"when__{p}" for p in ("year", "month", "day", "dayofweek", "hour", "is_weekend",
                                       "days_since_min")}  # fmt: skip
    assert expected <= set(out.columns)
    assert out.loc[0, "when__is_weekend"] == 1.0  # 2024-01-06 is a Saturday
    assert out.loc[1, "when__hour"] == 13.0
    assert out.loc[0, "when__days_since_min"] == 0.0
    assert np.isnan(out.loc[2, "when__year"])
    assert out["x__missing"].tolist()[:4] == [0, 1, 0, 0]
    assert set(out["c"].dropna().unique()) == {"common", "__rare__"}  # rare_one has 25 < 30 rows
    assert list(t.get_feature_names_out()) == list(out.columns)
    date_only = BaselineFeatureTransformer(["d1"], [], []).fit(frame())
    assert "d1__hour" not in date_only.transform(frame()).columns


@pytest.mark.unit
def test_rare_grouper_and_frequency_encoder_learn_on_fit_only() -> None:
    train = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 45 + ["c"] * 5})
    test = pd.DataFrame({"c": ["a", "c", "new", None]})
    rare = RareCategoryGrouper(["c"], min_share=0.1, min_count=1).fit(train)
    assert rare.transform(test)["c"].tolist()[:3] == ["a", "__rare__", "__rare__"]
    assert pd.isna(rare.transform(test)["c"].tolist()[3])
    freq = FrequencyEncoder(["c"]).fit(train)
    assert freq.transform(test)["c"].tolist() == pytest.approx([0.5, 0.05, 0.0, 0.0])


# ---------------------------------------------------------------- T-FE-02 ops
@pytest.mark.unit
def test_T_FE_02_ops_against_hand_computed_values() -> None:
    df = frame()
    specs = [
        spec("s_add", "add", ["a", "b"], i=1), spec("s_sub", "subtract", ["a", "b"], i=2),
        spec("s_mul", "multiply", ["a", "b"], i=3), spec("s_ratio", "ratio", ["a", "b"], {"eps": 1e-3}, i=4),
        spec("s_sq", "square", ["neg"], i=5), spec("s_rowsum", "row_sum", ["a", "b", "neg"], i=6),
        spec("s_rowmean", "row_mean", ["a", "b"], i=7), spec("s_cm", "count_missing", ["a", "b"], i=8),
        spec("s_dd", "date_diff_days", ["d1", "d2"], i=9), spec("s_eq", "equals", ["flag"], {"value": 1}, i=10),
        spec("s_concat", "concat_categories", ["cat", "cat2"], i=11),
    ]  # fmt: skip
    _, out = fit_dsl(specs, df)
    a, b = df["a"].to_numpy(), df["b"].to_numpy()
    np.testing.assert_allclose(out["llm__s_add"], (a + b).astype(np.float32), equal_nan=True)
    np.testing.assert_allclose(out["llm__s_sub"], (a - b).astype(np.float32), equal_nan=True)
    np.testing.assert_allclose(
        out["llm__s_mul"], (a * b).astype(np.float32), equal_nan=True, rtol=1e-6
    )
    ratio = out["llm__s_ratio"].to_numpy()
    assert np.isnan(ratio[:3]).all()  # |den| < eps → NaN
    np.testing.assert_allclose(ratio[10:], (a[10:] / b[10:]).astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(
        out["llm__s_rowmean"], ((a + b) / 2).astype(np.float32), equal_nan=True
    )
    assert out["llm__s_cm"].tolist()[3] == 1.0
    np.testing.assert_allclose(out["llm__s_dd"], ((df["d1"] - df["d2"]).dt.days).astype(np.float32))
    assert out["llm__s_eq"].dtype == np.uint8
    assert out["llm__s_eq"].tolist() == (df["flag"] == 1).astype(np.uint8).tolist()
    assert out["llm__s_concat"].dtype == "category"
    assert out["llm__s_concat"].iloc[0] == f"{df['cat'].iloc[0]}|{df['cat2'].iloc[0]}"
    assert out["llm__s_add"].dtype == np.float32


@pytest.mark.unit
def test_T_FE_02_stateful_ops() -> None:
    df = frame()
    specs = [spec("clipped", "clip", ["a"], {"lower_q": 0.1, "upper_q": 0.9}, i=1),
             spec("binned", "bin_quantile", ["a"], {"n_bins": 4}, i=2),
             spec("gmean", "group_mean", ["cat", "a"], i=3), spec("gstd", "group_std", ["cat", "a"], i=4),
             spec("gcount", "group_count", ["cat"], i=5), spec("gdev", "group_deviation", ["cat", "b"], i=6)]  # fmt: skip
    _, out = fit_dsl(specs, df)
    lo, hi = np.nanquantile(df["a"], [0.1, 0.9])
    assert out["llm__clipped"].min() == pytest.approx(lo, rel=1e-6)
    assert out["llm__clipped"].max() == pytest.approx(hi, rel=1e-6)
    assert list(out["llm__binned"].cat.categories) == ["bin_0", "bin_1", "bin_2", "bin_3"]
    means = df.groupby("cat", observed=True)["a"].mean()
    assert out.loc[0, "llm__gmean"] == pytest.approx(means[df.loc[0, "cat"]], rel=1e-6)
    counts = df["cat"].value_counts()
    assert out.loc[0, "llm__gcount"] == counts[df.loc[0, "cat"]]
    b_means = df.groupby("cat", observed=True)["b"].mean()
    assert out.loc[0, "llm__gdev"] == pytest.approx(
        df.loc[0, "b"] - b_means[df.loc[0, "cat"]], rel=1e-5
    )
    assert set(STATEFUL) == {
        "clip",
        "bin_quantile",
        "group_mean",
        "group_std",
        "group_count",
        "group_deviation",
    }
    assert set(OPS) == {c.op for c in catalog_for_planner()}


# ---------------------------------------------------------------- T-FE-03 static validation
@pytest.mark.unit
@pytest.mark.parametrize(("s", "code"), [
    (spec("f", "add", ["a", "nope"]), "unknown_column"),
    (spec("f", "add", ["a", "customer_id"]), "excluded_column"),
    (spec("f", "add", ["a", "y"]), "target_column"),
    (spec("f", "add", ["a", "cat"]), "feature_type_mismatch"),
    (spec("f", "group_mean", ["a", "b"]), "feature_type_mismatch"),
    (spec("f", "date_diff_days", ["a", "d1"]), "feature_type_mismatch"),
    (spec("f", "add", ["a", "http://x"]), "unsafe_feature"),
    (spec("f", "add", ["a", "../etc"]), "unsafe_feature"),
    (spec("f", "add", ["a", "__import__"]), "unsafe_feature"),
    (spec("a", "square", ["b"]), "duplicate_feature"),
])  # fmt: skip
def test_T_FE_03_static_validation_rejects(s: FeatureSpec, code: str) -> None:
    result = validate_spec(s, SCHEMA)
    assert not result.ok
    assert [i.code for i in result.issues] == [code]


@pytest.mark.unit
def test_T_FE_03_composition_depth_count_dedupe_cost() -> None:
    chain = [spec("f1", "add", ["a", "b"], i=1), spec("f2", "square", ["f1"], i=2),
             spec("f3", "log1p", ["f2"], i=3)]  # fmt: skip
    accepted, issues = validate_specs(chain, SCHEMA)
    assert [s.name for s in accepted] == ["f1", "f2"]
    assert [i.code for i in issues] == ["depth_exceeded"]
    commuted = [spec("p", "add", ["a", "b"], i=1), spec("q", "add", ["b", "a"], i=2),
                spec("r", "subtract", ["a", "b"], i=3), spec("s", "subtract", ["b", "a"], i=4)]  # fmt: skip
    accepted, issues = validate_specs(commuted, SCHEMA)
    assert [s.name for s in accepted] == ["p", "r", "s"]  # subtract is not commutative
    assert canonical_form(commuted[0]) == canonical_form(commuted[1])
    many = [spec(f"m{i}", "add", ["a", "b"], i=i) for i in range(1, 9)]
    many = [spec(f"m{i}", "ratio", ["a", "b"], {"eps": 1e-3 * i}, i=i) for i in range(1, 9)]
    accepted, issues = validate_specs(many, SCHEMA, max_count=6)
    assert len(accepted) == 6
    assert [i.code for i in issues] == ["too_many_features"] * 2
    tight = FeatureSchema(columns=SCHEMA.columns, target="y", n_rows=1_000_000, memory_budget_mb=5)
    accepted, issues = validate_specs(many[:3], tight)
    assert len(accepted) == 1
    assert {i.code for i in issues} == {"cost_exceeded"}


@pytest.mark.unit
def test_dependents_of_rejected_specs_are_rejected() -> None:
    specs = [spec("bad", "add", ["a", "nope"], i=1), spec("child", "square", ["bad"], i=2)]
    accepted, issues = validate_specs(specs, SCHEMA)
    assert accepted == []
    assert [i.code for i in issues] == ["unknown_column", "unknown_column"]


# ---------------------------------------------------------------- T-FE-04 NaN semantics
@pytest.mark.unit
def test_T_FE_04_nan_semantics_table() -> None:
    df = pd.DataFrame({"x": [-1.0, 0.0, 4.0, np.nan, 9.0, 2.0], "z": [1.0, 1e-9, 2.0, 1.0, np.nan, 3.0],
                       "k": pd.Categorical(["u", "u", "v", None, "v", "u"])})  # fmt: skip
    test = pd.DataFrame({"x": [1.0, 2.0], "z": [1.0, 1.0], "k": pd.Categorical(["w", None])})
    schema = FeatureSchema(columns={"x": SemanticType.NUMERIC, "z": SemanticType.NUMERIC,
                                    "k": SemanticType.CATEGORICAL}, target="y",
                           minimums={"x": -1.0})  # fmt: skip
    specs = [spec("lg", "log1p", ["x"], i=1), spec("sq", "sqrt", ["x"], i=2),
             spec("rt", "ratio", ["x", "z"], {"eps": 1e-6}, i=3), spec("ad", "add", ["x", "z"], i=4),
             spec("gm", "group_mean", ["k", "x"], i=5), spec("eq", "equals", ["x"], {"value": 4}, i=6)]  # fmt: skip
    accepted, issues = validate_specs(specs, schema)
    assert len(accepted) == 6
    assert {(i.code, i.fatal) for i in issues} == {
        ("negative_inputs", False)
    }  # A-007: flagged only
    t, out = fit_dsl(accepted, df)
    assert np.isnan(out["llm__lg"].iloc[0])  # negative → NaN (no clipping)
    assert t.negative_inputs_ == {"lg": 1, "sq": 1}  # A-007 counter
    assert out["llm__lg"].iloc[2] == pytest.approx(np.log1p(4.0))
    assert np.isnan(out["llm__sq"].iloc[0])
    assert np.isnan(out["llm__rt"].iloc[1])  # |den| < eps
    assert np.isnan(out["llm__ad"].iloc[3])  # NaN propagates
    assert out["llm__eq"].iloc[3] == 0  # equals on missing → 0
    unseen = t.transform(test)
    assert np.isnan(unseen["llm__gm"].iloc[0])  # unseen key → NaN
    assert (
        unseen["llm__gm"].iloc[1] == pytest.approx(np.nan, nan_ok=True)
        or unseen["llm__gm"].iloc[1] >= 0
    )


@pytest.mark.unit
def test_boolean_tokens_map_to_numbers() -> None:
    """Boolean columns keep their original tokens (P04); DSL ops read them as 1/0."""
    df = pd.DataFrame({"flag": pd.Categorical(["Yes", "no", "YES", None, "No", "yes"]),
                       "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})  # fmt: skip
    specs = [spec("is_yes", "equals", ["flag"], {"value": 1}, i=1),
             spec("sx", "row_sum", ["flag", "x"], i=2)]  # fmt: skip
    _, out = fit_dsl(specs, df)
    assert out["llm__is_yes"].tolist() == [1, 0, 1, 0, 0, 1]
    assert out["llm__sx"].iloc[0] == 2.0
    assert np.isnan(out["llm__sx"].iloc[3])


@pytest.mark.unit
def test_runtime_boolean_tokens_match_ingestion() -> None:
    assert TRUE_TOKENS == inference.TRUE_TOKENS
    assert TRUE_TOKENS | FALSE_TOKENS == inference.BOOL_TOKENS
    assert not TRUE_TOKENS & FALSE_TOKENS


# ---------------------------------------------------------------- T-FE-05/06/07
@pytest.mark.unit
def test_T_FE_05_degenerate_features_dropped() -> None:
    df = frame()
    df["const"] = 1.0
    df["mostly_nan"] = np.where(np.arange(len(df)) == 0, 1.0, np.nan)
    specs = [spec("c1", "square", ["const"], i=1), spec("c2", "square", ["mostly_nan"], i=2),
             spec("ok", "square", ["a"], i=3)]  # fmt: skip
    t, out = fit_dsl(specs, df)
    assert t.failed_features_ == {"c1": "degenerate", "c2": "degenerate"}
    assert [c for c in out.columns if c.startswith("llm__")] == ["llm__ok"]


@pytest.mark.unit
def test_T_FE_06_dependency_skipping() -> None:
    df = frame()
    df["const"] = 2.0
    specs = [spec("base", "square", ["const"], i=1), spec("child", "add", ["base", "a"], i=2),
             spec("free", "add", ["a", "b"], i=3)]  # fmt: skip
    t, out = fit_dsl(specs, df)
    assert t.failed_features_["child"] == "depends on failed feature base"
    assert "llm__free" in out.columns
    records = provenance(specs, [], t, "llm_plan", 0)
    assert {r.name: r.status for r in records} == {
        "base": "degenerate",
        "child": "failed",
        "free": "active",
    }


@pytest.mark.unit
def test_T_FE_07_per_feature_time_cap() -> None:
    t, out = fit_dsl([spec("ok", "square", ["a"])], frame(), seconds_per_100k=0.0, min_cap_s=0.0)
    assert t.failed_features_["ok"].startswith("failed: exceeded")
    assert not [c for c in out.columns if c.startswith("llm__")]
    fast = DSLFeatureTransformer(runtime_specs([spec("ok", "square", ["a"])]), seconds_per_100k=0.0,
                                 min_cap_s=0.0)  # fmt: skip
    fast.active_, fast.resolved_, fast.state_ = (
        [runtime_specs([spec("ok", "square", ["a"])])[0]],
        {"ok": "llm__ok"},
        {"ok": {}},
    )
    fast.transform(frame())
    assert fast.slow_features_ == ["ok"]  # transform-time overrun is recorded for the CV runner


# ---------------------------------------------------------------- naming, failures, provenance
@pytest.mark.unit
def test_names_prefixed_and_collisions_suffixed() -> None:
    df = frame()
    df["llm__dup"] = 0.0
    t, out = fit_dsl([spec("dup", "square", ["a"])], df)
    assert "llm__dup__2" in out.columns
    assert t.resolved_ == {"dup": "llm__dup__2"}


@pytest.mark.unit
def test_runtime_failure_is_recorded_not_raised() -> None:
    bad = [{"id": "f1", "name": "boom", "op": "add", "inputs": ["a", "missing_col"], "params": {}}]
    t = DSLFeatureTransformer(bad).fit(frame())
    assert t.failed_features_ == {"boom": "failed: KeyError"}


@pytest.mark.unit
def test_provenance_records_rejected_and_baseline() -> None:
    specs = [spec("bad", "add", ["a", "nope"], i=1), spec("good", "add", ["a", "b"], i=2)]
    accepted, issues = validate_specs(specs, SCHEMA)
    records = provenance(accepted, issues, None, "llm_reflection", 1)
    assert {r.name: r.status for r in records} == {"good": "active", "bad": "rejected_static"}
    assert baseline_provenance(["x__missing"])[0].source == "baseline"


# ---------------------------------------------------------------- T-LEAK-07 (DSL part)
@pytest.mark.unit
@pytest.mark.parametrize("s", [
    spec("c", "clip", ["a"], {"lower_q": 0.1, "upper_q": 0.9}), spec("bq", "bin_quantile", ["a"], {"n_bins": 5}),
    spec("gm", "group_mean", ["cat", "a"]), spec("gs", "group_std", ["cat", "a"]),
    spec("gc", "group_count", ["cat"]), spec("gd", "group_deviation", ["cat", "a"]),
])  # fmt: skip
def test_T_LEAK_07_stateful_dsl_ops_fit_on_training_fold_only(s: FeatureSpec) -> None:
    df = frame(400, seed=7)
    folds = list(KFold(4, shuffle=True, random_state=0).split(df))
    pipe = Pipeline([("dsl", FitSpy(DSLFeatureTransformer(runtime_specs([s])), "dsl"))])
    for train, valid in folds:
        fitted = clone(pipe).fit(df.iloc[train])
        state = fitted.named_steps["dsl"].inner_.state_[s.name]
        reference = DSLFeatureTransformer(runtime_specs([s])).fit(df.iloc[train]).state_[s.name]
        full = DSLFeatureTransformer(runtime_specs([s])).fit(df).state_[s.name]
        assert state == reference
        assert state != full  # state depends on the rows it saw
        fitted.transform(df.iloc[valid])  # transform never refits
        assert fitted.named_steps["dsl"].inner_.state_[s.name] == reference
    for rows, (train, valid) in zip(fitted_rows("dsl"), folds, strict=True):
        assert rows == set(df.index[train])
        assert rows.isdisjoint(df.index[valid])


# ---------------------------------------------------------------- sklearn and pickling
@pytest.mark.unit
def test_clone_equivalence() -> None:
    df = frame()
    t = DSLFeatureTransformer(runtime_specs([spec("r", "ratio", ["a", "b"], {"eps": 1e-3})]))
    pd.testing.assert_frame_equal(clone(t).fit(df).transform(df), t.fit(df).transform(df))
    assert clone(t).get_params() == t.get_params()
    b = BaselineFeatureTransformer(["d1"], ["a"], ["cat"])
    assert clone(b).get_params() == b.get_params()


@pytest.mark.unit
def test_pickle_loads_without_tabpilot(tmp_path: Path) -> None:
    """A-001: fitted runtime objects unpickle with only `tabpilot_runtime` importable."""
    df = frame()
    specs = runtime_specs(
        [
            spec("gm", "group_mean", ["cat", "a"]),
            spec("bq", "bin_quantile", ["b"], {"n_bins": 3}, i=2),
        ]
    )
    objects = (DSLFeatureTransformer(specs).fit(df), BaselineFeatureTransformer(["d1"], ["a"], ["cat"]).fit(df),
               FrequencyEncoder(["cat"]).fit(df))  # fmt: skip
    (tmp_path / "objs.pkl").write_bytes(pickle.dumps(objects))
    df.to_pickle(tmp_path / "frame.pkl")
    expected = [o.transform(df) for o in objects]
    pd.to_pickle(expected, tmp_path / "expected.pkl")
    shutil.copytree(SRC / "tabpilot_runtime", tmp_path / "tabpilot_runtime")
    code = (
        "import importlib.util, sys, pickle, pandas as pd\n"
        "if importlib.util.find_spec('tabpilot') is not None:\n"
        "    raise SystemExit('tabpilot importable: ' + str(sys.path))\n"
        "objs = pickle.load(open('objs.pkl', 'rb'))\n"
        "df = pd.read_pickle('frame.pkl')\n"
        "exp = pd.read_pickle('expected.pkl')\n"
        "for o, e in zip(objs, exp):\n    pd.testing.assert_frame_equal(o.transform(df), e)\n"
        "print('ok', type(objs[0]).__module__)\n"
    )  # fmt: skip
    # -S skips site processing, so the editable install's .pth (which adds `src/`) is not read;
    # site-packages is added explicitly for pandas/numpy/sklearn.
    site_packages = sysconfig.get_paths()["purelib"]
    env = {"PYTHONPATH": os.pathsep.join([str(tmp_path), site_packages]), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path, env=env, capture_output=True,
                         text=True, check=False)  # fmt: skip
    assert out.returncode == 0, out.stderr + out.stdout
    assert out.stdout.strip() == "ok tabpilot_runtime.transformers"


@pytest.mark.unit
def test_catalog_for_planner_lists_every_op_with_bounds() -> None:
    entries = {e.op: e for e in catalog_for_planner()}
    assert entries["ratio"].params == {"eps": [1e-9, 1.0]}
    assert entries["clip"].params == {"lower_q": [0.0, 0.2], "upper_q": [0.8, 1.0]}
    assert entries["group_mean"].inputs == ["boolean|categorical", "numeric"]
    assert entries["row_sum"].inputs == ["numeric (2-20 inputs)"]


@pytest.mark.unit
def test_clip_bounds_come_from_fit_not_transform() -> None:
    """L-4 (KI-018): `clip` learns its quantiles on the training rows; transform reuses them."""
    train = frame()
    t = DSLFeatureTransformer(runtime_specs([spec("c", "clip", ["a"],
                                                  {"lower_q": 0.1, "upper_q": 0.9})])).fit(train)  # fmt: skip
    lo, hi = np.nanquantile(train["a"], [0.1, 0.9])
    shifted = train.assign(a=train["a"] * 10 + 1000)  # new rows far outside the training range
    out = t.transform(shifted)["llm__c"]
    assert out.min() == pytest.approx(hi, rel=1e-6)  # everything clipped to the fitted upper bound
    assert out.max() == pytest.approx(hi, rel=1e-6)
    assert lo < hi
