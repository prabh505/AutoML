"""P03 fixture tests: manifest completeness, deterministic regeneration, and ground truth.

These check that the fixtures really have the properties later tests rely on, using the
PRD's own methods where one is defined (e.g. the §11.7 single-feature depth-3 tree scan).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from tabpilot.schemas import PlannerOutput

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures"
MANIFEST: dict[str, Any] = yaml.safe_load((FIX / "MANIFEST.yaml").read_text(encoding="utf-8"))


def load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_fixtures", FIX / "make_fixtures.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / "datasets" / name)


@pytest.mark.unit
def test_manifest_lists_every_file_and_every_file_is_listed() -> None:
    on_disk = {
        p.relative_to(FIX).as_posix()
        for p in FIX.rglob("*")
        if p.is_file()
        and p.name not in {"make_fixtures.py", "MANIFEST.yaml"}
        and "__pycache__" not in p.parts
    }
    assert set(MANIFEST["files"]) == on_disk


@pytest.mark.unit
def test_every_prd_fixture_exists() -> None:
    """PRD §31.2: every listed dataset is committed (hosted_limit has a generator)."""
    for name in ("binary_basic", "binary_small", "binary_imbalanced", "multiclass_basic",
                 "multiclass_rare", "regression_basic", "regression_skewed", "leaky_binary",
                 "grouped", "quality_issues", "wide"):  # fmt: skip
        assert f"datasets/{name}.csv" in MANIFEST["files"]
    assert "datasets/hosted_limit.csv" in MANIFEST["not_committed"]
    for folder in ("parsing", "malicious", "plans", "llm_responses"):
        assert any(k.startswith(f"{folder}/") for k in MANIFEST["files"])


@pytest.mark.unit
@pytest.mark.parametrize(("name", "shape"), [
    ("binary_basic.csv", (2500, 12)), ("binary_small.csv", (180, 6)),
    ("binary_imbalanced.csv", (5000, 10)), ("multiclass_basic.csv", (3000, 10)),
    ("multiclass_rare.csv", (600, 6)), ("regression_basic.csv", (3000, 10)),
    ("regression_skewed.csv", (2500, 8)), ("leaky_binary.csv", (2500, 8)),
    ("grouped.csv", (3000, 8)), ("quality_issues.csv", (1000, 15)), ("wide.csv", (500, 190)),
])  # fmt: skip
def test_shapes_match_prd(name: str, shape: tuple[int, int]) -> None:
    assert dataset(name).shape == shape
    assert MANIFEST["files"][f"datasets/{name}"]["shape"] == list(shape)


@pytest.mark.unit
def test_regeneration_is_byte_identical() -> None:
    result = subprocess.run([sys.executable, "fixtures/make_fixtures.py", "--check"], cwd=ROOT,
                            capture_output=True, text=True, check=False)  # fmt: skip
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_hosted_limit_generator(tmp_path: Path) -> None:
    gen = load_generator()
    path = gen.generate_hosted_limit(tmp_path / "hl.csv")
    head = pd.read_csv(path, nrows=5)
    assert head.shape[1] == 30
    assert sum(1 for _ in path.open(encoding="utf-8")) == 100_001


# ---------------------------------------------------------------- ground truth
def univariate_auc(df: pd.DataFrame, col: str, target: str, seed: int) -> float:
    """PRD §11.7 scan: depth-3 tree on one feature, 3-fold CV ROC-AUC."""
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    return float(cross_val_score(tree, df[[col]], df[target], cv=cv, scoring="roc_auc").mean())


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(3))
def test_leaky_binary_levels_under_a002(seed: int) -> None:
    df = dataset("leaky_binary.csv")
    assert univariate_auc(df, "status_code", "defaulted", seed) >= 0.95  # near-deterministic
    assert 0.90 <= univariate_auc(df, "risk_score", "defaulted", seed) < 0.95  # strong only
    assert univariate_auc(df, "w2", "defaulted", seed) < 0.90  # ordinary column
    assert (df["status_code"] != df["defaulted"]).sum() == 12


@pytest.mark.unit
def test_grouped_leakage_is_detectable() -> None:
    """Memorizing patients via fingerprints inflates random CV versus group CV."""
    df = dataset("grouped.csv")
    X, y = df[["trait_a", "trait_b"]], df["readmitted"]
    knn = KNeighborsClassifier(5)
    rand = cross_val_score(
        knn, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="roc_auc"
    ).mean()
    grp = cross_val_score(
        knn, X, y, cv=GroupKFold(5), groups=df["patient_id"], scoring="roc_auc"
    ).mean()
    assert rand - grp > 0.15
    assert df.groupby("patient_id").size().eq(10).all()
    assert df["patient_id"].nunique() == 300


@pytest.mark.unit
def test_class_and_target_properties() -> None:
    assert dataset("binary_imbalanced.csv")["is_fraud"].mean() == pytest.approx(0.03)
    assert dataset("multiclass_rare.csv")["color"].value_counts().to_dict() == {
        "red": 250, "green": 200, "blue": 147, "rare": 3}  # fmt: skip
    assert dataset("multiclass_basic.csv")["species"].nunique() == 5
    claim = dataset("regression_skewed.csv")["claim_amount"]
    assert claim.min() > 0
    assert claim.skew() > 2
    assert dataset("binary_basic.csv")["Income"].isna().mean() == pytest.approx(0.06)
    assert len(dataset("binary_small.csv")) < 2000  # no-holdout path (A-003)


@pytest.mark.unit
def test_quality_issues_triggers_every_check() -> None:
    q = pd.read_csv(FIX / "datasets" / "quality_issues.csv", dtype=str, keep_default_na=False)
    n = len(q)
    assert q.duplicated().sum() == 20
    assert q.drop(columns="outcome").duplicated().sum() - q.duplicated().sum() == 5
    assert q["Record ID"].nunique() / n >= 0.95
    assert q["constant_col"].nunique() == 1
    assert 0.30 <= (q["missing_60"] == "").mean() <= 0.95
    assert (q["missing_97"] == "").mean() > 0.95
    assert q["near_constant"].value_counts(normalize=True).iloc[0] >= 0.995
    assert q["high_card_cat"].nunique() > 50
    assert 0.5 < q["very_high_card"].nunique() / n < 0.95
    numeric = pd.to_numeric(q["mixed_type"], errors="coerce")
    assert numeric.isna().sum() == 20
    assert numeric.notna().mean() >= 0.95
    assert np.isinf(pd.to_numeric(q["inf_values"], errors="coerce")).sum() == 5
    assert q["free_text"].str.len().mean() >= 50
    assert q["free_text"].nunique() / n >= 0.5
    assert set(q["flag_yn"]) == {"Y", "N"}
    assert q["zip_code"].nunique() <= 15


# ---------------------------------------------------------------- plans
PLAN_FILES = sorted(k for k in MANIFEST["files"] if k.startswith("plans/"))


@pytest.mark.unit
@pytest.mark.parametrize("name", PLAN_FILES)
def test_plan_fixture_schema_expectation(name: str) -> None:
    """Outcome `schema_invalid` ⇔ Pydantic rejects the plan; others pass the schema."""
    entry = MANIFEST["files"][name]
    text = (FIX / name).read_text(encoding="utf-8")
    if entry["expected_outcome"] == "schema_invalid":
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            PlannerOutput.model_validate_json(text)
    else:
        assert entry["expected_outcome"] in {"valid", "repaired", "invalid"}
        PlannerOutput.model_validate_json(text)


@pytest.mark.unit
def test_plan_catalogue_covers_p03_cases() -> None:
    required = {"valid", "invalid_json", "unknown_column", "excluded_column", "target_column",
                "prohibited_metric", "unknown_model", "too_many_features", "unknown_dsl_op",
                "out_of_range_params", "code_in_params", "contradictory_decisions",
                "mostly_unsafe_features", "prompt_injection_rationale"}  # fmt: skip
    assert required <= {Path(p).stem for p in PLAN_FILES}
    outcomes = {MANIFEST["files"][p]["expected_outcome"] for p in PLAN_FILES}
    assert outcomes == {"valid", "repaired", "invalid", "schema_invalid"}


@pytest.mark.unit
def test_llm_scripts_cover_every_provider_and_failure_mode() -> None:
    providers = {"groq", "gemini", "openrouter", "ollama"}
    scenarios = {"success", "malformed_json", "fenced_json", "rate_limit_then_success",
                 "server_error_then_success", "timeout_then_success", "auth_401"}  # fmt: skip
    for provider in providers:
        for scenario in scenarios:
            assert f"llm_responses/{provider}/{scenario}.json" in MANIFEST["files"]


@pytest.mark.unit
def test_malicious_upload_names() -> None:
    names = json.loads((FIX / "malicious" / "upload_names.json").read_text(encoding="utf-8"))
    assert len(names["long_300_chars"]) == 300
    assert ".." in names["path_traversal"]
    assert (FIX / "malicious" / "zip_renamed.csv").read_bytes()[:4] == b"PK\x03\x04"
