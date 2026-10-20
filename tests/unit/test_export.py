"""P14 export tests: T-ART-01…06, T-SEC-03, isolated `pipeline.py verify` (PRD §22, §24.9)."""

from __future__ import annotations

import ast
import json
import shutil
import site
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from tests.helpers.report_state import report_state
from tests.helpers.synthetic import synthetic

import tabpilot.export.bundle as bundle_module
from tabpilot.config import load_settings
from tabpilot.evaluation.metrics import Problem
from tabpilot.export import (
    BundleTooLargeError,
    ExportSettings,
    RawRowError,
    csv_text,
    dataset_slug,
    estimator_source,
    export_bundle,
    find_raw_rows,
    fixture_frame,
    known_categories,
    pinned,
    requirements_for,
    row_hashes,
    safe_cell,
    sha256_file,
)
from tabpilot.export.parity import column_rules
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import build_pipeline
from tabpilot.schemas import (
    CandidateSpec,
    ErrorCode,
    ExecutionProfileName,
    ModelName,
    TabPilotError,
    TargetTransform,
    Task,
)
from tabpilot.schemas.experiment import ModelConfig
from tabpilot.schemas.report import ArtifactManifest
from tabpilot_runtime import VotingEnsemble
from tabpilot_runtime.bundle import predict_outputs, read_table, verify

BIN = Problem(Task.BINARY, ("0", "1"), 1)
REG = Problem(Task.REGRESSION)
FIXED = datetime(2026, 10, 20, tzinfo=UTC)
LOCAL = ExecutionProfileName.LOCAL
LAYOUT = {
    "model/model.joblib", "pipeline.py", "requirements.txt", "manifest.json", "REPRODUCIBILITY.md",
    "metadata/dataset_metadata.json", "metadata/experiment.json", "metadata/leaderboard.csv",
    "metadata/plan.json", "metadata/plan_validation.json", "metadata/features.json",
    "metadata/preprocessing.json", "metadata/holdout.json", "report/report.html",
    "report/report.md", "report/figures/leaderboard.png", "parity/parity_input.csv",
    "parity/parity_expected.json", "tabpilot_runtime/__init__.py", "tabpilot_runtime/bundle.py",
}  # fmt: skip


@cache
def data() -> tuple[pd.DataFrame, np.ndarray, Any]:
    return synthetic(400)


def fit(model: ModelName, task: Task = Task.BINARY, y: Any = None, **spec: Any) -> Any:
    frame, y0, schema = data()
    cand = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=model, task=task, seed=1, n_jobs=1), **spec)  # fmt: skip
    return build_pipeline(cand, schema, ModelRegistry(load_settings({}))).fit(
        frame, y0 if y is None else y)  # fmt: skip


def regression_y() -> np.ndarray:
    return np.asarray(data()[0]["z"] * 3 + 40.0)


def report_dir(tmp: Path) -> Path:
    out = tmp / "report"
    (out / "figures").mkdir(parents=True)
    for name in ("report.html", "report.md", "report.json", "facts.json"):
        (out / name).write_text(f"{name}\n", encoding="utf-8")
    (out / "figures" / "leaderboard.png").write_bytes(b"\x89PNG" + bytes(64 * 1024))
    return out


def export(tmp: Path, model: Any, problem: Problem = BIN, **kw: Any) -> Any:
    frame = data()[0]
    state = report_state(problem, winner_oof=np.zeros((len(frame), 2), dtype=np.float32))
    settings = ExportSettings(out_dir=tmp / "out", max_bundle_mb=kw.pop("max_bundle_mb", 500),
                              profile=kw.pop("profile", LOCAL), target_original="Target Y",
                              dev_positions=list(range(len(frame))), n_rows_valid=len(frame),
                              scan_frame=frame, created_at=FIXED, **kw)  # fmt: skip
    return export_bundle(state, model, report_dir(tmp), settings)


KINDS = ["lightgbm", "xgboost", "catboost", "random_forest", "hist_gb", "linear", "ensemble",
         "log1p"]  # fmt: skip


def model_for(kind: str) -> tuple[Any, Problem]:
    if kind == "ensemble":
        model = VotingEnsemble([fit(ModelName.LIGHTGBM), fit(ModelName.LINEAR)], [0.6, 0.4])
        return model.fit(None, None), BIN
    if kind == "log1p":
        return fit(ModelName.LIGHTGBM, Task.REGRESSION, regression_y(),
                   target_transform=TargetTransform.LOG1P), REG  # fmt: skip
    return fit(ModelName(kind)), BIN


def unzip(path: Path, where: Path) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(where)
    return where


def isolated(bundle: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run `pipeline.py` with `-I -S`: only the stdlib, the bundle dir and third-party
    site-packages are importable. The editable-install `.pth` that exposes `src/` (and so
    `tabpilot`) is never processed.
    """
    packages = [p for p in site.getsitepackages() if Path(p).is_dir()]
    launcher = (
        "import sys, runpy\n"
        "sys.path[:] = [p for p in sys.path if 'site-packages' not in p] + "
        f"{packages!r}\n"
        "try:\n    import tabpilot\n    print('TABPILOT IMPORTABLE', tabpilot.__file__)\n"
        "except ImportError:\n    print('tabpilot not importable')\n"
        "print('sys.path =', sys.path)\n"
        f"sys.argv = ['pipeline.py', *{list(args)!r}]\n"
        "try:\n    runpy.run_path('pipeline.py', run_name='__main__')\n"
        "finally:\n    rt = sys.modules.get('tabpilot_runtime')\n"
        "    print('runtime from', rt.__file__ if rt else None, flush=True)\n"
    )  # fmt: skip
    return subprocess.run([sys.executable, "-I", "-S", "-c", launcher], cwd=bundle,
                          capture_output=True, text=True, timeout=300, check=False)  # fmt: skip


# ---------------------------------------------------------------- T-ART-01 structure + manifest
@pytest.mark.integration
def test_T_ART_01_layout_and_manifest_checksums(tmp_path: Path) -> None:
    result = export(tmp_path, fit(ModelName.LIGHTGBM))
    assert result.path.name == "tabpilot_telco_churn_01JAB3K8X2V7Q9M4T6R1Z0YH5C.zip"
    assert not (tmp_path / "out" / result.path.stem).exists()  # staging removed
    root = unzip(result.path, tmp_path / "u")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert files >= LAYOUT
    manifest = ArtifactManifest.model_validate_json((root / "manifest.json").read_text("utf-8"))
    assert {f.path for f in manifest.files} == files - {"manifest.json"}
    for entry in manifest.files:
        assert sha256_file(root / entry.path) == entry.sha256
        assert (root / entry.path).stat().st_size == entry.bytes
    roles = {f.path: f.role for f in manifest.files}
    assert roles["model/model.joblib"] == "model"
    assert roles["report/figures/leaderboard.png"] == "figure"
    assert roles["tabpilot_runtime/bundle.py"] == "runtime"
    assert manifest.reproducibility.r1_verified
    assert manifest.created_at == FIXED
    assert result.sha256 == sha256_file(result.path)
    assert result.r1.report.ok
    assert not result.r1.rebuilt


@pytest.mark.integration
def test_bundle_zip_is_deterministic(tmp_path: Path) -> None:
    model = fit(ModelName.LINEAR)
    first = export(tmp_path / "a", model)
    second = export(tmp_path / "b", model)
    assert first.sha256 == second.sha256


# ---------------------------------------------------------------- T-ART-02 R1
@pytest.mark.integration
def test_T_ART_02_reload_reproduces_expected_predictions(tmp_path: Path) -> None:
    model = fit(ModelName.LIGHTGBM)
    root = unzip(export(tmp_path, model).path, tmp_path / "u")
    report = verify(root, joblib.load(root / "model" / "model.joblib"))
    assert report.ok
    assert report.rows == 50
    assert report.max_abs_diff <= 1e-9
    expected = json.loads((root / "parity" / "parity_expected.json").read_text("utf-8"))
    assert set(expected["labels"]) <= {"0", "1"}
    assert len(expected["probabilities"][0]) == 2


# ---------------------------------------------------------------- isolated verify (decisive)
@pytest.mark.integration
@pytest.mark.parametrize("kind", ["lightgbm", "ensemble", "log1p", "catboost"])
def test_isolated_verify_without_tabpilot(tmp_path: Path, kind: str) -> None:
    model, problem = model_for(kind)
    root = unzip(export(tmp_path, model, problem).path, tmp_path / "clean")
    run = isolated(root, "verify")
    print(run.stdout)  # noqa: T201 - sys.path proof in the test log (P14 self-review)
    assert run.returncode == 0, run.stderr
    assert "tabpilot not importable" in run.stdout
    assert "TABPILOT IMPORTABLE" not in run.stdout
    assert "R1 parity PASSED: 50 rows" in run.stdout
    assert f"runtime from {root.resolve()}" in run.stdout


@pytest.mark.integration
def test_isolated_verify_fails_on_tampered_expectations(tmp_path: Path) -> None:
    root = unzip(export(tmp_path, fit(ModelName.LINEAR)).path, tmp_path / "u")
    path = root / "parity" / "parity_expected.json"
    expected = json.loads(path.read_text("utf-8"))
    expected["probabilities"][0][0] += 1e-6
    path.write_text(json.dumps(expected), encoding="utf-8")
    run = isolated(root, "verify")
    assert run.returncode == 1
    assert "R1 parity FAILED" in run.stdout


# ---------------------------------------------------------------- T-ART-03 / T-ART-04
@pytest.mark.integration
@pytest.mark.parametrize("kind", KINDS)
def test_T_ART_03_pipeline_py_passes_ruff_and_compiles(tmp_path: Path, kind: str) -> None:
    model, problem = model_for(kind)
    root = unzip(export(tmp_path, model, problem).path, tmp_path / "u")
    source = (root / "pipeline.py").read_text("utf-8")
    ast.parse(source)
    assert "joblib/pickle files can execute arbitrary code" in source
    ruff = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
    lint = subprocess.run([ruff, "check", "--isolated", "--target-version", "py311",
                           str(root / "pipeline.py")], capture_output=True, text=True, check=False)  # fmt: skip
    assert lint.returncode == 0, lint.stdout
    assert "import tabpilot\n" not in source
    assert "from tabpilot " not in source
    assert "from tabpilot." not in source


@pytest.mark.integration
def test_T_ART_04_predict_accepts_original_and_normalized_names(tmp_path: Path) -> None:
    root = unzip(export(tmp_path, fit(ModelName.LIGHTGBM)).path, tmp_path / "u")
    parity = read_table(root / "parity" / "parity_input.csv")
    assert "Amount ($)" in parity.columns  # the fixture uses original names
    parity.rename(columns={"Amount ($)": "amount"}).to_csv(root / "normalized.csv", index=False)
    for name in ("parity/parity_input.csv", "normalized.csv"):
        run = isolated(root, "predict", "--input", name, "--out", "preds.csv")
        assert run.returncode == 0, run.stderr
        preds = pd.read_csv(root / "preds.csv", dtype={"prediction": str})
        expected = json.loads((root / "parity" / "parity_expected.json").read_text("utf-8"))
        assert preds["prediction"].tolist() == expected["labels"]
        np.testing.assert_allclose(preds[["proba_0", "proba_1"]].to_numpy(),
                                   expected["probabilities"], atol=1e-12)  # fmt: skip


# ---------------------------------------------------------------- T-ART-05 R2
def dev_csv(path: Path, y: np.ndarray, extra_rows: pd.DataFrame | None = None) -> Path:
    frame = data()[0].rename(columns={"amount": "Amount ($)"}).copy()
    frame["Target Y"] = y
    if extra_rows is not None:
        frame = pd.concat([frame, extra_rows], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["linear", "lightgbm", "ensemble"])
def test_T_ART_05_retraining_parity(tmp_path: Path, kind: str) -> None:
    _, y, _ = data()
    model, problem = model_for(kind)
    root = unzip(export(tmp_path, model, problem).path, tmp_path / "u")
    dev_csv(root / "dev.csv", y)
    run = isolated(root, "train", "--data", "dev.csv", "--out", "retrained.joblib")
    assert run.returncode == 0, run.stderr
    retrained = joblib.load(root / "retrained.joblib")
    exported = joblib.load(root / "model" / "model.joblib")
    parity = read_table(root / "parity" / "parity_input.csv")
    a = predict_outputs(exported, parity, "binary", ["0", "1"])
    b = predict_outputs(retrained, parity, "binary", ["0", "1"])
    assert np.max(np.abs(np.asarray(a["probabilities"]) - b["probabilities"])) <= 1e-6


@pytest.mark.integration
def test_resplit_selects_the_recorded_dev_rows(tmp_path: Path) -> None:
    frame, y, _ = data()
    root = unzip(export(tmp_path, fit(ModelName.LINEAR)).path, tmp_path / "u")
    dev_csv(root / "full.csv", y)
    run = isolated(root, "train", "--data", "full.csv", "--resplit", "--out", "r.joblib")
    assert run.returncode == 0, run.stderr
    extra = frame.head(3).rename(columns={"amount": "Amount ($)"}).assign(**{"Target Y": 1})
    dev_csv(root / "other.csv", y, extra)
    bad = isolated(root, "train", "--data", "other.csv", "--resplit", "--out", "x.joblib")
    assert bad.returncode != 0
    assert "does not match the training file" in bad.stderr


# ---------------------------------------------------------------- parity failure path (§22.8)
class Shifted:
    """A reloaded model whose probabilities are off by 1e-6 (a corrupted serialization)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.classes_ = inner.classes_

    def predict_proba(self, X: Any) -> Any:
        return np.asarray(self.inner.predict_proba(X)) + 1e-6


@pytest.mark.integration
def test_parity_failure_rebuilds_once_then_raises(tmp_path: Path) -> None:
    calls: list[Path] = []

    def always_bad(path: Path) -> Any:
        calls.append(path)
        return Shifted(joblib.load(path))

    with pytest.raises(TabPilotError) as err:
        export(tmp_path, fit(ModelName.LINEAR), loader=always_bad)
    assert err.value.code == ErrorCode.EXPORT_PARITY
    assert len(calls) == 2
    assert list((tmp_path / "out").iterdir()) == []  # nothing offered


@pytest.mark.integration
def test_parity_failure_once_is_rebuilt(tmp_path: Path) -> None:
    calls: list[Path] = []

    def bad_once(path: Path) -> Any:
        calls.append(path)
        model = joblib.load(path)
        return Shifted(model) if len(calls) == 1 else model

    result = export(tmp_path, fit(ModelName.LINEAR), loader=bad_once)
    assert result.r1.rebuilt
    assert any("rebuilt" in w for w in result.manifest.warnings)


# ---------------------------------------------------------------- size limits
@pytest.mark.integration
def test_size_limit_drops_oof_then_figures_then_fails(tmp_path: Path) -> None:
    model = fit(ModelName.LINEAR)
    full = export(tmp_path / "full", model, include_oof=True)
    root = unzip(full.path, tmp_path / "u")
    total = sum(f.bytes for f in full.manifest.files)
    figures = (root / "report" / "figures" / "leaderboard.png").stat().st_size
    oof = (root / "oof_predictions.parquet").stat().st_size
    limit_mb = (total - oof - figures // 2) / 2**20  # needs both drops
    trimmed = export(tmp_path / "t", model, include_oof=True, max_bundle_mb=limit_mb)
    assert trimmed.excluded == ["OOF predictions", "report figures"]
    paths = {f.path for f in trimmed.manifest.files}
    assert "oof_predictions.parquet" not in paths
    assert not any(p.startswith("report/figures/") for p in paths)
    text = unzip(trimmed.path, tmp_path / "t2").joinpath("REPRODUCIBILITY.md").read_text("utf-8")
    assert "OOF predictions, report figures" in text
    with pytest.raises(BundleTooLargeError, match="above the"):
        export(tmp_path / "tiny", model, max_bundle_mb=0.001)
    assert list((tmp_path / "tiny" / "out").iterdir()) == []


@pytest.mark.integration
def test_oof_only_on_local_and_without_features(tmp_path: Path) -> None:
    local = unzip(export(tmp_path / "l", fit(ModelName.LINEAR), include_oof=True).path,
                  tmp_path / "lu")  # fmt: skip
    oof = pd.read_parquet(local / "oof_predictions.parquet")
    assert list(oof.columns) == ["dev_row", "pred_0", "pred_1"]
    hosted = export(tmp_path / "h", fit(ModelName.LINEAR), include_oof=True,
                    profile=ExecutionProfileName.HOSTED)  # fmt: skip
    assert "oof_predictions.parquet" not in {f.path for f in hosted.manifest.files}


# ---------------------------------------------------------------- T-ART-06 no raw rows
@pytest.mark.integration
def test_T_ART_06_no_dataset_row_in_the_bundle(tmp_path: Path) -> None:
    frame, _, _ = data()
    model = fit(ModelName.LIGHTGBM)
    root = unzip(export(tmp_path, model).path, tmp_path / "u")
    _, kinds = column_rules(model)
    real = set(row_hashes(frame, kinds))
    parity = read_table(root / "parity" / "parity_input.csv").rename(
        columns={"Amount ($)": "amount"})  # fmt: skip
    assert real.isdisjoint(row_hashes(parity, kinds))
    distinctive = {repr(float(v)) for v in frame["z"]} | {f"{v:.6f}" for v in frame["z"]}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".md", ".html", ".py", ".txt"}:
            text = path.read_text("utf-8")
            assert not any(v in text for v in distinctive), path


@pytest.mark.unit
def test_raw_row_scan_detects_a_copied_row(tmp_path: Path) -> None:
    frame, _, _ = data()
    model = fit(ModelName.LINEAR)
    leak = frame.head(1).rename(columns={"amount": "Amount ($)"})
    leak.to_csv(tmp_path / "leak.csv", index=False)
    assert find_raw_rows(tmp_path, frame, model) == ["leak.csv"]


@pytest.mark.integration
def test_raw_row_guard_blocks_the_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bundle_module, "find_raw_rows", lambda *_: ["parity/parity_input.csv"])
    with pytest.raises(RawRowError):
        export(tmp_path, fit(ModelName.LINEAR))
    assert list((tmp_path / "out").iterdir()) == []


@pytest.mark.unit
def test_fixture_redraws_rows_that_equal_real_rows() -> None:
    model = fit(ModelName.LINEAR)
    first = fixture_frame(model, None, seed=5)
    _, kinds = column_rules(model)
    column_map, _ = column_rules(model)
    banned = set(row_hashes(first.rename(columns=column_map), kinds)[:10])
    again = fixture_frame(model, None, seed=5, forbidden=banned)
    assert banned.isdisjoint(row_hashes(again.rename(columns=column_map), kinds))


@pytest.mark.unit
def test_fixture_uses_known_categories_and_profile_ranges() -> None:
    from tests.helpers.report_state import contract  # noqa: PLC0415

    from tabpilot.schemas import ProfileSummary  # noqa: PLC0415

    model = fit(ModelName.LINEAR)
    known = known_categories(model)
    assert "__rare__" not in {c for cats in known.values() for c in cats}
    profile = ProfileSummary.model_validate(contract("ProfileSummary"))
    column = profile.columns[0].model_copy(update={"name": "z", "type": "numeric", "p5": 1.0,
                                                    "p95": 2.0})  # fmt: skip
    profile = profile.model_copy(update={"columns": [column]})
    frame = fixture_frame(model, profile, seed=1)
    z = pd.to_numeric(frame["z"]).dropna()
    assert z.between(1.0, 2.0).all()
    assert frame["z"].isna().mean() <= 0.3
    for column_name in ("low", "mid"):
        assert set(frame[column_name].dropna()) <= set(known[column_name])
    assert set(frame["flag"].dropna()) <= {"true", "false"}
    assert len(frame) == 50


# ---------------------------------------------------------------- T-SEC-03 formula injection
@pytest.mark.unit
def test_T_SEC_03_formula_cells_are_prefixed() -> None:
    for bad in ("=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tx", "\rx"):
        assert safe_cell(bad) == "'" + bad
    assert safe_cell(-2.5) == -2.5
    assert safe_cell("plain") == "plain"
    text = csv_text(["=h"], [['=HYPERLINK("http://x")', 3]])
    assert text.splitlines()[0] == "'=h"
    assert text.splitlines()[1].startswith("\"'=HYPERLINK")


@pytest.mark.integration
def test_T_SEC_03_leaderboard_csv_in_the_bundle_is_safe(tmp_path: Path) -> None:
    root = unzip(export(tmp_path, fit(ModelName.LINEAR)).path, tmp_path / "u")
    board = (root / "metadata" / "leaderboard.csv").read_text("utf-8")
    assert board.splitlines()[0].startswith("entry_id,model")
    assert "0-lightgbm-3c9e1a20" in board


# ---------------------------------------------------------------- codegen and requirements
@pytest.mark.unit
@pytest.mark.parametrize("name", list(ModelName))
def test_generated_constructor_is_a_fixpoint(name: ModelName) -> None:
    model = fit(name)
    source = estimator_source(model)
    namespace: dict[str, Any] = {}
    code = "\n".join(source.all_imports) + f"\nmodel = {source.expression}\n"
    exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - template-generated code under test, no LLM input
    assert (
        estimator_source(namespace["model"]).expression == estimator_source(clone(model)).expression
    )


@pytest.mark.unit
def test_injected_column_names_stay_string_literals() -> None:
    from tabpilot_runtime import ColumnNameMapper  # noqa: PLC0415

    evil = "x'); __import__('os').system('echo pwned') #"
    source = estimator_source(ColumnNameMapper({evil: "x", '"""': "y"}))
    tree = ast.parse(source.expression, mode="eval")
    call = tree.body
    assert isinstance(call, ast.Call)
    keys = call.keywords[0].value
    assert isinstance(keys, ast.Dict)
    assert [k.value for k in keys.keys if isinstance(k, ast.Constant)] == [evil, '"""']


@pytest.mark.unit
def test_requirements_pin_exact_versions_of_what_is_used() -> None:
    from importlib.metadata import version  # noqa: PLC0415

    lines = requirements_for(fit(ModelName.LIGHTGBM))
    assert f"lightgbm=={version('lightgbm')}" in lines
    assert f"scikit-learn=={version('scikit-learn')}" in lines
    assert not any(line.startswith(("xgboost", "catboost")) for line in lines)
    assert all("==" in line for line in lines)
    ensemble = VotingEnsemble([fit(ModelName.XGBOOST), fit(ModelName.CATBOOST)], [0.5, 0.5])
    assert {line.split("==")[0] for line in requirements_for(ensemble)} >= {"xgboost", "catboost"}
    with pytest.raises(RuntimeError, match="not installed"):
        pinned(["no-such-distribution-xyz"])


@pytest.mark.unit
def test_dataset_slug() -> None:
    assert dataset_slug("Telco Churn (v2).CSV") == "telco_churn_v2"
    assert dataset_slug("../../etc/passwd") == "passwd"
    assert dataset_slug("日本.csv") == "dataset"
    assert len(dataset_slug("a" * 100 + ".csv")) == 40


@pytest.mark.unit
def test_codegen_literals_round_trip() -> None:
    from sklearn.preprocessing import FunctionTransformer  # noqa: PLC0415

    from tabpilot.export import CodegenError  # noqa: PLC0415

    values = [np.float32(1.5), np.int64(3), float("nan"), float("inf"), -float("inf"), (1,), [],
              {}, (), np.asarray([1.0, 2.0]), {"k": [1, 2, {"x": None}]}, b"raw", True]  # fmt: skip
    source = estimator_source(FunctionTransformer(func=np.log1p, kw_args={"values": values}))
    namespace: dict[str, Any] = {}
    exec(compile("\n".join(source.all_imports) + f"\nt = {source.expression}\n", "<g>", "exec"),  # noqa: S102 - generated code under test
         namespace)  # fmt: skip
    rebuilt = namespace["t"].kw_args["values"]
    assert rebuilt[0] == np.float32(1.5)
    assert isinstance(rebuilt[0], np.float32)
    assert np.isnan(rebuilt[2])
    assert rebuilt[3] == float("inf")
    assert rebuilt[4] == -float("inf")
    assert rebuilt[5] == (1,)
    assert np.array_equal(rebuilt[9], [1.0, 2.0])
    assert rebuilt[10] == {"k": [1, 2, {"x": None}]}
    assert namespace["t"].func is np.log1p

    class Local:
        def __init__(self, a: int) -> None:
            self.b = a

    with pytest.raises(CodegenError, match="cannot import"):
        estimator_source(Local(1))  # a class that cannot be imported by name
    with pytest.raises(CodegenError, match="cannot import"):
        estimator_source(lambda x: x)


@pytest.mark.unit
def test_runtime_prediction_csv_is_formula_safe(tmp_path: Path) -> None:
    from tabpilot_runtime.bundle import compare, write_predictions  # noqa: PLC0415

    outputs = {"labels": ["=cmd", "ok"], "probabilities": [[0.9, 0.1], [0.2, 0.8]]}
    write_predictions(tmp_path / "p.csv", outputs, ["=cmd", "ok"])
    text = (tmp_path / "p.csv").read_text("utf-8")
    assert text.splitlines()[0] == "prediction,proba_=cmd,proba_ok"  # starts with "p": safe
    assert text.splitlines()[1].startswith("'=cmd,")
    write_predictions(tmp_path / "r.csv", {"values": [-1.5]}, [])
    assert (tmp_path / "r.csv").read_text("utf-8").splitlines() == ["prediction", "-1.5"]
    assert not compare({"values": [1.0, 2.0]}, {"values": [1.0]}).ok  # shape mismatch
