"""P16 support code: override checks, samples, catalog, RunManager UI helpers, results reader,
text escaping, config drafting, and `llm_rationale` events (PRD §27, FR-ING-04, FR-ING-06,
§24.5, §27.4).
"""

from __future__ import annotations

import ast
import io
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tabpilot.config import load_settings
from tabpilot.ingestion.overrides import check_override
from tabpilot.ingestion.samples import sample_bytes, sample_path, samples, sha256
from tabpilot.orchestration.context import WorkerOptions
from tabpilot.runtime import RunManager, UnknownRunError, read_artifacts
from tabpilot.runtime.catalog import metric_direction, metric_options, provider_options
from tabpilot.runtime.slots import acquire_slot
from tabpilot.schemas import (
    ErrorCode,
    EventType,
    MetricName,
    ProviderName,
    RunConfig,
    RunEvent,
    SemanticType,
    Stage,
    TabPilotError,
    Task,
)
from tabpilot.tracking.records import RunRecorder
from tabpilot.ui import state as sk
from tabpilot.ui.components.columns import warning_text
from tabpilot.ui.components.outputs import figure_caption, plan_source_text
from tabpilot.ui.config import draft, needs_consent
from tabpilot.ui.text import duration, md_escape, quote, size

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "datasets"


@pytest.fixture
def hosted(tmp_path: Path) -> Any:
    return load_settings({"TABPILOT_PROFILE": "hosted"}).model_copy(
        update={"data_dir": tmp_path / "data"}
    )


@pytest.fixture
def manager(tmp_path: Path, hosted: Any) -> RunManager:
    return RunManager(tmp_path / "runs", hosted, mode="thread")


# ------------------------------------------------------------------ overrides (FR-ING-04)
def test_override_numeric_requires_95_percent_parse() -> None:
    ok = pd.Series([str(i) for i in range(96)] + ["x", "y", "z", "w"])
    bad = pd.Series([str(i) for i in range(94)] + ["x"] * 6)
    assert check_override("c", ok, SemanticType.NUMERIC) is None
    message = check_override("c", bad, SemanticType.NUMERIC)
    assert message is not None
    assert "94%" in message
    assert "x" not in message.split("'c'")[1]


def test_override_categorical_always_allowed_and_text_refused() -> None:
    values = pd.Series([1.5, 2.5, 3.5])
    assert check_override("c", values, SemanticType.CATEGORICAL) is None
    assert check_override("c", values, SemanticType.TEXT) is not None


def test_override_boolean_and_datetime() -> None:
    assert check_override("c", pd.Series(["Y", "N", None]), SemanticType.BOOLEAN) is None
    assert check_override("c", pd.Series(["a", "b", "c"]), SemanticType.BOOLEAN) is not None
    dates = pd.Series(["2020-01-01", "2021-03-04"] * 50)
    assert check_override("c", dates, SemanticType.DATETIME) is None
    assert check_override("c", pd.Series(["soon"] * 10), SemanticType.DATETIME) is not None


# ------------------------------------------------------------------ samples (FR-ING-06)
def test_six_samples_including_multiclass_are_deterministic() -> None:
    catalog = samples()
    assert len(catalog) == 6
    assert {s.task for s in catalog} == {Task.BINARY, Task.MULTICLASS, Task.REGRESSION}
    for s in catalog:
        assert sample_bytes(s.sample_id) == sample_bytes(s.sample_id)
        assert "MIT" in s.license
        assert "synthetic" in s.title
        frame = pd.read_csv(io.BytesIO(sample_bytes(s.sample_id)))
        assert len(frame) == s.rows
        assert s.target in frame.columns


def test_sample_path_is_written_once(tmp_path: Path) -> None:
    path = sample_path("passengers", tmp_path)
    assert path.read_bytes() == sample_bytes("passengers")
    assert len(sha256("passengers")) == 64
    with pytest.raises(KeyError):
        sample_bytes("nope")


def test_validate_sample_and_check_target(manager: RunManager) -> None:
    for s in samples():
        upload = manager.validate_sample(s.sample_id)
        choice = manager.check_target(upload.run_id, s.target)
        assert choice.detection.task == s.task
        assert choice.task == s.task
        assert (choice.positive_label is not None) == (s.task == Task.BINARY)


# ------------------------------------------------------------------ catalog (§24.5, §10.2)
def test_metric_options_default_first_and_direction() -> None:
    options, default = metric_options(Task.BINARY, 0.05)
    assert default == MetricName.AVERAGE_PRECISION
    assert options[0] == default
    assert set(options) == {MetricName.ROC_AUC, MetricName.AVERAGE_PRECISION, MetricName.LOG_LOSS}
    assert metric_direction(MetricName.RMSE) == "lower is better"
    assert metric_direction(MetricName.ROC_AUC) == "higher is better"


def test_provider_options_reasons_never_expose_keys(hosted: Any) -> None:
    env = {"GROQ_API_KEY": "secret-value-123", "TABPILOT_GROQ_MODEL": "m"}
    settings = load_settings({"TABPILOT_PROFILE": "hosted", **env})
    by_name = {o.name: o for o in provider_options(settings, env)}
    assert by_name[ProviderName.GROQ].available
    assert by_name[ProviderName.GEMINI].reason == "no key (set GEMINI_API_KEY)"
    assert all("secret-value-123" not in (o.reason or "") + o.notice for o in by_name.values())
    no_model = provider_options(load_settings({"TABPILOT_PROFILE": "hosted"}), env)
    assert next(o for o in no_model if o.name == ProviderName.GROQ).reason == \
        "no model id configured"  # fmt: skip
    local = load_settings({"TABPILOT_PROFILE": "local"})
    hosted_ollama = load_settings({"TABPILOT_PROFILE": "hosted"}).model_copy(
        update={"provider_order": [ProviderName.OLLAMA]}
    )
    assert provider_options(hosted_ollama, {})[0].reason == "local profile only"
    assert ProviderName.OLLAMA in [o.name for o in provider_options(local, {})]


# ------------------------------------------------------------------ RunManager helpers
def test_upload_info_and_preview_rows(manager: RunManager) -> None:
    with (FIXTURES / "binary_basic.csv").open("rb") as fh:
        upload = manager.validate_upload(fh, "b.csv")
    assert manager.upload_info(upload.run_id) is upload
    rows = manager.preview_rows(upload.run_id)
    assert len(rows) == 20
    assert all(isinstance(v, str) for r in rows for v in r.values())
    assert set(rows[0]) == {c.original for c in upload.metadata.columns}
    with pytest.raises(UnknownRunError):
        manager.preview_rows("01J00000000000000000000000")


def test_check_target_raises_e_code(manager: RunManager) -> None:
    data = pd.DataFrame({"x": range(60), "y": [1] * 59 + [None]}).to_csv(index=False)
    upload = manager.validate_upload(io.BytesIO(data.encode()), "c.csv")
    y = next(c for c in upload.metadata.columns if c.original == "y")
    if y.role.value == "excluded":  # constant columns are excluded at ingestion
        with pytest.raises(TabPilotError) as err:
            manager.check_target(upload.run_id, "y", Task.REGRESSION)
    else:
        with pytest.raises(TabPilotError) as err:
            manager.check_target(upload.run_id, "y")
    assert err.value.code == ErrorCode.TARGET_CONSTANT


def test_check_override_through_manager(manager: RunManager) -> None:
    with (FIXTURES / "quality_issues.csv").open("rb") as fh:
        upload = manager.validate_upload(fh, "q.csv")
    assert manager.check_override(upload.run_id, "zip_code", SemanticType.CATEGORICAL) is None
    assert manager.check_override(upload.run_id, "free_text", SemanticType.NUMERIC) is not None
    with pytest.raises(ValueError, match="not in the file"):
        manager.check_override(upload.run_id, "missing", SemanticType.NUMERIC)


def test_discard_upload_deletes_files_and_memory(manager: RunManager) -> None:
    upload = manager.validate_sample("passengers")
    assert manager.run_dir(upload.run_id).exists()
    manager.discard_upload(upload.run_id)
    assert not manager.run_dir(upload.run_id).exists()
    with pytest.raises(UnknownRunError):
        manager.upload_info(upload.run_id)


def test_slot_notice(manager: RunManager) -> None:
    assert manager.slot_notice() is None
    held = acquire_slot(manager.root, 1)
    try:
        notice = manager.slot_notice()
        assert notice is not None
        assert notice.est_wait_s is None
    finally:
        held.release()


def test_sweep_forgets_uploads_whose_directory_is_gone(manager: RunManager) -> None:
    upload = manager.validate_sample("passengers")
    shutil.rmtree(manager.run_dir(upload.run_id))
    manager.sweep()
    with pytest.raises(UnknownRunError):
        manager.upload_info(upload.run_id)


# ------------------------------------------------------------------ results reader
def test_read_artifacts_missing_leaderboard_is_none(tmp_path: Path) -> None:
    assert read_artifacts(tmp_path) is None


def test_read_artifacts_precomputed_label(tmp_path: Path) -> None:
    (tmp_path / "leaderboard.json").write_text("[]")
    (tmp_path / "precomputed.json").write_text(json.dumps({"date": "2026-11-20"}))
    art = read_artifacts(tmp_path, bundle_path="missing.zip", bundle_sha256="0" * 64)
    assert art is not None
    assert art.precomputed == {"date": "2026-11-20"}
    assert art.bundle is None


def test_committed_precomputed_results_are_labeled_and_readable() -> None:
    """D-14: every bundled sample has a dated precomputed result without a bundle."""
    base = ROOT / "samples" / "precomputed"
    for s in samples():
        art = read_artifacts(base / s.sample_id)
        assert art is not None, s.sample_id
        assert art.precomputed is not None
        assert art.precomputed["date"]
        assert art.leaderboard
        assert art.report is not None
        assert art.bundle is None
        assert not list((base / s.sample_id).rglob("*.joblib"))
        assert not list((base / s.sample_id).rglob("*.zip"))


# ------------------------------------------------------------------ text and state
def test_md_escape_neutralizes_markdown_and_html() -> None:
    hostile = "[x](http://e.test) ![i](u) <b>b</b> **s** `c`\nnext"
    out = md_escape(hostile)
    assert "[x](" not in out
    assert "![" not in out
    assert "<b>" not in out
    assert "**" not in out
    assert "\n" not in out
    assert quote("a").startswith("> ")
    assert duration(75) == "1:15"
    assert duration(None) == "–"
    assert size(512) == "512 B"
    assert size(2048) == "2.0 KB"
    assert size(3 * 1024**2) == "3.0 MB"


def test_assert_primitive_rejects_frames() -> None:
    sk.assert_primitive({"a": 1, "b": [1, "x"], "c": {"d": None}})
    with pytest.raises(TypeError, match="frame"):
        sk.assert_primitive({"frame": pd.DataFrame({"a": [1]})})
    with pytest.raises(ValueError, match="unknown page"):
        sk.go({}, "Nowhere")


def test_no_unsafe_allow_html_in_ui() -> None:
    """P16: no `unsafe_allow_html` anywhere (user strings can never render as HTML)."""
    for path in (ROOT / "src" / "tabpilot" / "ui").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword):
                assert node.arg != "unsafe_allow_html", path
            if isinstance(node, ast.Attribute):
                assert node.attr != "html", path  # st.html renders raw HTML


# ------------------------------------------------------------------ config draft
def _state(**extra: Any) -> dict[str, Any]:
    info = {"column": "y", "detected": "binary", "confidence": "high", "task": "binary",
            "alternatives": [], "positive_label": "1", "positive_share": 0.4,
            "requires_confirmation": False}  # fmt: skip
    return {sk.UPLOAD_ID: "01J00000000000000000000000", sk.TARGET_INFO: info, **extra}


def test_draft_complete_with_no_llm(hosted: Any) -> None:
    d = draft(_state(**{sk.LLM_MODE: "off"}), hosted, [])
    assert d.config is not None
    assert not d.missing
    assert d.config.llm_mode == "off"
    assert d.config.provider_order == []
    assert d.config.primary_metric == MetricName.ROC_AUC
    assert d.config.time_budget_s == hosted.limits.time_budget_default_s


def test_draft_requires_task_click_when_ambiguous(hosted: Any) -> None:
    state = _state(**{sk.LLM_MODE: "off"})
    state[sk.TARGET_INFO] |= {"confidence": "ambiguous", "alternatives": ["regression"]}
    assert "confirm the task" in draft(state, hosted, []).missing
    state[sk.TASK] = "regression"
    d = draft(state, hosted, [])
    assert d.config is not None
    assert d.config.task == Task.REGRESSION
    assert d.config.primary_metric == MetricName.RMSE
    assert d.config.positive_label is None


def test_draft_requires_consent_per_provider(hosted: Any) -> None:
    state = _state(**{sk.LLM_MODE: "providers", sk.PROVIDERS: ["groq", "gemini"]})
    assert needs_consent(state, ["groq", "gemini"]) == ["groq", "gemini"]
    assert draft(state, hosted, ["groq", "gemini"]).config is None
    state[sk.CONSENTED] = ["groq"]
    assert needs_consent(state, ["groq", "gemini"]) == ["gemini"]
    state[sk.CONSENTED] = ["groq", "gemini"]
    d = draft(state, hosted, ["groq", "gemini"])
    assert d.config is not None
    assert d.config.consented_providers == [
        ProviderName.GROQ,
        ProviderName.GEMINI,
    ]
    assert "choose at least one" in " ".join(
        draft(_state(**{sk.LLM_MODE: "providers", sk.PROVIDERS: []}), hosted, ["groq"]).missing
    )


def test_draft_overrides_and_group(hosted: Any) -> None:
    state = _state(
        **{
            sk.LLM_MODE: "off",
            sk.GROUP: "cust",
            sk.OVERRIDES: {"zip": {"semantic_type": "categorical", "role": None}},
        }
    )
    d = draft(state, hosted, [])
    assert d.config is not None
    assert d.config.group_column == "cust"
    assert d.config.column_overrides[0].semantic_type == SemanticType.CATEGORICAL


def test_draft_missing_target_confirmation(hosted: Any) -> None:
    state = _state(**{sk.LLM_MODE: "off"})
    state[sk.TARGET_INFO]["requires_confirmation"] = True
    assert draft(state, hosted, []).config is None
    state[sk.CONFIRM_MISSING] = True
    assert draft(state, hosted, []).config is not None


# ------------------------------------------------------------------ llm_rationale events
def test_recorder_labels_llm_rationale_origin(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "01J00000000000000000000000", now=lambda: datetime.now(UTC))
    e = rec.event(Stage.PLAN, EventType.LLM_RATIONALE, source="plan", text="why")
    s = rec.event(Stage.PLAN, EventType.STAGE_COMPLETED, seconds=1.0)
    assert e.origin == "llm_rationale"
    assert s.origin == "system"


def test_rule_based_runs_emit_no_rationale(tmp_path: Path) -> None:
    """A rationale event exists only when the LLM produced the plan (fallback plans are ours).
    Local profile: the thread-mode worker shares the pytest process RSS (hosted limit 750 MB).
    """
    local = load_settings({"TABPILOT_PROFILE": "local"}).model_copy(
        update={"data_dir": tmp_path / "data"}
    )
    manager = RunManager(tmp_path / "local-runs", local, mode="thread")
    upload = manager.validate_sample("land_cover")
    choice = manager.check_target(upload.run_id, "land_cover")
    cfg = RunConfig(run_id=upload.run_id, target=choice.column, task=choice.task,
                    primary_metric=metric_options(choice.task)[1], time_budget_s=600, llm_mode="off",
                    execution_profile=manager.settings.profile)  # fmt: skip
    manager.start(cfg, options=WorkerOptions(tuning_mode="trials", trials_per_model=1,
                                             models=("linear",)))  # fmt: skip
    result = manager.wait(upload.run_id, timeout=240)
    assert result is not None
    assert result.status == "completed"
    assert not [e for e in manager.events(upload.run_id) if e.type == EventType.LLM_RATIONALE]
    art = manager.artifacts(upload.run_id)
    assert art is not None
    assert art.bundle is not None
    assert art.selected_entry_id
    manager.delete(upload.run_id)
    assert manager.artifacts(upload.run_id) is None
    assert np.isfinite(art.leaderboard[0].mean or 0.0)


# ------------------------------------------------------------------ U2 (KI-061…KI-063)
def _art(tmp_path: Path, experiment: dict[str, Any], events: list[Any]) -> Any:
    (tmp_path / "records").mkdir(parents=True)
    (tmp_path / "leaderboard.json").write_text("[]")
    (tmp_path / "records" / "experiment.json").write_text(json.dumps(experiment))
    (tmp_path / "records" / "events.jsonl").write_text(
        "".join(e.model_dump_json() + "\n" for e in events)
    )
    return read_artifacts(tmp_path)


def test_U2_KI_061_llm_fallback_is_named_on_the_plan_tab(tmp_path: Path) -> None:
    """US-F2: "LLM unavailable — rule-based plan used" when an LLM was chosen but failed."""
    fallback = RunEvent(run_id="01J00000000000000000000000", seq=0, ts=datetime.now(UTC),
                        stage=Stage.PLAN, type=EventType.PLANNER_FALLBACK,
                        data={"reason": "all providers failed"})  # fmt: skip
    failed = _art(tmp_path / "a", {"config": {"llm": {"enabled": True}}}, [fallback])
    assert plan_source_text(failed, "rule_based") == (
        "LLM unavailable — rule-based plan used (all providers failed)"
    )
    off = _art(tmp_path / "b", {"config": {"llm": {"enabled": False}}}, [])
    assert plan_source_text(off, "rule_based") == "rule-based plan (No LLM was selected)"
    assert plan_source_text(off, "llm") == "LLM planner (validated)"


def test_U2_KI_062_target_check_reads_the_typed_copy(manager: RunManager) -> None:
    """§27.2 < 2 s: target and override checks read one column of the typed copy, not the raw
    file (proved by corrupting the raw file after validation)."""
    with (FIXTURES / "binary_basic.csv").open("rb") as fh:
        upload = manager.validate_upload(fh, "b.csv")
    raw = next((manager.run_dir(upload.run_id) / "input").glob("raw.*"))
    assert (raw.parent / "typed.parquet").is_file()
    raw.write_bytes(b"\x00garbage")
    choice = manager.check_target(upload.run_id, "Churn")
    assert choice.task == Task.BINARY
    assert manager.check_override(upload.run_id, "Region", SemanticType.CATEGORICAL) is None
    assert manager.can_start(upload.run_id)


def test_U2_KI_063_captions_and_warnings_carry_meaning() -> None:
    base = ROOT / "samples" / "precomputed" / "passengers"
    art = read_artifacts(base)
    assert art is not None
    importance = next(f for f in art.figures if "importance" in f.caption.lower())
    assert "top features:" in figure_caption(art, importance.caption)
    other = next(f for f in art.figures if "importance" not in f.caption.lower()
                 and "shap" not in f.caption.lower())  # fmt: skip
    assert "cross-validated" in figure_caption(art, other.caption)
    assert warning_text("W-HIGH-MISSING:cabin_deck") == (
        "W-HIGH-MISSING — many missing values (cabin_deck)"
    )
