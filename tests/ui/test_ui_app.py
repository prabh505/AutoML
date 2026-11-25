"""Streamlit AppTest tests for the P16 UI (PRD §27, §3.2, §31.3 T-UI-01…04, US-06).

Real `RunManager` and real runs (offline stub LLM, fast worker options) unless a test injects
one of the controlled workers from `tests/helpers/fake_workers.py`.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest
from tests.helpers.fake_workers import polite, quick_fail
from tests.ui.conftest import UI

from tabpilot.llm.cache import canonical
from tabpilot.llm.stub import StubProvider
from tabpilot.runtime.slots import acquire_slot
from tabpilot.schemas import EventType, RunEvent, Stage
from tabpilot.ui import state as sk
from tabpilot.ui.components.status import STEPS, steps
from tabpilot.ui.pages.run import partial_leaderboard

pytestmark = pytest.mark.ui
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "datasets"


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _state(ui: UI) -> dict[str, Any]:
    state = ui.at.session_state
    return {k: state[k] for k in state._state.filtered_state}


def _upload(ui: UI, data: bytes, name: str) -> None:
    ui.at.session_state[sk.PAGE] = "Upload"  # the uploader lives on the Upload page
    ui.run()
    ui.upload.put(data, name)
    ui.run()


def _configure_basic(ui: UI) -> None:
    """binary_basic.csv → Configure with target `churn`, stub LLM, shortest budget."""
    _upload(ui, (FIXTURES / "binary_basic.csv").read_bytes(), "binary_basic.csv")
    assert ui.page() == "Configure"
    ui.at.selectbox(key=sk.TARGET).set_value("Churn").run()
    ui.at.radio(key=sk.LLM_MODE).set_value("stub").run()
    ui.at.slider(key=sk.BUDGET).set_value(600).run()


def _start_and_finish(ui: UI) -> None:
    ui.at.button(key="start").click().run()
    assert ui.page() == "Run", ui.texts()
    ui.wait_until(
        lambda: (
            ui.page() != "Run"
            or ui.manager.status(ui.at.session_state[sk.RUN_ID]).state != "running"
        )
    )
    ui.run()


def test_T_UI_01_upload_configure_run_results(make_ui: Any) -> None:
    """§3.2 journey: upload → configure → run (stub LLM) → results with a downloadable bundle."""
    ui = make_ui()
    _configure_basic(ui)
    assert not ui.at.exception
    assert not ui.at.button(key="start").disabled
    _start_and_finish(ui)
    assert ui.page() == "Results", ui.texts()
    assert not ui.at.exception
    assert any(s.value.startswith("Completed in ") for s in ui.at.success)
    assert [t.label for t in ui.at.tabs] == ["Summary", "Data profile", "Plan", "Leaderboard",
                                             "Explainability", "Report", "Downloads"]  # fmt: skip
    run_id = ui.at.session_state[sk.RUN_ID]
    art = ui.manager.artifacts(run_id)
    assert art is not None
    assert art.bundle is not None
    assert art.bundle.sha256 in ui.texts()  # SHA-256 shown next to the bundle (§27.2)
    assert "model.joblib is a Python pickle" in ui.texts()  # joblib warning
    assert any("★ Selected" in str(row) for df in ui.at.dataframe for row in df.value.values)
    # session state holds only the run id and UI choices (no DataFrame, no upload bytes)
    values = _state(ui)
    sk.assert_primitive(values)
    assert not any(isinstance(v, pd.DataFrame | pd.Series | bytes) for v in values.values())


def test_session_state_has_no_dataframe_after_start(make_ui: Any) -> None:
    ui = make_ui(worker=polite)
    _configure_basic(ui)
    ui.at.button(key="start").click().run()
    assert ui.page() == "Run"
    values = _state(ui)
    assert not any(isinstance(v, pd.DataFrame | pd.Series) for v in values.values())
    sk.assert_primitive(values)
    ui.manager.cancel(values[sk.RUN_ID])


def _ambiguous_csv() -> bytes:
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    return _csv(
        pd.DataFrame(
            {
                "x1": x,
                "x2": rng.normal(size=n),
                "rating": np.clip(np.round(3 + 1.5 * x), 1, 7).astype(int),
            }
        )
    )


def test_T_UI_02_ambiguous_task_requires_choice(make_ui: Any) -> None:
    """FR-TGT-02 / US-03: radio without default; no start until clicked; metrics follow task."""
    ui = make_ui()
    _upload(ui, _ambiguous_csv(), "ratings.csv")
    ui.at.selectbox(key=sk.TARGET).set_value("rating").run()
    ui.at.radio(key=sk.LLM_MODE).set_value("off").run()
    radio = ui.at.radio(key=sk.TASK)
    assert radio.value is None
    assert len(radio.options) >= 2
    assert ui.at.button(key="start").disabled
    assert "confirm the task" in ui.texts()
    assert not [s for s in ui.at.selectbox if s.key == sk.METRIC]
    radio.set_value("multiclass").run()
    assert not ui.at.button(key="start").disabled
    assert set(ui.at.selectbox(key=sk.METRIC).options) == {
        "log_loss (lower is better)", "macro_f1 (higher is better)",
        "balanced_accuracy (higher is better)"}  # fmt: skip
    ui.at.radio(key=sk.TASK).set_value("regression").run()
    assert ui.at.selectbox(key=sk.METRIC).value == "rmse"


def test_T_UI_03_llm_rationale_labeled(make_ui: Any) -> None:
    """US-07 / §27.4: the LLM rationale is a labeled quoted callout, never a system line."""
    ui = make_ui()
    _configure_basic(ui)
    _start_and_finish(ui)
    run_id = ui.at.session_state[sk.RUN_ID]
    events = ui.manager.events(run_id)
    rationale = [e for e in events if e.origin == "llm_rationale"]
    assert rationale
    assert all(e.type.value == "llm_rationale" for e in rationale)
    labeled = [i.value for i in ui.at.info if "LLM-generated rationale" in i.value]
    assert labeled
    assert all("\n\n> " in v for v in labeled)
    # the Run page event log: rationales in a labeled callout, never among the system lines
    log = AppTest.from_function(_render_log, args=([e.model_dump_json() for e in events],))
    log.run()
    assert not log.exception
    lines = "\n".join(str(t.value) for t in log.text)
    assert "llm_rationale" not in lines
    assert "stage_completed" in lines
    callouts = [i.value for i in log.info]
    assert len(callouts) == len(rationale)
    assert all("LLM-generated rationale" in c for c in callouts)
    # every stage_completed event is reflected as a finished step (§27.4 state change)
    done = {s.label for s in steps(events, finished=True) if s.status == "done"}
    for event in events:
        if event.type.value == "stage_completed":
            label = next((lbl for lbl, stages in STEPS if event.stage in stages), None)
            assert label is None or label in done
    finished = {r["Entry"] for r in partial_leaderboard(events)}
    board = {e.entry_id for e in ui.manager.artifacts(run_id).leaderboard  # type: ignore[union-attr]
             if e.feature_set != "ensemble"}  # fmt: skip
    assert finished >= board  # each leaderboard row was announced by candidate_* events


def _render_log(payloads: list[str]) -> None:
    from tabpilot.schemas import RunEvent  # noqa: PLC0415 - AppTest runs this source alone
    from tabpilot.ui.components import events as event_log  # noqa: PLC0415

    event_log.render([RunEvent.model_validate_json(p) for p in payloads])


def _zip_as_csv() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.csv", "a,b\n1,2\n")
    return buf.getvalue()


def test_T_UI_04_error_states_render(make_ui: Any) -> None:
    """NFR-UX-01: E-code messages, no stack traces, for upload, target and run failures."""
    ui = make_ui(worker=quick_fail)
    _upload(ui, _zip_as_csv(), "data.csv")
    assert ui.page() == "Upload"
    assert not ui.at.exception
    assert "E-ARCHIVE-UNSUPPORTED" in ui.texts()
    assert "Traceback" not in ui.texts()
    constant = _csv(pd.DataFrame({"x": range(100), "y": ["same"] * 100}))
    _upload(ui, constant, "const.csv")
    assert ui.page() == "Configure"
    # a constant column is excluded at ingestion, so it is never offered as a target (the
    # E-TARGET inline error itself is driven in test_U2_KI_060)
    assert "y" not in ui.at.selectbox(key=sk.TARGET).options
    # a failed run shows its E-code message and what completed
    _upload(ui, (FIXTURES / "binary_basic.csv").read_bytes(), "binary_basic.csv")
    ui.at.selectbox(key=sk.TARGET).set_value("Churn").run()
    ui.at.radio(key=sk.LLM_MODE).set_value("off").run()
    _start_and_finish(ui)
    assert ui.page() == "Run"
    assert not ui.at.exception
    assert "E-NO-VALID-MODEL" in ui.texts()
    assert "Completed before the failure" in ui.texts()
    assert "Traceback" not in ui.texts()


def test_cancelled_state_and_data_deleted(make_ui: Any) -> None:
    """§27.3 Cancelled: "Run cancelled. Data deleted." and the run directory is gone."""
    ui = make_ui(worker=polite)
    _configure_basic(ui)
    ui.at.button(key="start").click().run()
    run_id = ui.at.session_state[sk.RUN_ID]
    ui.at.button(key="cancel").click().run()
    ui.run()
    assert "Run cancelled. Data deleted." in ui.texts()
    assert not ui.manager.run_dir(run_id).exists()


def test_waiting_for_slot_offers_retry_and_sample(make_ui: Any) -> None:
    """§27.3 Waiting for slot: the wait text, Retry, and "View sample result" when available."""
    ui = make_ui()
    _configure_basic(ui)
    held = acquire_slot(ui.manager.root, 1)
    try:
        ui.at.button(key="start").click().run()
        assert ui.page() == "Configure"
        assert "Another analysis is running (est. wait" in ui.texts()
        assert ui.at.button(key="start").label == "Retry"
        view = [b for b in ui.at.button if b.key == "view_sample"]
        if view:  # precomputed results are committed under samples/precomputed (D-14)
            view[0].click().run()
            assert ui.page() == "Results"
            assert "precomputed on" in ui.texts()
            ui.at.button(key="back_sample").click().run()
    finally:
        held.release()
    ui.at.button(key="start").click().run()
    assert ui.page() == "Run"
    ui.manager.wait(ui.at.session_state[sk.RUN_ID], timeout=240)


def test_inspector_equals_request_payload(make_ui: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """US-06: the inspector shows exactly the planner request's user JSON."""
    sent: list[str] = []
    original = StubProvider.complete

    def spy(self: StubProvider, req: Any) -> Any:
        if req.purpose == "plan":
            sent.append(canonical(req.user_json))
        return original(self, req)

    monkeypatch.setattr(StubProvider, "complete", spy)
    ui = make_ui()
    _configure_basic(ui)
    ui.at.button(key="inspect_build").click().run()
    shown = [c.value for c in ui.at.code if c.language == "json"]
    assert len(shown) == 1
    _start_and_finish(ui)
    assert sent
    assert sent[0] == shown[0]


def test_delete_my_data_now_removes_run_directory(make_ui: Any) -> None:
    """§24.7 "Delete now": the whole run directory is deleted and confirmed."""
    ui = make_ui()
    _configure_basic(ui)
    _start_and_finish(ui)
    run_id = ui.at.session_state[sk.RUN_ID]
    assert ui.manager.run_dir(run_id).exists()
    ui.at.button(key="delete_run").click().run()
    assert not ui.manager.run_dir(run_id).exists()
    assert "Your data has been deleted." in ui.texts()
    assert ui.page() == "Upload"


def test_delete_before_start_removes_upload(make_ui: Any) -> None:
    ui = make_ui()
    _upload(ui, (FIXTURES / "binary_basic.csv").read_bytes(), "binary_basic.csv")
    upload_id = ui.at.session_state[sk.UPLOAD_ID]
    assert ui.manager.run_dir(upload_id).exists()
    ui.at.button(key="delete_upload").click().run()
    assert not ui.manager.run_dir(upload_id).exists()
    assert "Your data has been deleted." in ui.texts()


def test_user_strings_are_not_rendered_as_markup(make_ui: Any) -> None:
    """A hostile column name is escaped wherever Markdown would render it."""
    name = "[click](http://x.test)<b>bold</b>"
    data = _csv(pd.DataFrame({name: [0, 1] * 60, "y": ["a", "b", "b"] * 40}))
    ui = make_ui()
    _upload(ui, data, "evil.csv")
    ui.at.selectbox(key=sk.TARGET).set_value(name).run()
    rendered = "\n".join(str(m.value) for m in ui.at.markdown) + ui.texts()
    assert "[click](http" not in rendered


def test_sample_card_starts_configure_with_target(make_ui: Any) -> None:
    """FR-ING-06: a sample goes through the upload path with its pre-selected target."""
    ui = make_ui()
    ui.run()
    ui.at.button(key="sample_land_cover").click().run()
    assert ui.page() == "Configure"
    assert ui.at.selectbox(key=sk.TARGET).value == "land_cover"
    assert ui.at.selectbox(key=sk.METRIC).value == "log_loss"


# ------------------------------------------------------------------ U2 fixes (KI-056…KI-063)
def test_U2_KI_056_rerun_with_changes_reloads_sample_and_keeps_choices(make_ui: Any) -> None:
    """§3.2 stage 7: a re-run keeps every choice, gets a new id, and uses the changed setting."""
    ui = make_ui()
    ui.run()
    ui.at.button(key="sample_land_cover").click().run()
    ui.at.radio(key=sk.LLM_MODE).set_value("off").run()
    ui.at.slider(key=sk.BUDGET).set_value(600).run()
    _start_and_finish(ui)
    assert ui.page() == "Results"
    first = ui.at.session_state[sk.RUN_ID]
    ui.at.button(key="rerun").click().run()
    assert ui.page() == "Configure"
    assert ui.at.session_state[sk.UPLOAD_ID] != first
    assert ui.at.selectbox(key=sk.TARGET).value == "land_cover"
    assert ui.at.radio(key=sk.LLM_MODE).value == "off"
    assert ui.at.slider(key=sk.BUDGET).value == 600
    ui.at.slider(key=sk.BUDGET).set_value(630).run()
    _start_and_finish(ui)
    second = ui.at.session_state[sk.RUN_ID]
    assert second != first
    assert ui.manager.run_config(second).time_budget_s == 630
    assert ui.manager.run_config(first).time_budget_s == 600


def test_U2_KI_056_rerun_own_file_asks_for_it_again(make_ui: Any) -> None:
    ui = make_ui()
    _configure_basic(ui)
    _start_and_finish(ui)
    first = ui.at.session_state[sk.RUN_ID]
    # Configure for the finished run offers the re-run instead of a Start that reopens it
    ui.at.session_state[sk.PAGE] = "Configure"
    ui.run()
    assert not [b for b in ui.at.button if b.key == "start"]
    ui.at.button(key="rerun_configure").click().run()
    assert ui.page() == "Upload"
    assert "Upload the file again" in ui.texts()
    ui.upload.put((FIXTURES / "binary_basic.csv").read_bytes(), "binary_basic.csv")
    ui.run()
    assert ui.page() == "Configure"
    assert ui.at.session_state[sk.UPLOAD_ID] != first
    assert ui.at.selectbox(key=sk.TARGET).value == "Churn"
    assert ui.at.radio(key=sk.LLM_MODE).value == "stub"
    assert not ui.at.button(key="start").disabled
    assert not ui.manager.can_start(first)


def test_U2_KI_057_no_new_data_while_a_run_is_active(make_ui: Any) -> None:
    ui = make_ui(worker=polite)
    _configure_basic(ui)
    ui.at.button(key="start").click().run()
    run_id = ui.at.session_state[sk.RUN_ID]
    _upload(ui, (FIXTURES / "binary_basic.csv").read_bytes(), "other.csv")
    assert ui.page() == "Upload"
    assert ui.at.session_state[sk.RUN_ID] == run_id
    assert "An analysis is running" in ui.texts()
    assert not [b for b in ui.at.button if (b.key or "").startswith("sample_")]
    ui.manager.cancel(run_id)


def test_U2_KI_058_hostile_rationale_and_name_are_escaped(make_ui: Any) -> None:
    """The raw Markdown sent to Streamlit holds the escaped form, never a live link."""
    hostile = "[click](http://evil.test) <img src=x onerror=alert(1)> **bold**"
    event = RunEvent(run_id="01J00000000000000000000000", seq=0, ts=datetime.now(UTC),
                     stage=Stage.PLAN, type=EventType.LLM_RATIONALE, origin="llm_rationale",
                     data={"source": "plan", "text": hostile})  # fmt: skip
    log = AppTest.from_function(_render_log, args=([event.model_dump_json()],))
    log.run()
    raw = [i.value for i in log.info]
    assert len(raw) == 1
    assert "\\[click\\]\\(http" in raw[0]
    for live in (r"(?<!\\)\[click(?<!\\)\]", r"(?<!\\)<img", r"(?<!\\)\*(?<!\\)\*bold"):
        assert not re.search(live, raw[0]), live  # every markup character is escaped
    ui = make_ui()
    _upload(ui, (FIXTURES / "binary_basic.csv").read_bytes(), "a[b](e.test)*x*.csv")
    shown = "\n".join(str(m.value) for m in ui.at.markdown)
    assert "a\\[b\\]\\(e\\.test\\)\\*x\\*\\.csv" in shown  # the display name, escaped
    assert not re.search(r"(?<!\\)\[b(?<!\\)\]", shown)


def test_U2_KI_059_polling_stops_when_the_run_ends(make_ui: Any,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """§27.5: the Run fragment polls every second while running and not after the end."""
    seen: list[Any] = []
    real = streamlit.fragment

    def spy(func: Any = None, **kwargs: Any) -> Any:
        seen.append(kwargs.get("run_every"))
        return real(func, **kwargs)

    monkeypatch.setattr(streamlit, "fragment", spy)
    ui = make_ui(worker=polite)
    _configure_basic(ui)
    ui.at.button(key="start").click().run()
    assert seen
    assert seen[-1] == 1.0
    ui.manager.cancel(ui.at.session_state[sk.RUN_ID])
    ui.run()
    assert seen[-1] is None


def test_U2_KI_060_target_error_infeasible_and_partial_states(make_ui: Any) -> None:
    n = 120
    data = _csv(pd.DataFrame({"x": np.arange(n) % 7, "y": ["a"] * 60 + ["b"] * 59 + ["c"]}))
    ui = make_ui()
    _upload(ui, data, "tiny_class.csv")
    ui.at.selectbox(key=sk.TARGET).set_value("y").run()
    errors_shown = "\n".join(e.value for e in ui.at.error)
    assert "E\\-TARGET\\-CLASS\\-TOO\\-SMALL" in errors_shown
    assert ui.at.button(key="start").disabled
    infeasible = make_ui(limits={"rss_soft_fraction_of_ram": 1e-6,
                                 "rss_hard_fraction_of_ram": 2e-6})  # fmt: skip
    _configure_basic(infeasible)
    assert "E-RESOURCE-INFEASIBLE" in infeasible.texts()
    assert infeasible.at.button(key="start").disabled
    banner = AppTest.from_function(_render_banner)
    banner.run()
    assert [w.value for w in banner.warning] == [
        "Stopped at time budget after train_candidates; results are valid but limited"
    ]


def _render_banner() -> None:
    from tabpilot.schemas import FinalResult, RunResult  # noqa: PLC0415 - AppTest source
    from tabpilot.schemas.experiment import HoldoutResult  # noqa: PLC0415
    from tabpilot.ui.pages.results import banner  # noqa: PLC0415

    final = FinalResult(selected_entry_id="0-linear",
                        holdout=HoldoutResult(present=False, reason="small data"))  # fmt: skip
    banner(RunResult(run_id="01J00000000000000000000000", status="completed_partial",
                     duration_s=10.0, stopped_after_stage="train_candidates",
                     final=final))  # fmt: skip


# ------------------------------------------------------------------ U2b (KI-065…KI-068)
def test_U2b_KI_065_configure_never_offers_a_running_run_for_rerun(make_ui: Any) -> None:
    ui = make_ui(worker=polite)
    _configure_basic(ui)
    ui.at.button(key="start").click().run()
    run_id = ui.at.session_state[sk.RUN_ID]
    ui.at.session_state[sk.PAGE] = "Configure"
    ui.run()
    assert "An analysis is running" in ui.texts()
    assert not [b for b in ui.at.button if b.key in ("rerun_configure", "start")]
    ui.at.button(key="to_run_cfg").click().run()
    assert ui.page() == "Run"
    assert ui.at.session_state[sk.RUN_ID] == run_id
    ui.manager.cancel(run_id)


def _x_file(values: list[str]) -> bytes:
    n = len(values)
    return _csv(pd.DataFrame({"x": values, "z": np.arange(n) % 5,
                              "y": (["a", "b"] * n)[:n]}))  # fmt: skip


def test_U2b_KI_066_rerun_rechecks_kept_overrides(make_ui: Any) -> None:
    """A kept override that no longer parses on the re-uploaded file is removed with a notice."""
    ui = make_ui()
    _upload(ui, _x_file([str(i) for i in range(120)]), "first.csv")
    ui.at.selectbox(key="ov_column").set_value("x").run()
    ui.at.selectbox(key="ov_type").set_value("numeric").run()
    next(b for b in ui.at.button if b.label == "Apply override").click().run()
    assert ui.at.session_state[sk.OVERRIDES] == {"x": {"semantic_type": "numeric", "role": None}}
    ui.at.selectbox(key=sk.TARGET).set_value("y").run()
    ui.at.radio(key=sk.LLM_MODE).set_value("off").run()
    _start_and_finish(ui)
    ui.at.button(key="rerun").click().run()
    assert ui.page() == "Upload"
    ui.upload.put(_x_file([f"word{i}" for i in range(120)]), "second.csv")
    ui.run()
    assert ui.page() == "Configure"
    assert ui.at.session_state[sk.OVERRIDES] == {}
    assert "Column changes that no longer fit the new file were removed" in ui.texts()
    assert "must parse" in ui.texts()


def test_U2b_KI_067_rerun_deletes_the_previous_run(make_ui: Any) -> None:
    """§24.7: the previous run's files do not outlive the session's access to them."""
    ui = make_ui()
    ui.run()
    ui.at.button(key="sample_land_cover").click().run()
    ui.at.radio(key=sk.LLM_MODE).set_value("off").run()
    ui.at.slider(key=sk.BUDGET).set_value(600).run()
    _start_and_finish(ui)
    first = ui.at.session_state[sk.RUN_ID]
    assert ui.manager.run_dir(first).exists()
    assert "deleted" in ui.at.button(key="rerun").help
    ui.at.button(key="rerun").click().run()
    assert not ui.manager.run_dir(first).exists()
    assert ui.page() == "Configure"


def test_U2b_KI_068_results_page_shows_captions_and_explained_warnings(make_ui: Any) -> None:
    """The Results page itself (not only the helpers) carries key numbers and explanations."""
    ui = make_ui()
    ui.at.session_state[sk.PAGE] = "Results"
    ui.at.session_state[sk.VIEW_SAMPLE] = "passengers"
    ui.run()
    assert not ui.at.exception
    captions = [img.proto.imgs[0].caption for img in ui.at.get("image")]
    assert any("top features:" in c for c in captions)
    assert any("cross-validated" in c for c in captions)
    assert all(c == img.proto.imgs[0].alt for c, img in zip(captions, ui.at.get("image"),
                                                            strict=True))  # fmt: skip
    assert "W-HIGH-MISSING — many missing values (cabin_deck)" in ui.texts()


def _call_rerun(run_id: str) -> None:
    import streamlit as st  # noqa: PLC0415 - AppTest runs this source alone

    from tabpilot.ui import state as keys  # noqa: PLC0415
    from tabpilot.ui.rerun import start_rerun  # noqa: PLC0415

    st.session_state[keys.RUN_ID] = run_id
    start_rerun()
    st.text(f"run_id={st.session_state.get(keys.RUN_ID)} page={st.session_state.get(keys.PAGE)}")


def test_U2b_KI_065_start_rerun_never_drops_a_running_run(make_ui: Any) -> None:
    """Defense in depth: even if called, the re-run keeps (and does not delete) a running run."""
    ui = make_ui(worker=polite)
    _configure_basic(ui)
    ui.at.button(key="start").click().run()
    run_id = ui.at.session_state[sk.RUN_ID]
    probe = AppTest.from_function(_call_rerun, args=(run_id,))
    probe.run()
    assert not probe.exception
    assert [t.value for t in probe.text] == [f"run_id={run_id} page=Run"]
    assert ui.manager.status(run_id).state == "running"
    ui.manager.cancel(run_id)
