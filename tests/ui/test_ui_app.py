"""Streamlit AppTest tests for the P16 UI (PRD §27, §3.2, §31.3 T-UI-01…04, US-06).

Real `RunManager` and real runs (offline stub LLM, fast worker options) unless a test injects
one of the controlled workers from `tests/helpers/fake_workers.py`.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from tests.helpers.fake_workers import polite, quick_fail
from tests.ui.conftest import UI

from tabpilot.llm.cache import canonical
from tabpilot.llm.stub import StubProvider
from tabpilot.runtime.slots import acquire_slot
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
    options = ui.at.selectbox(key=sk.TARGET).options
    if "y" in options:  # a constant column is excluded at ingestion; if offered, it must fail
        ui.at.selectbox(key=sk.TARGET).set_value("y").run()
        assert "E-TARGET-CONSTANT" in ui.texts()
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
