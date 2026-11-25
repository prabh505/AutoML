"""Page 4 — Results (PRD §27.1, §3.2 stages 7–9, §27.3 Success/Partial, D-14).

Tabs: Summary · Data profile · Plan · Leaderboard · Explainability · Report · Downloads. The page
shows the user's finished run, or a precomputed sample result labeled "precomputed on {date}"
(D-14) when the user asked for one while waiting for the slot.
"""

from __future__ import annotations

import streamlit as st

from tabpilot.runtime import RunArtifacts, UnknownRunError
from tabpilot.schemas import RunResult, RunWarning
from tabpilot.ui import errors, precomputed, services
from tabpilot.ui import state as sk
from tabpilot.ui.components import leaderboard, outputs
from tabpilot.ui.components.columns import effective
from tabpilot.ui.rerun import RERUN_HELP, start_rerun
from tabpilot.ui.text import duration, md_escape

TABS = ("Summary", "Data profile", "Plan", "Leaderboard", "Explainability", "Report",
        "Downloads")  # fmt: skip
DELETED_TEXT = "Your data has been deleted."


def banner(result: RunResult) -> None:
    """§27.3 Success / Partial banners."""
    if result.status == "completed":
        st.success(f"Completed in {duration(result.duration_s)}", icon=":material/check_circle:")
    elif result.status == "completed_partial":
        stage = result.stopped_after_stage.value if result.stopped_after_stage else "a stage"
        st.warning(f"Stopped at time budget after {stage}; results are valid but limited",
                   icon=":material/timer:")  # fmt: skip


def _delete() -> None:
    state = st.session_state
    run_id = state.get(sk.RUN_ID)
    if run_id:
        services.manager().delete(run_id)
    sk.reset_choices(state)
    for key in (sk.UPLOAD_ID, sk.DISPLAY_NAME, sk.SAMPLE_ID):
        state.pop(key, None)
    state[sk.NOTICE] = {"kind": "success", "text": DELETED_TEXT}
    sk.go(state, "Upload")


def _back_from_sample() -> None:
    st.session_state.pop(sk.VIEW_SAMPLE, None)
    sk.go(st.session_state, "Configure")


def tabs(art: RunArtifacts, warnings: list[RunWarning], columns: list[dict[str, object]] | None,
         deletable: bool) -> None:  # fmt: skip
    t = st.tabs(list(TABS))
    with t[0]:
        outputs.summary(art, warnings)
    with t[1]:
        outputs.data_profile(art, columns)
    with t[2]:
        outputs.plan(art)
    with t[3]:
        leaderboard.render(art.leaderboard, art.selected_entry_id, art.selection)
    with t[4]:
        outputs.explainability(art)
    with t[5]:
        outputs.report(art)
    with t[6]:
        outputs.downloads(art, _delete if deletable else None)


def sample_view(sample_id: str) -> None:
    art = precomputed.load(sample_id)
    if art is None:
        st.info("No precomputed result is available for this sample.")
        return
    info = art.precomputed or {}
    st.info(f"Sample result precomputed on {md_escape(info.get('date', 'unknown date'))} with "
            f"the {md_escape(info.get('llm_mode', 'off'))} LLM mode — not your data.",
            icon=":material/history:")  # fmt: skip
    st.button("Back to my run settings", on_click=_back_from_sample, key="back_sample")
    tabs(art, [], None, deletable=False)


def render() -> None:
    state = st.session_state
    st.header("Results")
    if state.get(sk.VIEW_SAMPLE):
        sample_view(state[sk.VIEW_SAMPLE])
        return
    run_id = state.get(sk.RUN_ID)
    if not run_id:
        st.info("No results yet. Run an analysis first.")
        return
    manager = services.manager()
    try:
        result = manager.result(run_id)
        art = manager.artifacts(run_id)
    except UnknownRunError:
        st.info("This run is no longer on the server.")
        return
    if result is None:
        st.info("The analysis is still running.")
        st.button("Go to the Run page", on_click=sk.go, args=(state, "Run"))
        return
    if result.status in ("failed", "cancelled"):
        errors.show({"code": result.error_code.value if result.error_code else None,
                     "message": result.error_message or "The run was cancelled."})  # fmt: skip
        return
    banner(result)
    if art is None:
        st.info(DELETED_TEXT)
        return
    columns = None
    try:
        upload = manager.upload_info(run_id)
        columns = effective(upload.metadata, state.get(sk.OVERRIDES, {}))
    except UnknownRunError:
        pass
    st.button("Re-run with changes", on_click=start_rerun, key="rerun", help=RERUN_HELP)
    tabs(art, list(result.warnings), columns, deletable=True)
