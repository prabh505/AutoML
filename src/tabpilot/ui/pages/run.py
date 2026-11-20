"""Page 3 — Run (PRD §27.1, §3.2 stage 6, §27.3 Running/Failure/Cancelled, §27.4, US-07).

Live view polled once per second with `st.fragment(run_every=1)` while the run is active; the
fragment is created without `run_every` once the run has ended, so polling stops (§27.5).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from tabpilot.runtime import UnknownRunError
from tabpilot.schemas import EventType, RunEvent, RunResult
from tabpilot.ui import errors, services
from tabpilot.ui import state as sk
from tabpilot.ui.components import events as event_log
from tabpilot.ui.components.status import STEPS, meters, stepper, steps

POLL_S = 1.0
CANCELLED_TEXT = "Run cancelled. Data deleted."


def partial_leaderboard(events: Sequence[RunEvent]) -> list[dict[str, Any]]:
    """Candidates finished so far, from `candidate_completed` events (emitted with the scores)."""
    rows = []
    for e in events:
        if e.type in (EventType.CANDIDATE_COMPLETED, EventType.CANDIDATE_FAILED):
            mean = e.data.get("mean")
            rows.append({"Model": e.data.get("model"), "Entry": e.data.get("entry_id"),
                         "Status": e.data.get("status"),
                         "Mean score": round(mean, 4) if isinstance(mean, float) else None,
                         "Trials": e.data.get("trials")})  # fmt: skip
    return rows


def failure_panel(result: RunResult, events: Sequence[RunEvent]) -> None:
    """§27.3 Failure: E-code message, what was completed, and the action (in the message)."""
    errors.show({"code": result.error_code.value if result.error_code else None,
                 "message": result.error_message or errors.GENERIC})  # fmt: skip
    done = [s.label for s in steps(events, finished=True) if s.status == "done"]
    st.write("Completed before the failure: " + (", ".join(done) if done else "nothing") + ".")
    st.caption("The uploaded data of a failed run is deleted automatically.")


def _cancel(run_id: str) -> None:
    services.manager().cancel(run_id)


def _new_run() -> None:
    state = st.session_state
    sk.reset_choices(state)
    for key in (sk.UPLOAD_ID, sk.DISPLAY_NAME, sk.SAMPLE_ID):
        state.pop(key, None)
    sk.go(state, "Upload")


def live(run_id: str, polling: bool) -> None:
    """One poll: status, stepper, meters, partial leaderboard, event log, Cancel. When a polling
    fragment sees the run end, the whole page reruns so the next fragment stops polling.
    """
    manager = services.manager()
    settings = manager.settings
    status = manager.status(run_id)
    events = manager.events(run_id)
    running = status.state == "running"
    if not running and status.state in ("completed", "completed_partial"):
        sk.go(st.session_state, "Results")
        st.rerun(scope="app")
    if not running and polling:
        st.rerun(scope="app")
    budget = manager.run_config(run_id).time_budget_s
    if running:
        st.button("Cancel run", key="cancel", on_click=_cancel, args=(run_id,),
                  help="Stops the analysis and deletes its data.")  # fmt: skip
        meters(events, status.elapsed_s or 0.0, budget, settings.limits.max_trials_per_run,
               settings.limits.llm_calls_per_run)  # fmt: skip
    result = manager.result(run_id)
    if result is not None and result.status == "cancelled":
        st.info(CANCELLED_TEXT, icon=":material/cancel:")
    elif result is not None and result.status == "failed":
        failure_panel(result, events)
    stepper(events, finished=not running)
    board = partial_leaderboard(events)
    if board:
        st.markdown("**Models finished so far**")
        st.dataframe(board, hide_index=True)
    event_log.render(events)
    if not running:
        st.button("Start a new analysis", on_click=_new_run, key="new_run")


def render() -> None:
    state = st.session_state
    st.header("Run")
    run_id = state.get(sk.RUN_ID)
    if not run_id:
        st.info("No analysis is running. Configure one first.")
        return
    try:
        running = services.manager().status(run_id).state == "running"
    except UnknownRunError:
        st.info("This run is no longer on the server.")
        return
    st.caption(f"{len(STEPS)} stages · updates every second while running")
    st.fragment(live, run_every=POLL_S if running else None)(run_id, running)
