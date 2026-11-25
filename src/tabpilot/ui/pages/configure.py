"""Page 2 — Configure (PRD §27.1, §3.2 stages 2–5, §27.3 Waiting for slot).

Preview (first 20 rows), column table with overrides, warnings, target/task/metric/group,
time budget with live feasibility, LLM mode with consent, the planner-input inspector, and the
Start button with the feasibility summary. A busy slot shows the wait state with Retry and
"View sample result" (D-14).
"""

from __future__ import annotations

import streamlit as st

from tabpilot.runtime import SlotBusyError, UnknownRunError
from tabpilot.runtime.catalog import samples
from tabpilot.schemas import RunConfig
from tabpilot.ui import errors, services
from tabpilot.ui import state as sk
from tabpilot.ui.components import columns, run_settings, target
from tabpilot.ui.components.status import slot_wait_text
from tabpilot.ui.config import draft
from tabpilot.ui.precomputed import available_samples
from tabpilot.ui.rerun import BUSY_TEXT, RERUN_HELP, active_run, start_rerun
from tabpilot.ui.text import md_escape


def _start(config: RunConfig) -> None:
    state = st.session_state
    manager = services.manager()
    try:
        manager.start(config, display_name=state.get(sk.DISPLAY_NAME),
                      options=services.worker_options)  # fmt: skip
    except SlotBusyError:
        state[sk.WAITING] = True
        st.rerun()
    except Exception as exc:
        errors.remember(state, exc)
        return
    state.pop(sk.WAITING, None)
    state.pop(sk.ERROR, None)
    state[sk.RUN_ID] = config.run_id
    sk.go(state, "Run")
    st.rerun()


def _delete_upload() -> None:
    state = st.session_state
    services.manager().discard_upload(state[sk.UPLOAD_ID])
    sk.reset_choices(state)
    for key in (sk.UPLOAD_ID, sk.DISPLAY_NAME, sk.SAMPLE_ID):
        state.pop(key, None)
    state[sk.NOTICE] = {"kind": "success", "text": "Your data has been deleted."}
    sk.go(state, "Upload")


def _view_sample(sample_id: str) -> None:
    st.session_state[sk.VIEW_SAMPLE] = sample_id
    sk.go(st.session_state, "Results")


def waiting_panel() -> None:
    """§27.3: "Another analysis is running (est. wait N min)" with Retry and a sample result."""
    notice = services.manager().slot_notice()
    st.warning(slot_wait_text(notice.est_wait_s if notice else None) + " Your settings are "
               "kept; press Retry to start when the slot is free.",
               icon=":material/hourglass_top:")  # fmt: skip
    ready = available_samples()
    if ready:
        current = st.session_state.get(sk.SAMPLE_ID)
        choice = current if current in ready else ready[0]
        titles = {s.sample_id: s.title for s in samples()}
        st.button(f"View sample result: {titles.get(choice, choice)}", on_click=_view_sample,
                  args=(choice,), key="view_sample")  # fmt: skip


def render() -> None:
    state = st.session_state
    upload_id = state.get(sk.UPLOAD_ID)
    st.header("Configure")
    if not upload_id:
        st.info("Upload a file or choose a sample first.")
        st.button("Go to Upload", on_click=sk.go, args=(state, "Upload"))
        return
    manager = services.manager()
    try:
        upload = manager.upload_info(upload_id)
    except UnknownRunError:
        sk.reset_choices(state)
        state.pop(sk.UPLOAD_ID, None)
        st.info("Your file is no longer on the server (it was deleted after the idle time). "
                "Please upload it again.")  # fmt: skip
        return
    if active_run(state):  # KI-065: a running analysis is never offered for a re-run
        st.warning(BUSY_TEXT, icon=":material/hourglass_top:")
        st.button("Go to the Run page", on_click=sk.go, args=(state, "Run"), key="to_run_cfg")
        return
    if not manager.can_start(upload_id):  # KI-057/KI-056: a started run's id is never reused
        st.info("This file was used by your last analysis and deleted afterwards. Use “Re-run "
                "with changes” to keep your settings and load the data again.")  # fmt: skip
        st.button("Re-run with changes", on_click=start_rerun, key="rerun_configure",
                  help=RERUN_HELP)  # fmt: skip
        return
    meta = upload.metadata
    st.markdown(f"**{md_escape(upload.display_name)}** — {meta.n_rows_raw:,} rows, "
                f"{meta.n_columns} columns")  # fmt: skip
    errors.show(state.get(sk.ERROR))
    with st.expander("Preview (first 20 rows)"):
        st.dataframe(manager.preview_rows(upload_id), hide_index=True)

    st.subheader("Columns")
    columns.column_table(meta)
    columns.warnings_panel(meta)
    columns.override_form(upload_id, meta)

    st.subheader("Target and task")
    sample_id = state.get(sk.SAMPLE_ID)
    default = next((s.target for s in samples() if s.sample_id == sample_id), None)
    target.target_section(meta, default)
    target.metric_section()
    target.group_section(meta)

    st.subheader("Run settings")
    run_settings.budget_slider()
    run_settings.llm_settings()

    start_section(upload_id)


def start_section(upload_id: str) -> None:
    """Feasibility summary, inspector, Start/Retry, wait state and "Delete my data now"."""
    state = st.session_state
    current = draft(state, services.settings(), run_settings.available_names())
    feasible = True
    if current.config is not None:
        try:
            feasible = run_settings.feasibility(
                services.manager().estimate(upload_id, current.config)
            )
        except Exception as exc:
            errors.show(errors.describe(exc))
            feasible = False
    run_settings.inspector(current.config)

    if current.missing:
        st.caption("To start: " + "; ".join(current.missing) + ".")
    ready = current.config is not None and feasible
    label = "Retry" if state.get(sk.WAITING) else "Start analysis"
    if st.button(label, type="primary", disabled=not ready, key="start") and current.config:
        _start(current.config)
    if state.get(sk.WAITING):
        waiting_panel()
    st.button("Delete my data now", on_click=_delete_upload, key="delete_upload",
              help="Deletes the uploaded file from the server right away.")  # fmt: skip
