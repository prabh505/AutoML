"""§3.2 stage 7 "Re-run with changes" (KI-056) and the active-run guard (KI-057).

A finished run's stored file is deleted after export (§24.7), and the run id cannot be started
again. A re-run therefore gets a new upload id while every choice is kept: a sample is reloaded
at once, and the user's own file is asked for again (it is not kept, by design). The target is
re-validated on the new file.
"""

from __future__ import annotations

import contextlib
from typing import Any

import streamlit as st

from tabpilot.ui import errors, services
from tabpilot.ui import state as sk

RERUN_FILE_TEXT = (
    "Upload the file again (the same one or a changed one): your settings are kept. The "
    "previous run's files were deleted to protect your data."
)
RERUN_SAMPLE_TEXT = (
    "Sample reloaded with your settings; the previous run's files were deleted. Change what you "
    "like, then start again."
)
RERUN_HELP = ("Keeps your settings and loads the data again. The previous run's files (report, "
              "bundle) are deleted: download what you need first.")  # fmt: skip
BUSY_TEXT = ("An analysis is running. Wait for it to finish or cancel it on the Run page before "
             "choosing new data.")  # fmt: skip


def active_run(state: Any) -> str | None:
    """The session's run id while that run is still running."""
    run_id = state.get(sk.RUN_ID)
    if not run_id:
        return None
    try:
        return run_id if services.manager().status(run_id).state == "running" else None
    except KeyError:
        return None


def start_rerun() -> None:
    """Callback: keep the choices, delete the finished run (KI-067: its files would otherwise
    become unreachable for "Delete my data now"), and load the data again. A running run is
    never dropped (KI-065): the user is sent to the Run page instead.
    """
    state = st.session_state
    if active_run(state):
        sk.go(state, "Run")
        return
    old = state.get(sk.RUN_ID)
    if old:
        with contextlib.suppress(KeyError):  # already gone (deleted or swept)
            services.manager().delete(old)
    sample_id = state.get(sk.SAMPLE_ID)
    for key in (sk.RUN_ID, sk.UPLOAD_ID, sk.DISPLAY_NAME, sk.WAITING, sk.ERROR, sk.INSPECT,
                sk.TARGET_INFO):  # fmt: skip
        state.pop(key, None)
    state[sk.RERUN] = True
    if not sample_id:
        state[sk.NOTICE] = {"kind": "info", "text": RERUN_FILE_TEXT}
        sk.go(state, "Upload")
        return
    try:
        upload = services.manager().validate_sample(sample_id)
    except Exception as exc:
        errors.remember(state, exc)
        sk.go(state, "Upload")
        return
    state.pop(sk.RERUN, None)
    state[sk.UPLOAD_ID] = upload.run_id
    state[sk.DISPLAY_NAME] = upload.display_name
    state[sk.NOTICE] = {"kind": "info", "text": RERUN_SAMPLE_TEXT}
    sk.go(state, "Configure")
