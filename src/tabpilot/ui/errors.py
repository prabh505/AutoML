"""Error display (PRD §8.9, NFR-UX-01, §27.3 Failure): every failure shows its E-code message
(what, where, what to do) and never a stack trace or a file-system path.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from tabpilot.logging import get_logger
from tabpilot.runtime import InvalidConfigError, SlotBusyError, UnknownRunError
from tabpilot.schemas import TabPilotError
from tabpilot.ui import state as sk
from tabpilot.ui.text import md_escape

log = get_logger("ui")

GENERIC = (
    "Something went wrong while handling your request. Nothing was lost; try again, or reload "
    "the page. If it keeps happening, run TabPilot locally (see the README)."
)


def describe(exc: BaseException) -> dict[str, str | None]:
    """`{"code", "message"}` for any exception; unknown errors get a generic message."""
    if isinstance(exc, TabPilotError):
        return {"code": exc.code.value, "message": exc.user_message}
    if isinstance(exc, SlotBusyError):
        return {"code": None, "message": str(exc)}
    if isinstance(exc, InvalidConfigError | ValueError):
        return {"code": None, "message": f"This setting cannot be used: {exc}"}
    if isinstance(exc, UnknownRunError):
        return {"code": None, "message": "This run no longer exists (its data was deleted)."}
    log.warning("ui_error", error_class=type(exc).__name__)  # class only, never the message (I-11)
    return {"code": None, "message": GENERIC}


def remember(state: Any, exc: BaseException) -> None:
    state[sk.ERROR] = describe(exc)


def show(error: dict[str, Any] | None) -> None:
    """Render an error panel: code in bold, message escaped (it may name a user's column)."""
    if not error:
        return
    code = error.get("code")
    head = f"**{md_escape(code)}** — " if code else ""
    st.error(head + md_escape(error.get("message", "")), icon=":material/error:")
