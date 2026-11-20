"""Event log (PRD §27.2, §27.4, US-07, T-UI-03).

An append-only list of `RunEvent`s. System events are plain text lines written after the action
they describe. `llm_rationale` events appear in a separate callout, labeled "LLM-generated", as
quoted text. They are never shown as completed actions. Event data is shown with `st.text`, so a
column name is never interpreted as Markdown or HTML.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from tabpilot.schemas import EventType, RunEvent
from tabpilot.ui import state as sk
from tabpilot.ui.text import quote

MAX_LINES = 300
LLM_LABEL = "LLM-generated rationale — the model's reasoning, not an action TabPilot performed"
HIDDEN = {"text"}  # rationale text is shown only in its labeled callout


def line(event: RunEvent) -> str:
    """`hh:mm:ss  stage · type  key=value …` for a system event."""
    details = " ".join(f"{k}={v}" for k, v in sorted(event.data.items()) if k not in HIDDEN
                       and not isinstance(v, dict | list))  # fmt: skip
    return f"{event.ts:%H:%M:%S}  {event.stage.value} · {event.type.value}  {details}".rstrip()


def rationale(event: RunEvent) -> None:
    """A labeled, quoted LLM rationale (§27.4)."""
    source = str(event.data.get("source", "LLM"))
    st.info(f"**{LLM_LABEL}** ({source})\n\n{quote(event.data.get('text', ''))}",
            icon=":material/smart_toy:")  # fmt: skip


def render(events: Sequence[RunEvent]) -> None:
    st.markdown("**Event log**")
    st.session_state.setdefault(sk.NEWEST_FIRST, True)
    newest_first = st.toggle("Newest first", key=sk.NEWEST_FIRST)
    system = [e for e in events if e.origin == "system"]
    llm = [e for e in events if e.origin == "llm_rationale" or e.type == EventType.LLM_RATIONALE]
    for event in llm:
        rationale(event)
    lines = [line(e) for e in system][-MAX_LINES:]
    if newest_first:
        lines.reverse()
    with st.container(height=320):
        st.text("\n".join(lines) if lines else "Waiting for the first event…")
