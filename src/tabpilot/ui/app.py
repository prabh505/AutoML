"""Application shell (PRD §27.1): page config, sidebar navigation, and page dispatch.

Navigation is a sidebar radio bound to the `page` session key, so pages can also switch
programmatically (e.g. Start → Run → Results). `assert_primitive` guards the session-state
invariant on every rerun: only the run id and UI choices are stored.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from tabpilot.ui import state as sk
from tabpilot.ui.pages import about, configure, results, run, upload
from tabpilot.ui.text import md_escape

TITLE = "TabPilot"
PAGES: dict[str, Callable[[], None]] = {
    "Upload": upload.render,
    "Configure": configure.render,
    "Run": run.render,
    "Results": results.render,
    "About & Privacy": about.render,
}


def notice() -> None:
    """One-shot notices (e.g. "Your data has been deleted.") shown once, then cleared."""
    item = st.session_state.pop(sk.NOTICE, None)
    if item:
        show = {"success": st.success, "warning": st.warning}.get(item["kind"], st.info)
        show(md_escape(item["text"]))


def _nav() -> None:
    st.session_state[sk.PAGE] = st.session_state[sk.NAV]


def main() -> None:
    """Render the current page (PRD §27)."""
    st.set_page_config(page_title=TITLE, page_icon=":material/insights:", layout="centered")
    state = st.session_state
    state.setdefault(sk.PAGE, "Upload")
    sk.keep_choices(state)
    state[sk.NAV] = state[sk.PAGE]
    with st.sidebar:
        st.markdown(f"**{TITLE}**")
        st.radio("Page", list(sk.PAGES), key=sk.NAV, on_change=_nav)
        st.caption("LLM-assisted AutoML for tables")
    notice()
    PAGES[state[sk.PAGE]]()
    sk.assert_primitive(state)
