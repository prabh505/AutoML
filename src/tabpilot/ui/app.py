"""Application shell rendered until the full UI exists (P01 deliverable 9; replaced in P16)."""

import streamlit as st

TITLE = "TabPilot"
STATUS = "TabPilot is being built — see README."


def main() -> None:
    """Render the app title and build status (PRD §27; full pages arrive in P16)."""
    st.set_page_config(page_title=TITLE)
    st.title(TITLE)
    st.write(STATUS)
