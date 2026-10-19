"""Streamlit pages and components (PRD §27).

Depends on runtime, schemas.

Must not: import orchestration internals or ML modules; render user-derived strings as HTML.
"""

from tabpilot.ui.app import main

__all__ = ["main"]
