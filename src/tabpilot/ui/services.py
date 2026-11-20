"""Process-wide services for the Streamlit app (PRD §28.1: the UI calls the `RunManager`
in-process).

One `RunManager` per server process (`st.cache_resource`), under `<data_dir>/runs`, with the
§24.7 sweeper started once. `TABPILOT_RUN_MODE=thread` selects the documented thread fallback
(D-02); the default is one spawned worker process per run.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from tabpilot.config import Settings, load_settings
from tabpilot.runtime import RunManager

RUN_MODE_ENV = "TABPILOT_RUN_MODE"
STUB_ENV = "TABPILOT_LLM_STUB"  # "1": offer the offline stub LLM (tests and demos without keys)

# Worker options for runs started from the UI (None: the defaults). Tests set a fast
# `WorkerOptions` here; the UI never imports orchestration itself (import contract R7).
worker_options: Any = None


def build_manager() -> RunManager:
    """Create the app's run manager and start the sweeper (on app start, §24.7)."""
    settings = load_settings()
    mode = os.environ.get(RUN_MODE_ENV, "process")
    manager = RunManager(settings.data_dir / "runs", settings, mode=mode)
    manager.start_sweeper()
    return manager


@st.cache_resource(show_spinner=False)
def manager() -> RunManager:
    """The shared run manager of this server process."""
    return build_manager()


def settings() -> Settings:
    return manager().settings


def stub_enabled() -> bool:
    return os.environ.get(STUB_ENV) == "1"
