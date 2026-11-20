"""Session-state keys and helpers (P16: "st.session_state stores only run_id and UI choices").

Values are plain Python primitives (str, int, float, bool, None, and lists/dicts of them); no
DataFrame, upload bytes or model is ever stored. `assert_primitive` is used by the tests.
"""

from __future__ import annotations

from typing import Any

PAGES = ("Upload", "Configure", "Run", "Results", "About & Privacy")

# Navigation: `PAGE` is the current page; `NAV` is the sidebar widget, synced from `PAGE`
# before it is drawn, so code can switch pages after the widget exists.
PAGE = "page"
NAV = "nav"
# Run identity
UPLOAD_ID = "upload_id"  # validated upload; becomes the run id when started
DISPLAY_NAME = "display_name"
SAMPLE_ID = "sample_id"  # the sample the upload came from, if any
RUN_ID = "run_id"  # started run
# Configure choices
TARGET = "target"
TARGET_INFO = "target_info"  # primitives describing the validated target
TASK = "task_choice"
METRIC = "metric"
GROUP = "group_column"
BUDGET = "budget_s"
LLM_MODE = "llm_mode"
PROVIDERS = "providers"
CONSENTED = "consented_providers"  # per session, per provider (§24.5)
OVERRIDES = "overrides"  # {column: {"semantic_type": str|None, "role": str|None}}
CONFIRM_MISSING = "confirm_missing_target"
INSPECT = "inspect_planner_input"
MINIMAL = "minimal_profile"  # §24.4 minimal compact profile
# Flow flags
WAITING = "waiting_for_slot"
VIEW_SAMPLE = "view_sample_result"
NOTICE = "notice"  # {"kind": "success"|"info"|"warning", "text": str}
ERROR = "error"  # {"code": str|None, "message": str}
UPLOAD_TOKEN = "upload_token"  # noqa: S105 - not a secret; id of the uploader file already validated
NEWEST_FIRST = "newest_first"

# Widget-backed choices that must survive page changes (Streamlit drops the state of widgets
# that were not drawn in a run unless the key is re-assigned).
PERSISTENT = (TARGET, TASK, METRIC, GROUP, BUDGET, LLM_MODE, PROVIDERS, MINIMAL, CONFIRM_MISSING,
              NEWEST_FIRST)  # fmt: skip

RUN_CHOICES = (TARGET, TARGET_INFO, TASK, METRIC, GROUP, OVERRIDES, CONFIRM_MISSING, INSPECT)

PRIMITIVES = (str, int, float, bool, type(None))


def is_primitive(value: Any) -> bool:
    """True for JSON-like primitives and lists/tuples/dicts of them."""
    if isinstance(value, PRIMITIVES):
        return True
    if isinstance(value, list | tuple):
        return all(is_primitive(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_primitive(v) for k, v in value.items())
    return False


def assert_primitive(state: Any) -> None:
    """Raise if any session-state entry holds something other than primitives."""
    bad = [k for k, v in state.items() if not is_primitive(v)]
    if bad:
        raise TypeError(f"session state holds non-primitive values under {sorted(bad)}")


def reset_choices(state: Any) -> None:
    """Forget the per-dataset choices (new upload or new sample)."""
    for key in (*RUN_CHOICES, RUN_ID, WAITING, ERROR, VIEW_SAMPLE):
        state.pop(key, None)


def keep_choices(state: Any) -> None:
    """Re-assign the persistent widget keys so their values outlive pages that hide them."""
    for key in PERSISTENT:
        if key in state:
            state[key] = state[key]


def go(state: Any, page: str) -> None:
    """Switch page; takes effect on the next rerun (callbacks run before the next draw)."""
    if page not in PAGES:
        raise ValueError(f"unknown page {page!r}")
    state[PAGE] = page
