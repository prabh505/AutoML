"""Target selector, task confirmation, metric selector and group-column selector
(PRD §27.2, §3.2 stage 3, FR-TGT-01…03, US-03, US-05).

Selecting a target validates it and detects the task (`RunManager.check_target`). An ambiguous
detection shows radio buttons with no default, and the run cannot start until one is clicked
(FR-TGT-02, T-UI-02). Changing the task re-validates the target and resets the metric, so the
metric options always belong to the confirmed task (US-03).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from tabpilot.runtime.catalog import metric_direction, metric_options
from tabpilot.schemas import ColumnRole, DatasetMetadata, MetricName, SemanticType, Task
from tabpilot.ui import errors, services
from tabpilot.ui import state as sk
from tabpilot.ui.text import md_escape

TARGET_TYPES = (SemanticType.NUMERIC, SemanticType.CATEGORICAL, SemanticType.BOOLEAN)
NO_GROUP = "(none)"


def target_options(meta: DatasetMetadata, overrides: dict[str, dict[str, Any]]) -> list[str]:
    """Original names of non-excluded columns whose (overridden) type can be a target."""
    out = []
    for col in meta.columns:
        o = overrides.get(col.name, {})
        role = o.get("role") or col.role.value
        stype = o.get("semantic_type") or col.semantic_type.value
        if role != ColumnRole.EXCLUDED.value and SemanticType(stype) in TARGET_TYPES:
            out.append(col.original)
    return out


def unset(key: str) -> dict[str, Any]:
    """`index=None` (no default selection) unless Session State already holds the value; passing
    both makes Streamlit warn about a widget with two sources of truth.
    """
    return {} if key in st.session_state else {"index": None}


def _info(choice: Any) -> dict[str, Any]:
    d = choice.detection
    return {
        "column": choice.column, "detected": d.task.value, "confidence": d.confidence,
        "reason": d.reason, "alternatives": [t.value for t in d.alternatives],
        "task": choice.task.value, "positive_label": choice.positive_label,
        "positive_share": choice.positive_share, "n_missing": choice.report.n_missing,
        "n_removed": choice.report.n_rows_removed,
        "requires_confirmation": choice.report.requires_confirmation,
    }  # fmt: skip


def _check(task: Task | None) -> None:
    """Validate the selected target under `task` and store primitives only."""
    state = st.session_state
    target = state.get(sk.TARGET)
    for key in (sk.METRIC, sk.CONFIRM_MISSING, sk.INSPECT):
        state.pop(key, None)
    if not target:
        state.pop(sk.TARGET_INFO, None)
        return
    try:
        choice = services.manager().check_target(state[sk.UPLOAD_ID], target, task)
    except Exception as exc:  # E-TARGET-* and friends are shown inline
        info = dict(state.get(sk.TARGET_INFO) or {})
        state[sk.TARGET_INFO] = {**info, "error": errors.describe(exc), "task": None}
        return
    info = _info(choice)
    if task is None:
        state.pop(sk.TASK, None)
    state[sk.TARGET_INFO] = info


def on_target_change() -> None:
    _check(None)


def on_task_change() -> None:
    choice = st.session_state.get(sk.TASK)
    _check(Task(choice) if choice else None)


def confirmed_task(state: Any) -> Task | None:
    """The task the run will use: the click for ambiguous detections, else the detection."""
    info = state.get(sk.TARGET_INFO) or {}
    if not info.get("task") or info.get("error"):
        return None
    choice = state.get(sk.TASK)
    if choice:
        return Task(choice)
    return None if info.get("confidence") == "ambiguous" else Task(info["task"])


def target_section(meta: DatasetMetadata, default: str | None) -> None:
    state = st.session_state
    options = target_options(meta, state.get(sk.OVERRIDES, {}))
    if state.get(sk.TARGET) is not None and state[sk.TARGET] not in options:
        for key in (sk.TARGET, sk.TARGET_INFO, sk.TASK, sk.METRIC):
            state.pop(key, None)  # the column was excluded by an override
    if sk.TARGET not in state and default in options:
        state[sk.TARGET] = default
        _check(None)
    st.selectbox("Target column (what to predict)", options, **unset(sk.TARGET),
                 placeholder="Choose a column…", key=sk.TARGET,
                 on_change=on_target_change)  # fmt: skip
    info = state.get(sk.TARGET_INFO) or {}
    if info.get("error"):
        errors.show(info["error"])
    if not info.get("detected"):
        return
    st.markdown(f"Detected task: **{info['detected']}** ({info['confidence']} confidence) — "
                f"{md_escape(info['reason'])}")  # fmt: skip
    choices = [info["detected"], *info["alternatives"]]
    if info["confidence"] == "ambiguous":
        st.radio("This could be either. Choose the task to continue:", choices, **unset(sk.TASK),
                 key=sk.TASK, on_change=on_task_change, horizontal=True)  # fmt: skip
        if state.get(sk.TASK) is None:
            st.info("Choose a task above; the run cannot start until you do.")
    elif len(choices) > 1:
        st.radio("Task", choices, key=sk.TASK, on_change=on_task_change, horizontal=True)
    if info.get("positive_label") is not None:
        st.caption(f"Positive class: {md_escape(info['positive_label'])}")
    if info.get("n_missing"):
        st.caption(f"{info['n_missing']:,} rows have no target value and will be removed.")
    if info.get("requires_confirmation"):
        st.checkbox("I understand that more than half of the rows have no target value and "
                    "will be removed.", key=sk.CONFIRM_MISSING)  # fmt: skip


def metric_section() -> None:
    state = st.session_state
    task = confirmed_task(state)
    if task is None:
        return
    info = state[sk.TARGET_INFO]
    options, default = metric_options(task, info.get("positive_share"))
    values = [m.value for m in options]
    if state.get(sk.METRIC) not in values:
        state[sk.METRIC] = default.value
    st.selectbox("Primary metric (used to pick the winner)", values, key=sk.METRIC,
                 format_func=lambda m: f"{m} ({metric_direction(MetricName(m))})",
                 help="Recommended first. Fixed for the whole run.")  # fmt: skip


def group_section(meta: DatasetMetadata) -> None:
    state = st.session_state
    target = (state.get(sk.TARGET_INFO) or {}).get("column")
    names = [NO_GROUP, *[c.name for c in meta.columns if c.name != target]]
    labels = {c.name: c.original for c in meta.columns}
    if state.get(sk.GROUP) not in names:
        state[sk.GROUP] = NO_GROUP
    st.selectbox("Group column (optional)", names, key=sk.GROUP,
                 format_func=lambda n: labels.get(n, n),
                 help="Rows with the same value (e.g. one customer) never span training and "
                      "validation folds.")  # fmt: skip
