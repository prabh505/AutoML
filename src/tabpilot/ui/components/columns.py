"""Column table with type/role overrides and the data-quality warnings panel
(PRD §27.1 Configure, §3.2 stage 2, FR-ING-04, US-04).

Every excluded column is listed with its reason (FR-ING-04 AC). Type overrides are checked by
`RunManager.check_override` (≥ 95% parse) and rejected inline; re-including an excluded column is
allowed only when its effective type can be modeled ("except unsupported types", §3.2).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from tabpilot.schemas import ColumnRole, DatasetMetadata, SemanticType
from tabpilot.ui import services
from tabpilot.ui import state as sk
from tabpilot.ui.text import md_escape

MODELABLE = (SemanticType.NUMERIC, SemanticType.CATEGORICAL, SemanticType.BOOLEAN,
             SemanticType.DATETIME)  # fmt: skip
KEEP = "(keep)"
REASONS = {
    "constant": "same value in every row",
    "free_text": "free text (not modeled in this version)",
    "missing_rate_gt_95": "more than 95% missing",
    "user_excluded": "excluded by you",
    "leakage_near_deterministic": "predicts the target almost perfectly (possible leak)",
}
WARNINGS = {
    "W-HIGH-MISSING": "many missing values",
    "W-NEAR-CONSTANT": "almost every row has the same value",
    "W-HIGH-CARDINALITY": "very many distinct values",
    "W-COERCED": "values that are not numbers were treated as missing",
    "W-INF": "infinite values were treated as missing",
    "W-DUPLICATES": "duplicate rows",
    "W-ENCODING-FALLBACK": "the text encoding was guessed; check special characters",
    "W-TARGET-MISSING": "rows without a target value will be removed",
    "W-WIDE": "many columns compared with rows",
    "W-SMALL-DATA": "small dataset: results vary more",
}


def reason_text(reason: str | None) -> str:
    """Plain-language exclusion reason (identifier reasons carry their unique ratio)."""
    if reason is None:
        return ""
    if reason.startswith("identifier"):
        return "looks like an ID (almost every value is unique)"
    return REASONS.get(reason, reason.replace("_", " "))


def warning_text(raw: str) -> str:
    """`W-CODE:column:count` → `W-CODE — explanation (column, count)`."""
    code, *rest = raw.split(":")
    detail = ", ".join(rest)
    meaning = WARNINGS.get(code, "see the report for details")
    return f"{code} — {meaning}" + (f" ({detail})" if detail else "")


def effective(meta: DatasetMetadata, overrides: dict[str, dict[str, Any]]
              ) -> list[dict[str, Any]]:  # fmt: skip
    """Column rows after the user's overrides (what the run will use)."""
    rows = []
    for col in meta.columns:
        o = overrides.get(col.name, {})
        stype = o.get("semantic_type") or col.semantic_type.value
        role = o.get("role") or col.role.value
        reason = (
            "user_excluded"
            if o.get("role") == ColumnRole.EXCLUDED.value
            else (None if o.get("role") == ColumnRole.FEATURE.value else col.exclusion_reason)
        )
        rows.append({"Column": col.original, "Name used": col.name, "Type": stype, "Role": role,
                     "Missing %": None if col.missing_rate is None
                     else round(100 * col.missing_rate, 1),
                     "Excluded because": reason_text(reason),
                     "Overridden": "yes" if o else ""})  # fmt: skip
    return rows


def column_table(meta: DatasetMetadata) -> None:
    overrides: dict[str, dict[str, Any]] = st.session_state.get(sk.OVERRIDES, {})
    st.dataframe(effective(meta, overrides), hide_index=True)


def warnings_panel(meta: DatasetMetadata) -> None:
    if not meta.warnings and not meta.row_removals:
        st.caption("No data-quality warnings.")
        return
    lines = [warning_text(w) for w in meta.warnings]
    lines += [f"{r.count} rows removed: {r.reason}" for r in meta.row_removals]
    st.warning("Data-quality warnings:\n\n" + "\n".join(f"- {md_escape(x)}" for x in lines),
               icon=":material/warning:")  # fmt: skip


def _apply(upload_id: str, meta: DatasetMetadata) -> None:
    """on_click: validate and store one override (FR-ING-04 Val/Err)."""
    state = st.session_state
    name = state.get("ov_column")
    new_type = state.get("ov_type", KEEP)
    new_role = state.get("ov_role", KEEP)
    col = next((c for c in meta.columns if c.name == name), None)
    state.pop("ov_error", None)
    if col is None or (new_type == KEEP and new_role == KEEP):
        return
    stype = col.semantic_type if new_type == KEEP else SemanticType(new_type)
    if new_type != KEEP:
        problem = services.manager().check_override(upload_id, col.name, stype)
        if problem:
            state["ov_error"] = problem
            return
    if new_role == ColumnRole.FEATURE.value and stype not in MODELABLE:
        state["ov_error"] = (f"Column '{col.original}' has type {stype.value}, which cannot be "
                             "modeled; change its type first.")  # fmt: skip
        return
    overrides = dict(state.get(sk.OVERRIDES, {}))
    overrides[col.name] = {"semantic_type": None if new_type == KEEP else new_type,
                           "role": None if new_role == KEEP else new_role}  # fmt: skip
    state[sk.OVERRIDES] = overrides
    state.pop(sk.INSPECT, None)


def _remove(column: str) -> None:
    overrides = dict(st.session_state.get(sk.OVERRIDES, {}))
    overrides.pop(column, None)
    st.session_state[sk.OVERRIDES] = overrides
    st.session_state.pop(sk.INSPECT, None)


def override_form(upload_id: str, meta: DatasetMetadata) -> None:
    """Change one column's type or role; current overrides can be removed."""
    state = st.session_state
    target = (state.get(sk.TARGET_INFO) or {}).get("column")
    names = [c.name for c in meta.columns if c.name != target]
    labels = {c.name: c.original for c in meta.columns}
    with st.expander("Change a column's type or role"):
        st.selectbox("Column", names, format_func=lambda n: labels[n], key="ov_column")
        st.selectbox("New type", [KEEP, *[t.value for t in MODELABLE]], key="ov_type")
        st.selectbox("New role", [KEEP, ColumnRole.FEATURE.value, ColumnRole.EXCLUDED.value],
                     key="ov_role",
                     help="feature: use the column; excluded: leave it out")  # fmt: skip
        st.button("Apply override", on_click=_apply, args=(upload_id, meta))
        if state.get("ov_error"):
            st.error(md_escape(state["ov_error"]))
        for name, o in state.get(sk.OVERRIDES, {}).items():
            what = ", ".join(f"{k.replace('semantic_type', 'type')} → {v}"
                             for k, v in o.items() if v)  # fmt: skip
            cols = st.columns([4, 1])
            cols[0].text(f"{labels.get(name, name)}: {what}")
            cols[1].button("Remove", key=f"ov_remove_{name}", on_click=_remove, args=(name,))
