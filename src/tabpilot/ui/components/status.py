"""Demo-status banner, stage stepper and budget meters (PRD §27.1 Upload/Run, §27.3).

State is always given in words and symbols, never by color alone (§27.5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import streamlit as st

from tabpilot.schemas import EventType, RunEvent, Stage
from tabpilot.ui import services
from tabpilot.ui.text import duration

# §27.1 Run page stepper: label → the stages it covers.
STEPS: tuple[tuple[str, tuple[Stage, ...]], ...] = (
    ("Ingest", (Stage.INGEST,)),
    ("Split", (Stage.SPLIT,)),
    ("Profile", (Stage.PROFILE,)),
    ("Plan", (Stage.PLAN, Stage.VALIDATE_PLAN)),
    ("Features", (Stage.BUILD_FEATURES,)),
    ("Train", (Stage.TRAIN_CANDIDATES, Stage.EVALUATE)),
    ("Reflect", (Stage.REFLECT, Stage.APPLY_CHANGES)),
    ("Ensemble", (Stage.ENSEMBLE,)),
    ("Finalize", (Stage.FINALIZE,)),
    ("Explain", (Stage.EXPLAIN,)),
    ("Report", (Stage.REPORT,)),
    ("Export", (Stage.EXPORT,)),
)
SYMBOLS = {"done": "✓ done", "running": "▶ running", "waiting": "○ waiting",
           "skipped": "– skipped", "failed": "✗ failed"}  # fmt: skip


def slot_wait_text(est_wait_s: float | None) -> str:
    """§27.3 waiting-for-slot text."""
    if est_wait_s is None:
        return "Another analysis is running (est. wait unknown)."
    return f"Another analysis is running (est. wait {max(1, round(est_wait_s / 60))} min)."


def demo_banner() -> None:
    """Upload page banner: shown only while every run slot is busy."""
    notice = services.manager().slot_notice()
    if notice is not None:
        st.warning(slot_wait_text(notice.est_wait_s) + " You can still prepare your run; "
                   "starting it waits for the slot.", icon=":material/hourglass_top:")  # fmt: skip


@dataclass(frozen=True)
class Step:
    label: str
    status: str
    seconds: float | None


def _seconds(event: RunEvent) -> float:
    value = event.data.get("seconds")
    return float(value) if isinstance(value, int | float) else 0.0


def steps(events: Sequence[RunEvent], *, finished: bool) -> list[Step]:
    """Each stepper step's status from the event stream (events follow completed actions)."""
    out = []
    for label, stages in STEPS:
        mine = [e for e in events if e.stage in stages]
        kinds = {e.type for e in mine}
        seconds = sum(_seconds(e) for e in mine if e.type == EventType.STAGE_COMPLETED)
        if EventType.STAGE_FAILED in kinds:
            status = "failed"
        elif EventType.STAGE_STARTED in kinds:
            started = sum(e.type == EventType.STAGE_STARTED for e in mine)
            done = sum(e.type == EventType.STAGE_COMPLETED for e in mine)
            status = "running" if started > done and not finished else "done"
        elif EventType.STAGE_SKIPPED in kinds:
            status = "skipped"
        else:
            status = "waiting"
        out.append(Step(label, status, seconds if seconds else None))
    return out


def stepper(events: Sequence[RunEvent], *, finished: bool) -> None:
    """Stage stepper as a compact two-column text table (readable at 360 px)."""
    rows = [{"Stage": s.label, "Status": SYMBOLS[s.status],
             "Time": duration(s.seconds) if s.seconds else ""}
            for s in steps(events, finished=finished)]  # fmt: skip
    st.dataframe(rows, hide_index=True)


def meters(events: Sequence[RunEvent], elapsed_s: float, budget_s: int, trials_max: int,
           llm_max: int) -> None:  # fmt: skip
    """Budget meters: time, trials and LLM calls (§27.1)."""
    trials = sum(e.type == EventType.TRIAL_COMPLETED for e in events)
    calls = sum(e.type in (EventType.LLM_CALL_SUCCEEDED, EventType.LLM_CALL_FAILED)
                for e in events)  # fmt: skip
    st.progress(min(1.0, elapsed_s / budget_s),
                text=f"Time: {duration(elapsed_s)} of {duration(budget_s)}")  # fmt: skip
    st.progress(min(1.0, trials / max(trials_max, 1)),
                text=f"Tuning trials: {trials} of at most {trials_max}")  # fmt: skip
    st.progress(min(1.0, calls / max(llm_max, 1)),
                text=f"LLM calls: {calls} of at most {llm_max}")  # fmt: skip
