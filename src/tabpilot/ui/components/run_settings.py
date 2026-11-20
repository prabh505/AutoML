"""Time-budget slider with live feasibility, LLM mode and provider settings, the consent
notice, and the planner-input inspector (PRD §27.2, §24.5, §3.2 stages 4–5, FR-ING-05, US-06).
"""

from __future__ import annotations

import math
import os

import streamlit as st

from tabpilot.runtime.catalog import WHAT_IS_SENT, ProviderOption, provider_options
from tabpilot.schemas import ErrorCode, ExecutionProfileName, FeasibilityReport, RunConfig
from tabpilot.schemas.errors import render_message
from tabpilot.schemas.run import MIN_TIME_BUDGET_S
from tabpilot.ui import errors, services
from tabpilot.ui import state as sk
from tabpilot.ui.config import (
    LLM_LABELS,
    LLM_OFF,
    LLM_OLLAMA,
    LLM_PROVIDERS,
    LLM_STUB,
    needs_consent,
)
from tabpilot.ui.text import duration

BUDGET_STEP_S = 30  # §27.2
SIMPLIFICATION_TEXT = {
    "reduce_tuning_trials": "run fewer tuning trials",
    "within_fold_subsample": "tune on a sample of each training fold",
    "disable_random_forest": "skip the random forest",
    "reduce_folds": "use 3 folds instead of 5",
    "cap_onehot": "group rare categories more aggressively",
    "disable_ensemble": "skip the ensemble",
}


def providers() -> list[ProviderOption]:
    return provider_options(services.settings(), os.environ)


def available_names() -> list[str]:
    return [p.name.value for p in providers() if p.available and p.name.value != LLM_OLLAMA]


def budget_slider() -> None:
    settings = services.settings()
    limits = settings.limits
    state = st.session_state
    if sk.BUDGET not in state:
        state[sk.BUDGET] = limits.time_budget_default_s
    st.slider("Time budget (seconds)", MIN_TIME_BUDGET_S, limits.time_budget_max_s,
              step=BUDGET_STEP_S, key=sk.BUDGET,
              help="The run stops at this budget and keeps the best result so far.")  # fmt: skip
    st.caption(f"= {duration(state[sk.BUDGET])} (m:ss)")


def feasibility(report: FeasibilityReport) -> bool:
    """Show the estimate; False when the run must not start (E-RESOURCE-INFEASIBLE)."""
    if report.status == "rejected":
        errors.show({"code": ErrorCode.RESOURCE_INFEASIBLE.value, "message": render_message(
            ErrorCode.RESOURCE_INFEASIBLE, {"est": math.ceil(report.estimated_peak_mb),
                                            "limit": math.floor(report.limit_mb)})})  # fmt: skip
        return False
    text = (f"Estimated peak memory {report.estimated_peak_mb:,.0f} MB of "
            f"{report.limit_mb:,.0f} MB available.")  # fmt: skip
    if report.status == "simplified":
        steps = "; ".join(SIMPLIFICATION_TEXT.get(s.value, s.value.replace("_", " "))
                          for s in report.simplifications)  # fmt: skip
        st.info(f"{text} To fit, the run will: {steps}.", icon=":material/info:")
    else:
        st.caption(f"Feasible: {text}")
    return True


def llm_modes() -> list[str]:
    modes = [LLM_PROVIDERS]
    if services.settings().profile == ExecutionProfileName.LOCAL:
        modes.append(LLM_OLLAMA)
    modes.append(LLM_OFF)
    if services.stub_enabled():
        modes.append(LLM_STUB)
    return modes


def _consent(granted: bool) -> None:
    state = st.session_state
    chosen = needs_consent(state, available_names())
    if granted:
        state[sk.CONSENTED] = sorted({*state.get(sk.CONSENTED, []), *chosen})
    state.pop(sk.INSPECT, None)


def _set_mode(mode: str) -> None:
    st.session_state[sk.LLM_MODE] = mode
    st.session_state.pop(sk.INSPECT, None)


def llm_settings() -> None:
    """LLM mode, ordered providers (unavailable ones listed with the reason) and consent."""
    state = st.session_state
    options = providers()
    names = available_names()
    modes = llm_modes()
    if state.get(sk.LLM_MODE) not in modes:
        state[sk.LLM_MODE] = LLM_PROVIDERS if names else LLM_OFF
    st.radio("LLM assistance", modes, key=sk.LLM_MODE, format_func=LLM_LABELS.__getitem__,
             help="The LLM only proposes a plan and reflections; models are always trained "
                  "and selected by TabPilot's own rules.")  # fmt: skip
    st.checkbox("Minimal profile (send less: no value ranges)", key=sk.MINIMAL)
    mode = state[sk.LLM_MODE]
    if mode == LLM_PROVIDERS:
        if sk.PROVIDERS not in state:
            state[sk.PROVIDERS] = names
        st.multiselect("Providers, in the order to try", names, key=sk.PROVIDERS)
        unavailable = [p for p in options if not p.available and p.name.value != LLM_OLLAMA]
        if unavailable:
            st.caption("Unavailable: " + "; ".join(f"{p.name.value} — {p.reason}"
                                                   for p in unavailable))  # fmt: skip
        if not names:
            st.info("No LLM provider is configured on this server. Choose No LLM to run with "
                    "the rule-based plan.")  # fmt: skip
    if mode == LLM_OLLAMA:
        ollama = next((p for p in options if p.name.value == LLM_OLLAMA), None)
        if ollama is not None and not ollama.available:
            st.caption(f"Ollama is unavailable: {ollama.reason}. The run falls back to the "
                       "rule-based plan.")  # fmt: skip
    consent_panel()


def consent_panel() -> None:
    """§24.5: which providers, what is sent, each provider's data-use summary, and the choices
    Continue / Use local model (Ollama, local only) / No LLM. Per session, per provider.
    """
    state = st.session_state
    pending = needs_consent(state, available_names())
    if not pending:
        return
    by_name = {p.name.value: p for p in providers()}
    with st.container(border=True):
        st.markdown("**Before the first LLM-enabled run: privacy notice**")
        st.write(f"Providers: {', '.join(pending)}.")
        st.write(WHAT_IS_SENT)
        for name in pending:
            st.caption(by_name[name].notice)
        cols = st.columns(3)
        cols[0].button("Continue", type="primary", on_click=_consent, args=(True,),
                       key="consent_continue")  # fmt: skip
        if LLM_OLLAMA in llm_modes():
            cols[1].button("Use local model (Ollama)", on_click=_set_mode, args=(LLM_OLLAMA,),
                           key="consent_ollama")  # fmt: skip
        cols[2].button("No LLM", on_click=_set_mode, args=(LLM_OFF,), key="consent_off")


def _inspect() -> None:
    st.session_state[sk.INSPECT] = True


def inspector(config: RunConfig | None) -> None:
    """Planner-input inspector: the exact JSON the planner call would carry (US-06)."""
    state = st.session_state
    with st.expander("Inspect planner input (exactly what is sent to the LLM)"):
        if config is None:
            st.caption("Complete the settings above to build the planner input.")
            return
        if not state.get(sk.INSPECT):
            st.button("Build planner input", on_click=_inspect, key="inspect_build",
                      help="Splits and profiles your data the same way the run will.")  # fmt: skip
            return
        try:
            with st.spinner("Profiling your data…"):
                text = services.manager().preview_planner_input(
                    config.run_id, config, services.worker_options
                )
        except Exception as exc:
            errors.show(errors.describe(exc))
            return
        if config.llm_mode == LLM_OFF:
            st.caption("No LLM is selected: nothing is sent. This is what would be sent.")
        st.code(text, language="json", wrap_lines=True)
        st.caption(f"{len(text.encode()):,} bytes. Use the copy button at the top right.")
        if config.llm_mode != LLM_OFF:
            st.button("Switch to rule-based mode (No LLM)", on_click=_set_mode, args=(LLM_OFF,),
                      key="inspect_off")  # fmt: skip
