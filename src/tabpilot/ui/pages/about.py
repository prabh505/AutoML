"""Page 5 — About & Privacy (PRD §27.1, §24.4–§24.7, §27.3 Cold start).

What is sent where, the provider notices, limitations, the retention policy, and the source link.
"""

from __future__ import annotations

import streamlit as st

from tabpilot.runtime.catalog import WHAT_IS_SENT, provider_options, samples
from tabpilot.ui import services

SOURCE_URL = "https://github.com/prabh505/automl"
RETENTION = [
    {"When": "Run completed", "What is deleted": "The uploaded file and its parsed copy, right "
     "after export. The report and bundle stay until you delete them or the idle time ends."},
    {"When": "Run failed or cancelled", "What is deleted": "The whole run directory, within 15 s."},
    {"When": "You click “Delete my data now”", "What is deleted": "The whole run directory."},
    {"When": "30 idle minutes (demo)", "What is deleted": "The whole run directory."},
    {"When": "Every 10 minutes", "What is deleted": "Any run directory older than 2 hours "
     "(demo) or 7 days (local)."},
]  # fmt: skip
LIMITATIONS = [
    "Tabular classification and regression only; no time-series forecasting.",
    "The demo limits files to the sizes shown on the Upload page and runs one analysis at a time.",
    "Scores are estimates from cross-validation and, for 2,000 rows or more, one holdout set.",
    "Free LLM providers can be slow or unavailable; the rule-based plan is then used.",
    "Deleted files may remain in the host's disk blocks until they are overwritten.",
]


def render() -> None:
    st.header("About & Privacy")
    st.write("TabPilot trains and validates models on your table with leakage-safe "
             "cross-validation. An optional LLM proposes the plan and reflections. TabPilot's "
             "own rules train, score and select every model.")  # fmt: skip
    st.subheader("What is sent where")
    st.write(WHAT_IS_SENT)
    st.write("With “No LLM”, nothing leaves this server. The Configure page shows the exact "
             "planner input before you start.")  # fmt: skip
    st.subheader("Provider notices")
    for option in provider_options(services.settings(), {}):
        st.caption(option.notice)
    st.subheader("Retention")
    st.dataframe(RETENTION, hide_index=True)
    st.subheader("Limitations")
    st.markdown("\n".join(f"- {x}" for x in LIMITATIONS))
    st.subheader("Sample data")
    st.markdown("\n".join(f"- {s.title}: {s.license}" for s in samples()))
    st.subheader("Cold start")
    st.write(
        "On the free hosting tier the app sleeps when nobody uses it. Waking it shows the "
        "host's own loading screen, which TabPilot cannot change; it takes about a minute."
    )
    st.subheader("Source")
    st.markdown(f"Source code and documentation: [{SOURCE_URL}]({SOURCE_URL})")
