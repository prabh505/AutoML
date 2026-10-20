"""Facts table, narrative with fact-check, HTML/Markdown report (PRD §21, FR-RPT-01).

Interface: build_report(state, out_dir, router=..., budget=...). Depends on llm, explain outputs.

Must not: training; numbers not in the facts table.
"""

from tabpilot.reporting.factcheck import FactCheckResult, fact_check
from tabpilot.reporting.facts import Fact, FactsTable, ReportState, build_facts
from tabpilot.reporting.figures import render_figures
from tabpilot.reporting.narrative import (
    Narrative,
    NarratorOutput,
    checked_template,
    narrate,
    template_narrative,
    versions,
)
from tabpilot.reporting.render import (
    ReportPaths,
    build_report,
    final_report,
    md_escape,
    render_report,
)

__all__ = [
    "Fact",
    "FactCheckResult",
    "FactsTable",
    "Narrative",
    "NarratorOutput",
    "ReportPaths",
    "ReportState",
    "build_facts",
    "build_report",
    "checked_template",
    "fact_check",
    "final_report",
    "md_escape",
    "narrate",
    "render_figures",
    "render_report",
    "template_narrative",
    "versions",
]
