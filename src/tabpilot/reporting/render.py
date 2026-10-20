"""HTML and Markdown report (PRD §21.1, §21.6, §18.3, §18.5, §24.8; FR-RPT-01).

The report is a list of sections made of blocks (paragraph, note, list, table, figure). Both
formats render the same blocks with Jinja2:

- HTML: `autoescape=True`, so every value is escaped; no template uses `|safe`.
- Markdown: a `finalize` hook escapes every value (`md_escape`), so column names such as
  `<script>` or `[x](javascript:…)` render as inert text.

Templates contain fixed text only; user-derived strings (file name, column names, category
labels in error analysis) only ever arrive as data. The CV leaderboard holds DEV scores only;
holdout numbers appear only in the summary and the "Independent test" section (T-EVL-06).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tabpilot.explain import ErrorAnalysis, ExplanationResult
from tabpilot.llm import BudgetTracker, LLMRouter
from tabpilot.reporting.facts import FactsTable, ReportState, build_facts
from tabpilot.reporting.figures import render_figures
from tabpilot.reporting.narrative import Narrative, narrate
from tabpilot.schemas import Task
from tabpilot.schemas.experiment import LeaderboardEntry
from tabpilot.schemas.report import FinalReport, ReportSections
from tabpilot.tracking.records import dumps

TEMPLATES = Path(__file__).parent / "templates"
MD_SPECIALS = set("\\`*_{}[]<>&|#!~")
CAVEATS = (
    "Importance and SHAP values describe how this model uses features, not causal effects in "
    "the world.",
    "Correlated features share or swap importance; a low importance does not mean a feature is "
    "irrelevant.",
    "SHAP values are computed on a sample and on preprocessed features; one-hot columns are "
    "aggregated.",
    "Permutation importance can be misleading with correlated features.",
)  # §21.6, fixed text
OPTIMISM = (
    "Cross-validation scores of the selected model are optimistically biased, because "
    "the same folds were used for tuning and selection."
)  # §18.5
PAIRED_NOTE = (
    "Paired fold differences are descriptive: with this few folds they are not a "
    "significance test."
)  # §18.3
HOLDOUT_TITLE = "Independent test (used once, after selection)"
DECIMALS = 4


def md_escape(value: object) -> str:
    """Escape Markdown and HTML specials; newlines become spaces (one-line cells)."""
    text = " ".join(str(value).split("\n"))
    return "".join(f"\\{ch}" if ch in MD_SPECIALS else ch for ch in text)


def _md_finalize(value: object) -> str:
    return "" if value is None else md_escape(value)


def _env(markdown: bool) -> Environment:
    if markdown:
        return Environment(loader=FileSystemLoader(TEMPLATES), autoescape=False,  # noqa: S701 - finalize escapes every value
                           finalize=_md_finalize, undefined=StrictUndefined,
                           keep_trailing_newline=True)  # fmt: skip
    return Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True,
                       undefined=StrictUndefined, keep_trailing_newline=True)  # fmt: skip


def num(value: float | int | None, decimals: int = DECIMALS) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------- blocks
def p(text: str) -> dict[str, Any]:
    return {"kind": "p", "text": text}


def note(text: str) -> dict[str, Any]:
    return {"kind": "note", "text": text}


def items(values: Sequence[str]) -> dict[str, Any]:
    return {"kind": "list", "items": list(values)}


def table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], caption: str = ""
) -> dict[str, Any]:
    return {"kind": "table", "headers": list(headers), "caption": caption,
            "rows": [[str(c) for c in r] for r in rows]}  # fmt: skip


def figure(src: str, alt: str) -> dict[str, Any]:
    return {"kind": "figure", "src": src, "alt": alt}


@dataclass
class Section:
    id: str
    title: str
    blocks: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------- sections
def _holdout_scores(state: ReportState) -> tuple[float | None, float | None, Any]:
    metric = state.config.primary_metric.name.value
    final = (state.holdout.final or {}).get(metric)
    base = (state.holdout.baseline or {}).get(metric)
    ci = (state.holdout.final or {}).get("ci95")
    return (final if isinstance(final, float | int) else None,
            base if isinstance(base, float | int) else None, ci)  # fmt: skip


def _summary(state: ReportState) -> Section:
    cfg, win = state.config, state.winner_entry
    final, _, _ = _holdout_scores(state)
    rows = [["Task", cfg.task.value], ["Target", cfg.target],
            ["Final model", f"{win.model} ({win.entry_id})"],
            ["Primary metric", f"{cfg.primary_metric.name.value} ({cfg.primary_metric.direction})"],
            ["CV score (DEV, mean ± std)", f"{num(win.mean)} ± {num(win.std)}"],
            ["Independent test score",
             num(final) if state.holdout.present else "none (no holdout)"],
            ["Run status", state.status]]  # fmt: skip
    return Section("summary", "Summary", [table(["Item", "Value"], rows)])


def _dataset(state: ReportState) -> Section:
    meta = state.metadata
    rows: list[list[Any]] = [["File", meta.source.display_name], ["Format", meta.source.format],
            ["Rows (raw / valid)", f"{meta.n_rows_raw} / {meta.n_rows_valid}"],
            ["Columns", meta.n_columns]]  # fmt: skip
    if state.profile is not None:
        rows += [["Development rows", state.profile.n_rows_dev],
                 ["Feature columns", state.profile.n_features],
                 ["Duplicate rows", state.profile.duplicate_rows]]  # fmt: skip
        counts = state.profile.target.class_counts
        if counts:
            rows.append(["Class counts (DEV)", ", ".join(f"{k}: {v}" for k, v in counts.items())])
    split = state.config.split
    rows.append(["Validation", f"{split.strategy}, k = {split.k}, repeats = {split.n_repeats}, "
                 f"holdout fraction = {split.holdout_fraction}"])  # fmt: skip
    return Section("dataset", "Dataset summary", [table(["Item", "Value"], rows)])


def _decisions(state: ReportState) -> Section:
    meta = state.metadata
    blocks = []
    removals = [[r.count, r.reason] for r in meta.row_removals if r.count]
    if removals:
        blocks.append(table(["Rows removed", "Reason"], removals))
    excluded = [[c.original, c.name, c.exclusion_reason] for c in meta.columns
                if c.exclusion_reason]  # fmt: skip
    if excluded:
        blocks.append(table(["Excluded column", "Normalized name", "Reason"], excluded))
    coerced = [[c.original, c.coerced_to_missing] for c in meta.columns if c.coerced_to_missing]
    if coerced:
        blocks.append(table(["Column", "Values coerced to missing"], coerced))
    renamed = [[c.original, c.name] for c in meta.columns if c.original != c.name]
    if renamed:
        blocks.append(table(["Original column name", "Normalized name"], renamed))
    return Section(
        "data-decisions",
        "Data decisions",
        blocks or [p("No rows were removed and no columns were excluded.")],
    )


def _warnings(state: ReportState) -> Section:
    found = list(state.metadata.warnings) + list(state.warnings)
    blocks = []
    if state.profile is not None:
        found += list(state.profile.warnings)
        if state.profile.leakage:
            blocks.append(table(["Column", "Score", "Metric", "Level"],
                                [[f.column, num(f.score, 3), f.metric, f.level]
                                 for f in state.profile.leakage]))  # fmt: skip
    if found:
        blocks.insert(0, items(sorted(set(found))))
    return Section(
        "warnings",
        "Data-quality and leakage warnings",
        blocks or [p("No data-quality or leakage warnings were raised.")],
    )


def _plan(state: ReportState) -> Section:
    plan = state.plan
    source = "LLM planner" if state.plan_source == "llm" else "rule-based planner"
    blocks: list[dict[str, Any]] = [p(f"Plan source: {source}.")]
    blocks.append(
        table(
            ["Model", "Priority", "Class weight", "Tuning share"],
            [
                [
                    m.name.value,
                    m.priority,
                    m.class_weight.value,
                    num(plan.tuning_allocation.get(m.name), 2),
                ]
                for m in plan.models
            ],
        )
    )
    if state.plan_validation is not None:
        v = state.plan_validation
        blocks.append(p(f"Validation status: {v.status}."))
        if v.issues:
            blocks.append(table(["Issue", "Severity", "Detail"],
                                [[i.code, i.severity, i.message] for i in v.issues]))  # fmt: skip
    return Section("plan", "Modeling plan", blocks)


def _preprocessing(state: ReportState) -> Section:
    cand, plan = state.winner, state.plan
    imputation = cand.imputation if cand else plan.imputation
    encoding = cand.encoding if cand else plan.encoding
    transform = cand.target_transform if cand else plan.target_transform
    rows = [["Numeric imputation", imputation.numeric],
            ["Categorical imputation", imputation.categorical],
            ["Missing indicators", "yes" if imputation.add_missing_indicators else "no"],
            ["Low-cardinality encoding", encoding.low_cardinality],
            ["High-cardinality encoding", encoding.high_cardinality],
            ["Target transform", transform.value]]  # fmt: skip
    blocks = [table(["Decision", "Value"], rows),
              note("Every learned transformation is fitted inside the pipeline on training-fold "
                   "rows only.")]  # fmt: skip
    decisions = cand.column_decisions if cand else plan.column_decisions
    if decisions:
        blocks.append(table(["Column", "Action", "Reason"],
                            [[d.column, d.action, d.reason] for d in decisions]))  # fmt: skip
    return Section("preprocessing", "Preprocessing decisions", blocks)


def _features(state: ReportState) -> Section:
    blocks = [p("Baseline features: datetime parts, missing-value indicators and rare-category "
                "grouping, built inside the pipeline.")]  # fmt: skip
    llm = [f for f in state.provenance if f.source != "baseline"]
    if llm:
        blocks.append(table(["Feature", "Source", "Iteration", "Status", "Note"],
                            [[f.name, f.source, f.iteration, f.status, f.failure_reason or ""]
                             for f in llm]))  # fmt: skip
    else:
        blocks.append(p("No LLM-proposed features were used."))
    return Section("features", "Features", blocks)


def _leaderboard_rows(entries: Sequence[LeaderboardEntry]) -> list[list[Any]]:
    return [[e.entry_id, e.model, e.feature_set, num(e.mean), num(e.std),
             ", ".join(num(s) for s in e.fold_scores), num(e.fit_seconds, 2), e.status]
            for e in entries]  # fmt: skip


def _leaderboard(state: ReportState, figures: dict[str, str]) -> Section:
    headers = ["Entry", "Model", "Features", "CV mean", "CV std", "Fold scores", "Fit s", "Status"]
    comparable = [e for e in state.entries if e.comparable]
    ok = [e for e in comparable if e.status == "ok" and e.mean is not None]
    sign = 1.0 if state.config.primary_metric.direction == "maximize" else -1.0
    ok.sort(key=lambda e: (-sign * (e.mean or 0.0), e.entry_id))
    failed = [e for e in comparable if e not in ok]
    blocks = [table(headers, _leaderboard_rows(ok + failed),
                    caption="Cross-validation on development data (DEV) only")]  # fmt: skip
    for name, alt in (("leaderboard", "CV mean score with ±std error bars"),
                      ("fold_scores", "CV score of every fold")):  # fmt: skip
        if name in figures:
            blocks.append(figure(figures[name], alt))
    other = [e for e in state.entries if not e.comparable]
    if other:
        blocks.append(table(headers, _leaderboard_rows(other), caption="Not comparable"))
    for eid, report in sorted(state.thresholds.items()):
        rows = [[name.replace("_", " "), num(v["threshold"], 3), num(v["accuracy"]), num(v["f1"])]
                for name, v in report.items()]  # fmt: skip
        blocks.append(table(["Threshold rule", "Threshold", "Accuracy", "F1"], rows,
                            caption=f"Threshold metrics on OOF predictions, {eid}"))  # fmt: skip
    return Section("leaderboard", "Leaderboard (cross-validation)", blocks)


def _reflection(state: ReportState) -> Section:
    if not state.reflections:
        return Section("reflection", "Reflection history", [p("No reflection rounds were run.")])
    blocks = [table(["Round", "Status", "Reason"],
                    [[r.round, r.status, r.reason] for r in state.reflections])]  # fmt: skip
    decisions = [[r.round, d.parent_entry_id, d.child_entry_id, num(d.rel_gain),
                  f"{d.folds_better}/{d.folds_required}", "yes" if d.accepted else "no"]
                 for r in state.reflections for d in r.decisions]  # fmt: skip
    if decisions:
        blocks.append(table(["Round", "Parent", "Child", "Relative gain", "Folds better/needed",
                             "Accepted"], decisions))  # fmt: skip
    return Section("reflection", "Reflection history", blocks)


def _ensemble(state: ReportState) -> Section:
    ens = state.ensemble
    if ens is None:
        return Section("ensemble", "Ensemble decision", [p("No ensemble was built.")])
    blocks = [p(f"Ensemble kept: {'yes' if ens.kept else 'no'}. Reason: {ens.reason}")]
    if ens.members:
        blocks.append(table(["Member", "Weight"], [[m.entry_id, num(m.weight, 3)]
                                                   for m in ens.members]))  # fmt: skip
    return Section("ensemble", "Ensemble decision", blocks)


def _selection(state: ReportState) -> Section:
    sel = state.selection
    blocks = [p(f"Rule: {sel.rule}. Winner: {sel.winner}."), note(OPTIMISM)]
    if sel.near_ties:
        blocks.append(p(f"Near ties: {', '.join(sel.near_ties)}. Tie-break: {sel.tie_break}."))
    paired = sel.runner_up_paired
    if paired is not None:
        blocks.append(
            table(
                ["Runner-up", "Mean difference", "Min", "Max", "Folds won"],
                [
                    [
                        paired.runner_up,
                        num(paired.mean_diff),
                        num(paired.min_diff),
                        num(paired.max_diff),
                        f"{paired.folds_won}/{paired.n_folds}",
                    ]
                ],
            )
        )
        blocks.append(note(PAIRED_NOTE))
    return Section("selection", "Why this model was selected", blocks)


def _holdout(state: ReportState) -> Section:
    hold = state.holdout
    metric = state.config.primary_metric.name.value
    if not hold.present:
        return Section("independent-test", HOLDOUT_TITLE,
                       [p(f"No independent test results exist: {hold.reason}.")])  # fmt: skip
    final, base, ci = _holdout_scores(state)
    rows = [["Final model", num(final), f"{num(ci[0])} – {num(ci[1])}" if ci else "—"]]
    if base is not None:
        rows.append([f"Reference baseline ({state.baseline_model or 'baseline'})", num(base), "—"])
    return Section("independent-test", HOLDOUT_TITLE, [
        table(["Model", metric, "Bootstrap 95% interval"], rows),
        note("These rows were used once, after the selection was frozen, and never for training, "
             "tuning or selection."),
    ])  # fmt: skip


def _importance(state: ReportState, figures: dict[str, str]) -> Section:
    exp = state.explanation
    if exp is None:
        return Section("importance", "Feature importance and SHAP",
                       [p("Explanation unavailable: the explainer did not run.")])  # fmt: skip
    blocks = [p(f"Method: {exp.method}. Rows explained: {exp.n_rows}.")]
    if exp.unavailable_reason:
        blocks.append(note(f"Explanation unavailable: {exp.unavailable_reason}"))
    blocks.extend(note(n) for n in exp.notes)
    for name, alt in (("shap_importance", "Mean |SHAP| per source column"),
                      ("permutation_importance", "Permutation importance (one DEV fold)"),
                      ("gain_importance", "Native model importance")):  # fmt: skip
        if name in figures:
            blocks.append(figure(figures[name], alt))
    for k, output in enumerate(exp.shap, start=1):
        key = f"shap_beeswarm_{k}"
        if key in figures:
            alt = f"SHAP values per row, class {output.label}" if output.label else "SHAP values"
            blocks.append(figure(figures[key], alt))
    ranked = exp.global_importance or exp.permutation or exp.gain
    if ranked:
        blocks.append(
            table(
                ["Rank", "Source column", "Importance"],
                [[i, imp.name, num(imp.value)] for i, imp in enumerate(ranked, 1)],
            )
        )
    return Section("importance", "Feature importance and SHAP", blocks)


def _errors(errors: ErrorAnalysis | None, figures: dict[str, str]) -> Section:
    if errors is None:
        return Section("errors", "Error analysis", [p("Error analysis is unavailable.")])
    blocks = [p(f"Overall OOF {errors.error_name} on DEV: {num(errors.overall_error)}.")]
    if errors.per_class:
        blocks.append(table(["Class", "Precision", "Recall", "Support"],
                            [[c.label, num(c.precision), num(c.recall), c.support]
                             for c in errors.per_class]))  # fmt: skip
    if errors.deciles:
        blocks.append(table(["Predicted from", "to", "Mean absolute error", "Rows"],
                            [[num(d.lower), num(d.upper), num(d.mae), d.n]
                             for d in errors.deciles]))  # fmt: skip
    if errors.segments:
        blocks.append(
            table(
                ["Column", "Rank", "Category", "Rows", errors.error_name.capitalize()],
                [[s.column, s.rank, s.label, s.n, num(s.error)] for s in errors.segments],
                caption="Worst segments (categories with at least 30 DEV rows)",
            )
        )
    for name, alt in (("confusion_matrix", "Confusion matrix (OOF, row-normalized)"),
                      ("roc_curve", "ROC curve (OOF)"), ("pr_curve", "Precision–recall (OOF)"),
                      ("reliability", "Reliability curve (OOF, 10 bins)"),
                      ("residuals_vs_predicted", "Residuals vs predicted (OOF)"),
                      ("residual_histogram", "Residual histogram (OOF)"),
                      ("actual_vs_predicted", "Actual vs predicted (OOF)")):  # fmt: skip
        if name in figures:
            blocks.append(figure(figures[name], alt))
    return Section("errors", "Error analysis", blocks)


def limitations(state: ReportState) -> list[str]:
    """Fixed caveats (§21.6, §18.5) plus run-specific items."""
    out = [*CAVEATS, OPTIMISM]
    if not state.holdout.present:
        out.append(f"No independent test set: {state.holdout.reason}.")
    if state.explanation is not None and state.explanation.unavailable_reason:
        out.append(f"Explanation unavailable: {state.explanation.unavailable_reason}")
    if state.status != "completed":
        out.append(f"The run ended with status {state.status}; some stages did not finish.")
    return out


def _reproducibility(state: ReportState) -> Section:
    cfg = state.config
    rows: list[list[Any]] = [
        ["Run id", cfg.run_id],
        ["TabPilot version", cfg.tabpilot_version],
        ["Execution profile", cfg.execution_profile.value],
        ["Split id", cfg.split.split_id],
        ["Seeds", ", ".join(f"{k}={v}" for k, v in cfg.seeds.model_dump().items())],
        ["Libraries", ", ".join(f"{k} {v}" for k, v in sorted(cfg.library_versions.items()))],
    ]
    if cfg.prompt_template_versions:
        rows.append(["Prompt templates", ", ".join(
            f"{k}@{v}" for k, v in sorted(cfg.prompt_template_versions.items()))])  # fmt: skip
    if state.ledger is not None:
        rows += [
            ["Time budget (s)", state.ledger.time_budget_s],
            ["Elapsed at report (s)", num(state.ledger.elapsed_s, 1)],
            ["Trials used", f"{state.ledger.trials_used} of {state.ledger.trials_max}"],
        ]
    return Section(
        "reproducibility", "Reproducibility information", [table(["Item", "Value"], rows)]
    )


def _narrative(narrative: Narrative) -> Section:
    if narrative.source == "llm":
        label = "Written by an LLM from the facts table and checked against it (fact-check passed)."
    else:
        label = "Template narrative, built deterministically from the facts table."
        if narrative.fallback_reason:
            label += f" Reason: {narrative.fallback_reason}."
    blocks = [p(par) for par in narrative.text.split("\n\n")]
    return Section("narrative", "Narrative", [note(label), *blocks])


def sections(state: ReportState, narrative: Narrative, figures: dict[str, str]) -> list[Section]:
    """The 17 sections of §21.1, in order."""
    return [
        _summary(state), _dataset(state), _decisions(state), _warnings(state), _plan(state),
        _preprocessing(state), _features(state), _leaderboard(state, figures), _reflection(state),
        _ensemble(state), _selection(state), _holdout(state), _importance(state, figures),
        _errors(state.errors, figures),
        Section("limitations", "Limitations", [items(limitations(state))]),
        _reproducibility(state), _narrative(narrative),
    ]  # fmt: skip


def render_report(
    state: ReportState, narrative: Narrative, figures: dict[str, str]
) -> tuple[str, str]:
    """(html, markdown) for one run."""
    context = {
        "run_id": state.config.run_id,
        "file": state.metadata.source.display_name,
        "status": state.status,
        "sections": [
            {"number": i, "id": s.id, "title": s.title, "blocks": s.blocks}
            for i, s in enumerate(sections(state, narrative, figures), start=1)
        ],
    }
    html = _env(markdown=False).get_template("report.html.j2").render(**context)
    md = _env(markdown=True).get_template("report.md.j2").render(**context)
    return html, md


# ---------------------------------------------------------------- build
@dataclass
class ReportPaths:
    """Files written by `build_report` (paths relative to the run directory's parent of report/)."""

    html: Path
    markdown: Path
    report_json: Path
    facts_json: Path
    figures: dict[str, str]
    narrative: Narrative


def final_report(state: ReportState, narrative: Narrative,
                 explanation: ExplanationResult | None) -> FinalReport:  # fmt: skip
    """The `report/1.0` contract."""
    win = state.winner_entry
    final, base, _ = _holdout_scores(state)
    summary: dict[str, Any] = {"task": state.config.task.value, "target": state.config.target,
                               "final_model": str(win.model), "winner": win.entry_id,
                               "primary_metric": state.config.primary_metric.name.value,
                               "cv_mean": win.mean, "cv_std": win.std,
                               "holdout": final}  # fmt: skip
    data_decisions: list[dict[str, Any]] = [
        {"kind": "row_removal", "reason": r.reason, "count": r.count}
        for r in state.metadata.row_removals
    ] + [{"kind": "column_excluded", "column": c.name, "reason": c.exclusion_reason}
         for c in state.metadata.columns if c.exclusion_reason]  # fmt: skip
    explanations = (
        explanation.section()
        if explanation is not None
        else ExplanationResult(method="none", unavailable_reason="explainer did not run").section()
    )
    sections_ = ReportSections(
        summary=summary, data_decisions=data_decisions,
        warnings=sorted(set(state.metadata.warnings) | set(state.warnings)),
        plan={"source": state.plan_source, "models": [m.name.value for m in state.plan.models]},
        leaderboard_ref="leaderboard.json",
        selection={"winner": state.selection.winner, "rule": state.selection.rule},
        holdout={"present": state.holdout.present, "final": final, "baseline": base},
        explanations=explanations, narrative=narrative.section(),
        limitations=limitations(state),
    )  # fmt: skip
    return FinalReport(run_id=state.config.run_id, status=state.status, sections=sections_,
                       facts_ref="report/facts.json")  # fmt: skip


def build_report(
    state: ReportState,
    out_dir: Path,
    *,
    router: LLMRouter | None = None,
    budget: BudgetTracker | None = None,
) -> ReportPaths:
    """Facts → narrative (LLM or template) → figures → HTML/Markdown/JSON in `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    facts: FactsTable = build_facts(state)
    narrative = narrate(facts, router, budget, state.events)
    oof = state.winner_oof if state.problem.task != Task.REGRESSION else None
    figures = render_figures(entries=state.entries, explanation=state.explanation,
                             errors=state.errors, oof=oof, y=state.y_dev, problem=state.problem,
                             out_dir=out_dir / "figures")  # fmt: skip
    html, md = render_report(state, narrative, figures)
    paths = ReportPaths(html=out_dir / "report.html", markdown=out_dir / "report.md",
                        report_json=out_dir / "report.json", facts_json=out_dir / "facts.json",
                        figures=figures, narrative=narrative)  # fmt: skip
    paths.html.write_text(html, encoding="utf-8")
    paths.markdown.write_text(md, encoding="utf-8")
    paths.facts_json.write_text(
        json.dumps(facts.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = final_report(state, narrative, state.explanation)
    paths.report_json.write_text(dumps(report, indent=2) + "\n", encoding="utf-8")
    return paths
