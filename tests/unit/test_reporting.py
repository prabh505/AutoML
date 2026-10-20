"""P13 reporting tests: facts, fact-check, narratives, figures, HTML/Markdown (PRD §21)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.helpers.llm_stub import ScriptedLLMProvider

from tabpilot.evaluation.metrics import Problem
from tabpilot.explain import (
    ClassMetrics,
    ErrorAnalysis,
    ExplanationResult,
    Importance,
    Segment,
    ShapOutput,
    error_analysis,
)
from tabpilot.llm import BudgetTracker, LLMRouter
from tabpilot.llm.consent import ConsentStore
from tabpilot.llm.errors import classify_status
from tabpilot.reporting import (
    FactsTable,
    NarratorOutput,
    ReportState,
    build_facts,
    build_report,
    fact_check,
    md_escape,
    narrate,
    render_figures,
    render_report,
    template_narrative,
)
from tabpilot.reporting.narrative import load, versions
from tabpilot.reporting.render import CAVEATS, HOLDOUT_TITLE
from tabpilot.schemas import (
    BudgetLedger,
    CandidateSpec,
    DatasetMetadata,
    EnsembleResult,
    EventType,
    ExperimentConfig,
    LLMBudget,
    PlannerOutput,
    PlanValidation,
    ProfileSummary,
    ProviderName,
    RunEvent,
    Stage,
    Task,
)
from tabpilot.schemas.experiment import EvaluationResult, HoldoutResult
from tabpilot.schemas.features import FeatureProvenance
from tabpilot.schemas.profile import LeakageFinding
from tabpilot.schemas.reflection import ReflectionRecord
from tabpilot.schemas.report import FinalReport

CONTRACTS = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"
INJECTION = Path(__file__).resolve().parents[2] / "fixtures" / "malicious" / "injection_columns.csv"
BIN = Problem(Task.BINARY, ("No", "Yes"), 1)
RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
SECTION_TITLES = [
    "Summary", "Dataset summary", "Data decisions", "Data-quality and leakage warnings",
    "Modeling plan", "Preprocessing decisions", "Features", "Leaderboard (cross-validation)",
    "Reflection history", "Ensemble decision", "Why this model was selected", HOLDOUT_TITLE,
    "Feature importance and SHAP", "Error analysis", "Limitations",
    "Reproducibility information", "Narrative",
]  # fmt: skip


def contract(name: str) -> dict[str, Any]:
    data = json.loads((CONTRACTS / f"{name}.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def event(kind: EventType, seq: int = 0, **data: Any) -> RunEvent:
    return RunEvent(run_id=RID, seq=seq, ts=datetime(2026, 10, 20, tzinfo=UTC),
                    stage=Stage.RUN, type=kind, data=data)  # fmt: skip


def explanation() -> ExplanationResult:
    shap = ShapOutput("", ["num__totalcharges", "num__tenure"], np.array([[0.1, -0.2], [0.3, 0.0]]),
                      np.array([[1.0, 2.0], [3.0, np.nan]]))  # fmt: skip
    return ExplanationResult(method="shap_tree",
                             global_importance=[Importance("totalcharges", 0.2),
                                                Importance("tenure", 0.1)],
                             gain=[Importance("totalcharges", 5.0)], shap=[shap], n_rows=2)  # fmt: skip


def state(**overrides: Any) -> ReportState:
    ev = EvaluationResult.model_validate(contract("prd_26_9_EvaluationResult"))
    paired = ev.selection.runner_up_paired
    assert paired is not None
    selection = ev.selection.model_copy(update={"runner_up_paired": paired.model_copy(
        update={"runner_up": "0-linear-77ab0c11", "n_folds": 5})})  # fmt: skip
    y = np.array([0, 1] * 50)
    oof = np.column_stack([1 - np.linspace(0.1, 0.9, 100), np.linspace(0.1, 0.9, 100)])
    base: dict[str, Any] = {
        "config": ExperimentConfig.model_validate(contract("prd_26_7_ExperimentConfig")),
        "metadata": DatasetMetadata.model_validate(contract("prd_26_1_DatasetMetadata")),
        "plan": PlannerOutput.model_validate(contract("prd_26_4_PlannerOutput")),
        "entries": ev.entries, "selection": selection, "holdout": ev.holdout, "problem": BIN,
        "events": [event(EventType.HOLDOUT_EVALUATED, present=True)], "baseline_model": "lightgbm",
        "explanation": explanation(),
        "errors": ErrorAnalysis(task=Task.BINARY, error_name="error rate", overall_error=0.2,
                             labels=["No", "Yes"], confusion=np.array([[40, 10], [10, 40]]),
                             per_class=[ClassMetrics("No", 0.8, 0.8, 50),
                                        ClassMetrics("Yes", 0.8, 0.8, 50)],
                             segments=[Segment("contract", 1, "Month-to-month", 40, 0.4)]),
        "winner_oof": oof.astype(np.float32), "y_dev": y,
    }  # fmt: skip
    base.update(overrides)
    return ReportState(**base)


def sections_of(html: str) -> dict[str, str]:
    parts = re.split(r'<section id="([a-z-]+)">', html)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


# ---------------------------------------------------------------- facts
@pytest.mark.unit
def test_facts_cover_report_numbers_and_names() -> None:
    facts = build_facts(state())
    by_key = {f.key: f for f in facts.facts.values()}
    assert list(facts.facts)[:3] == ["F1", "F2", "F3"]
    assert by_key["final_model"].value == "lightgbm"
    assert by_key["cv_mean"].display() == "0.8474"
    assert by_key["holdout_score"].display() == "0.8430"
    assert by_key["holdout_ci_low"].display() == "0.8200"
    assert by_key["runner_up_model"].value == "linear"
    assert by_key["folds_won"].value == 4
    assert by_key["top_feature_1"].value == "totalcharges"
    assert facts.model_names() == {"lightgbm", "linear"}
    assert build_facts(state()).to_json() == facts.to_json()  # deterministic ids


@pytest.mark.unit
def test_facts_payload_holds_no_category_labels() -> None:
    """PI-16 / I-6: class labels and worst-segment categories never enter the facts table."""
    payload = json.dumps(build_facts(state()).payload())
    for label in ("Month-to-month", '"Yes"', '"No"'):
        assert label not in payload


@pytest.mark.unit
def test_facts_require_rounding_for_floats() -> None:
    with pytest.raises(ValueError, match="needs a rounding"):
        FactsTable().add("x", 0.5)


# ---------------------------------------------------------------- fact-check (T-RPT-02/03)
def template_text(**overrides: Any) -> tuple[str, FactsTable, list[RunEvent]]:
    s = state(**overrides)
    facts = build_facts(s)
    return template_narrative(facts).text, facts, s.events


@pytest.mark.unit
def test_template_narrative_passes_and_is_deterministic() -> None:
    text, facts, events = template_text()
    assert fact_check(text, facts, events).passed, fact_check(text, facts, events).violations
    assert template_narrative(facts).text == text
    assert "0.8474 [F" in text
    assert "0.8430 [F" in text


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["no_holdout", "no_runner_up", "ensemble", "no_explanation"])
def test_template_narrative_passes_in_every_shape(variant: str) -> None:
    overrides: dict[str, Any] = {}
    events = [event(EventType.HOLDOUT_EVALUATED, present=True)]
    if variant == "no_holdout":
        overrides["holdout"] = HoldoutResult(present=False, reason="dataset too small")
        events = [event(EventType.HOLDOUT_EVALUATED, present=False)]
    if variant == "no_runner_up":
        ev = EvaluationResult.model_validate(contract("prd_26_9_EvaluationResult"))
        overrides["selection"] = ev.selection.model_copy(update={"runner_up_paired": None})
    if variant == "ensemble":
        overrides["ensemble"] = EnsembleResult.model_validate(contract("EnsembleResult"))
        assert overrides["ensemble"].kept
        events.append(event(EventType.ENSEMBLE_DECIDED, kept=True))
    if variant == "no_explanation":
        overrides["explanation"] = None
    text, facts, _ = template_text(events=events, **overrides)
    result = fact_check(text, facts, events)
    assert result.passed, result.violations
    assert ("ensemble" in text or "blends" in text) == (variant == "ensemble")


@pytest.mark.unit
def test_T_RPT_02_invented_number_is_caught() -> None:
    text, facts, events = template_text()
    result = fact_check(text.replace("0.8474", "0.8512"), facts, events)
    assert any("0.8512 is not in the facts table" in v for v in result.violations)


@pytest.mark.unit
def test_T_RPT_02_wrong_rounding_is_caught() -> None:
    text, facts, events = template_text()
    result = fact_check(text.replace("0.8474", "0.85"), facts, events)
    assert any("0.85 is not in the facts table" in v for v in result.violations)


@pytest.mark.unit
def test_T_RPT_02_uncited_number_is_rejected() -> None:
    text, facts, events = template_text()
    cv = facts.by_key("cv_mean")
    assert cv is not None
    result = fact_check(text.replace(f"0.8474 [{cv.id}]", "0.8474"), facts, events)
    assert any("0.8474 does not cite" in v for v in result.violations)


@pytest.mark.unit
def test_unknown_citation_and_names_are_rejected() -> None:
    text, facts, events = template_text()
    bad = text.replace("[F1]", "[F99]") + " `secret_column` and xgboost and other_name."
    violations = fact_check(bad, facts, events).violations
    assert any("unknown fact F99" in v for v in violations)
    assert any("`secret_column`" in v for v in violations)
    assert any("model xgboost" in v for v in violations)
    assert any("unknown name other_name" in v for v in violations)


@pytest.mark.unit
@pytest.mark.parametrize("phrase", ["Contract type causes churn.", "Tenure leads to churn.",
                                    "Churn happens because of tenure.", "This model will achieve "
                                    "good results.", "Accuracy is guaranteed."])  # fmt: skip
def test_T_RPT_03_causal_and_guarantee_phrases_are_rejected(phrase: str) -> None:
    text, facts, events = template_text()
    result = fact_check(text + " " + phrase, facts, events)
    assert any("causal language" in v or "guarantee" in v for v in result.violations)


@pytest.mark.unit
def test_actions_must_appear_in_the_event_log() -> None:
    text, facts, events = template_text()
    claim = text + " The models were tuned and an ensemble was kept after reflection."
    violations = fact_check(claim, facts, events).violations
    for action in ("tuning", "ensemble", "reflection"):
        assert any(f"mentions {action}" in v for v in violations)
    logged = [*events, event(EventType.CANDIDATE_COMPLETED, trials=8),
              event(EventType.ENSEMBLE_DECIDED, kept=True),
              event(EventType.REFLECTION_DECIDED, accepted=True)]  # fmt: skip
    assert fact_check(claim, facts, logged).passed
    no_holdout = [e for e in events if e.type != EventType.HOLDOUT_EVALUATED]
    assert any("mentions holdout" in v for v in fact_check(text, facts, no_holdout).violations)


@pytest.mark.unit
def test_shape_rules_paragraphs_and_words() -> None:
    text, facts, events = template_text()
    assert any("expected 4 paragraphs" in v
               for v in fact_check(text.replace("\n\n", " ", 1), facts, events).violations)  # fmt: skip
    long = text + " word" * 300
    assert any("more than 300 words" in v for v in fact_check(long, facts, events).violations)


@pytest.mark.unit
def test_names_with_digits_or_banned_words_are_masked() -> None:
    cfg = ExperimentConfig.model_validate(contract("prd_26_7_ExperimentConfig"))
    exp = ExplanationResult(method="gain", gain=[Importance("cause_code_1", 1.0),
                                                 Importance("noise_2", 0.5)])  # fmt: skip
    text, facts, events = template_text(config=cfg.model_copy(update={"target": "leads_to_churn"}),
                                        explanation=exp)  # fmt: skip
    assert "`cause_code_1`" in text
    assert fact_check(text, facts, events).passed, fact_check(text, facts, events).violations


# ---------------------------------------------------------------- narrator (T-RPT-04)
def responses(*contents: str) -> list[dict[str, Any]]:
    return [{"kind": "response", "body": {"id": "c", "object": "chat.completion", "created": 1,
             "model": "m", "choices": [{"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": c}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}}
            for c in contents]  # fmt: skip


def router_for(steps: list[dict[str, Any]]) -> tuple[LLMRouter, ScriptedLLMProvider]:
    provider = ScriptedLLMProvider("groq", steps, raise_with=lambda step: classify_status(
        int(step["status"])))  # fmt: skip
    return LLMRouter([provider], consent=ConsentStore.granted_for(list(ProviderName))), provider


def as_json(text: str) -> str:
    return json.dumps({"paragraphs": text.split("\n\n")})


@pytest.mark.unit
def test_T_RPT_04_no_provider_uses_the_template() -> None:
    _, facts, events = template_text()
    narrative = narrate(facts, None, None, events)
    assert narrative.source == "template"
    assert narrative.fact_check == "passed"
    assert narrative.fallback_reason == "no LLM provider available"


@pytest.mark.unit
def test_llm_narrative_that_passes_is_shipped() -> None:
    text, facts, events = template_text()
    router, provider = router_for(responses(as_json(text)))
    narrative = narrate(facts, router, BudgetTracker(LLMBudget(max_wall_s=100.0)), events)
    assert narrative.source == "llm"
    assert narrative.text == text
    assert provider.requests[0].purpose == "narrate"
    sent = json.dumps(provider.requests[0].user_json)
    assert "Month-to-month" not in sent  # PI-16
    assert provider.requests[0].system == load("narrator_system").text


@pytest.mark.unit
def test_T_RPT_04_failed_fact_check_repairs_once_then_falls_back() -> None:
    text, facts, events = template_text()
    invented = as_json(text.replace("0.8474", "0.9999"))
    router, provider = router_for(responses(invented, invented))
    narrative = narrate(facts, router, BudgetTracker(LLMBudget(max_wall_s=100.0)), events)
    assert provider.calls == 2
    assert provider.requests[1].purpose == "narrate_repair"
    assert "0.9999" in str(provider.requests[1].user_json["violations"])
    assert narrative.source == "template"
    assert narrative.fallback_reason == "LLM narrative failed the fact-check"
    assert any("0.9999" in v for v in narrative.violations)


@pytest.mark.unit
def test_repair_that_passes_is_shipped() -> None:
    text, facts, events = template_text()
    router, provider = router_for(responses("not json at all", as_json(text)))
    narrative = narrate(facts, router, BudgetTracker(LLMBudget(max_wall_s=100.0)), events)
    assert provider.calls == 2
    assert narrative.source == "llm"


@pytest.mark.unit
def test_provider_failure_falls_back_to_the_template() -> None:
    _, facts, events = template_text()
    router, _ = router_for([{"kind": "http_error", "status": 401}])
    narrative = narrate(facts, router, BudgetTracker(LLMBudget(max_wall_s=100.0)), events)
    assert narrative.source == "template"
    assert narrative.fact_check == "passed"


@pytest.mark.unit
def test_narrator_templates_are_versioned_and_filled() -> None:
    assert versions() == {"narrator_system": "1.0", "narrator_repair": "1.0"}
    system = load("narrator_system").text
    assert "{{" not in system
    assert "paragraphs" in system
    with pytest.raises(ValueError, match="at least 4"):
        NarratorOutput.model_validate({"paragraphs": ["one"]})


# ---------------------------------------------------------------- render (T-RPT-01, T-EVL-06)
@pytest.mark.unit
def test_T_RPT_01_every_section_is_present_in_order(tmp_path: Path) -> None:
    paths = build_report(state(), tmp_path / "report")
    html = paths.html.read_text(encoding="utf-8")
    md = paths.markdown.read_text(encoding="utf-8")
    assert re.findall(r"<h2>\d+\. ([^<]+)</h2>", html) == SECTION_TITLES
    assert re.findall(r"^## \d+\. (.+)$", md, flags=re.M) == SECTION_TITLES
    for caveat in CAVEATS:
        assert caveat in html
    assert "optimistically biased" in html
    assert "not a significance test" in html
    assert "Template narrative" in html
    report = FinalReport.model_validate_json(paths.report_json.read_text(encoding="utf-8"))
    assert report.sections.narrative.source == "template"
    assert report.sections.explanations.method == "shap_tree"
    assert json.loads(paths.facts_json.read_text(encoding="utf-8"))["F1"]["key"] == "task"
    for rel in paths.figures.values():
        assert (tmp_path / "report" / rel).stat().st_size > 0
        assert f'src="{rel}"' in html


@pytest.mark.unit
def test_T_EVL_06_holdout_is_reported_separately(tmp_path: Path) -> None:
    holdout = HoldoutResult(present=True, final={"roc_auc": 0.8123, "ci95": [0.7911, 0.8317]},
                            baseline={"roc_auc": 0.8011},
                            accessed_at=datetime(2026, 10, 20, tzinfo=UTC))  # fmt: skip
    html = build_report(state(holdout=holdout), tmp_path / "r").html.read_text(encoding="utf-8")
    parts = sections_of(html)
    assert "0.8123" in parts["independent-test"]
    assert "0.8011" in parts["independent-test"]
    assert "0.8123" in parts["summary"]  # the summary may show it (§18.5)
    for value in ("0.8123", "0.8011", "0.7911", "0.8317"):
        assert value not in parts["leaderboard"]
    assert "Cross-validation on development data (DEV) only" in parts["leaderboard"]
    assert "0.8474" in parts["leaderboard"]


@pytest.mark.unit
def test_absent_holdout_and_explanation_are_stated(tmp_path: Path) -> None:
    s = state(holdout=HoldoutResult(present=False, reason="dataset too small"),
              explanation=ExplanationResult(method="none", unavailable_reason="SHAP unavailable"),
              errors=None)  # fmt: skip
    html = build_report(s, tmp_path / "r").html.read_text(encoding="utf-8")
    parts = sections_of(html)
    assert "No independent test results exist: dataset too small." in parts["independent-test"]
    assert "Explanation unavailable: SHAP unavailable" in parts["importance"]
    assert "Error analysis is unavailable." in parts["errors"]


def injection_names() -> list[str]:
    header = INJECTION.read_text(encoding="utf-8").splitlines()[0]
    import csv  # noqa: PLC0415

    return next(csv.reader([header]))


@pytest.mark.unit
def test_T_SEC_02_injection_column_names_are_inert(tmp_path: Path) -> None:
    names = injection_names()
    s = state()
    meta = s.metadata
    columns = [meta.columns[0].model_copy(update={"original": name, "exclusion_reason": "test"})
               for name in names]  # fmt: skip
    source = meta.source.model_copy(update={"display_name": "<b>x</b>.csv"})
    s.metadata = meta.model_copy(update={"columns": [*meta.columns, *columns], "source": source})
    s.errors = ErrorAnalysis(task=Task.BINARY, error_name="error rate", overall_error=0.1,
                             segments=[Segment("contract", 1, "<script>alert(2)</script>", 30, 0.5)])  # fmt: skip
    paths = build_report(s, tmp_path / "r")
    html = paths.html.read_text(encoding="utf-8")
    md = paths.markdown.read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "<b>x</b>" not in html
    assert "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;" in html
    assert "{{ config.__class__ }}" in html  # data, never evaluated as a template
    assert "<script>" not in md
    assert "\\<script\\>" in md
    assert "\\{\\{ config.\\_\\_class\\_\\_ \\}\\}" in md
    assert "|safe" not in (Path(__file__).resolve().parents[2] / "src" / "tabpilot" / "reporting"
                           / "templates" / "report.html.j2").read_text(encoding="utf-8")  # fmt: skip


@pytest.mark.unit
def test_md_escape() -> None:
    assert md_escape("a|b") == "a\\|b"
    assert md_escape("[x](javascript:alert(1))") == "\\[x\\](javascript:alert(1))"
    assert md_escape("line1\nline2") == "line1 line2"
    assert md_escape("<i>&") == "\\<i\\>\\&"


@pytest.mark.unit
def test_render_is_deterministic() -> None:
    s = state()
    narrative = template_narrative(build_facts(s))
    assert render_report(s, narrative, {}) == render_report(s, narrative, {})


# ---------------------------------------------------------------- figures
@pytest.mark.unit
def test_figures_for_each_task(tmp_path: Path) -> None:
    s = state()
    binary = render_figures(entries=s.entries, explanation=s.explanation, errors=s.errors,
                            oof=s.winner_oof, y=s.y_dev, problem=BIN, out_dir=tmp_path / "b")  # fmt: skip
    assert set(binary) == {"leaderboard", "fold_scores", "shap_importance", "shap_beeswarm_1",
                           "gain_importance", "confusion_matrix", "roc_curve", "pr_curve",
                           "reliability"}  # fmt: skip
    rng = np.random.default_rng(0)
    actual = rng.normal(size=200)
    reg = ErrorAnalysis(task=Task.REGRESSION, error_name="mean absolute error",
                        overall_error=0.5, actual=actual, predicted=actual + rng.normal(size=200))  # fmt: skip
    dollars = ExplanationResult(method="permutation",
                                permutation=[Importance("price $ usd $", 0.3, 0.1)])  # fmt: skip
    regression = render_figures(entries=s.entries, explanation=dollars, errors=reg, oof=None,
                                y=None, problem=Problem(Task.REGRESSION), out_dir=tmp_path / "r")  # fmt: skip
    assert {"permutation_importance", "residuals_vs_predicted", "residual_histogram",
            "actual_vs_predicted"} <= set(regression)  # fmt: skip
    multi = Problem(Task.MULTICLASS, ("a", "b", "c"))
    proba = rng.dirichlet([1, 1, 1], size=90)
    out = render_figures(entries=[], explanation=None, errors=None, oof=proba,
                         y=np.repeat([0, 1, 2], 30), problem=multi, out_dir=tmp_path / "m")  # fmt: skip
    assert set(out) == {"reliability"}
    assert all((tmp_path / "m" / Path(p).name).exists() for p in out.values())


@pytest.mark.unit
def test_rich_run_renders_every_optional_part(tmp_path: Path) -> None:
    """Profile, plan validation, LLM features, reflection, ensemble, thresholds, ledger,
    failed and non-comparable entries, regression error analysis, partial status.
    """
    ev = EvaluationResult.model_validate(contract("prd_26_9_EvaluationResult"))
    failed = ev.entries[1].model_copy(update={"entry_id": "0-xgboost-00000001", "model": "xgboost",
                                              "status": "failed", "mean": None, "std": None,
                                              "fold_scores": []})  # fmt: skip
    odd = ev.entries[1].model_copy(update={"entry_id": "0-catboost-00000002", "model": "catboost",
                                           "comparable": False})  # fmt: skip
    profile = ProfileSummary.model_validate(contract("ProfileSummary"))
    profile = profile.model_copy(update={"leakage": [LeakageFinding(
        column="totalcharges", score=0.99, metric="auc", level="near_deterministic")]})  # fmt: skip
    cfg = ExperimentConfig.model_validate(contract("prd_26_7_ExperimentConfig"))
    rng = np.random.default_rng(1)
    actual = rng.normal(size=120)
    errors = error_analysis(actual + rng.normal(size=120), actual,
                            pd.DataFrame({"region": ["north"] * 60 + ["south"] * 60}),
                            Problem(Task.REGRESSION), ["region"])  # fmt: skip
    s = state(
        entries=[*ev.entries, failed, odd], profile=profile, plan_source="llm",
        plan_validation=PlanValidation.model_validate(contract("PlanValidation")),
        provenance=[FeatureProvenance.model_validate(contract("FeatureProvenance"))],
        reflections=[ReflectionRecord.model_validate(contract("ReflectionRecord"))],
        ensemble=EnsembleResult.model_validate(contract("EnsembleResult")),
        thresholds={"0-lightgbm-3c9e1a20": {"threshold_0_5": {"threshold": 0.5, "accuracy": 0.8,
                                                              "f1": 0.6}}},
        ledger=BudgetLedger.model_validate(contract("BudgetLedger")),
        config=cfg.model_copy(update={"prompt_template_versions": {"planner_system": "1.0"}}),
        winner=CandidateSpec.model_validate(contract("CandidateSpec")),
        errors=errors, status="completed_partial", warnings=["W-TEST"],
    )  # fmt: skip
    paths = build_report(s, tmp_path / "r")
    parts = sections_of(paths.html.read_text(encoding="utf-8"))
    assert "near_deterministic" in parts["warnings"]
    assert "W-TEST" in parts["warnings"]
    assert "Plan source: LLM planner." in parts["plan"]
    assert "repaired" in parts["plan"]
    assert "avg_monthly_spend" in parts["features"]
    assert "no change met the paired-fold rule" in parts["reflection"]
    assert "Ensemble kept: yes" in parts["ensemble"]
    assert "Not comparable" in parts["leaderboard"]
    assert "0-xgboost-00000001" in parts["leaderboard"]
    assert "Threshold metrics" in parts["leaderboard"]
    assert "Mean absolute error" in parts["errors"]
    assert "north" in parts["errors"]  # labels render locally, escaped
    assert "planner_system@1.0" in parts["reproducibility"]
    assert "completed_partial" in parts["limitations"]
    facts = build_facts(s)
    assert facts.by_key("n_models_failed") is not None
    assert facts.by_key("llm_features_active") is not None
    assert facts.by_key("reflection_rounds") is not None
    assert facts.by_key("ensemble_members") is not None
    assert "north" not in json.dumps(facts.payload())  # PI-16
