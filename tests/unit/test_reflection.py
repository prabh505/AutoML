"""P12 reflection tests: T-RFL-01, T-RFL-03…05, proposals, inputs, budget, stopping (PRD §19)."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from tests.helpers.llm_stub import ScriptedLLMProvider
from tests.helpers.synthetic import synthetic

from tabpilot.config import load_settings
from tabpilot.evaluation.metrics import spec as metric_spec
from tabpilot.llm import BudgetTracker, ConsentStore, LLMRouter, estimate_tokens
from tabpilot.models import ModelRegistry
from tabpilot.planning import PlanContext
from tabpilot.preprocessing import build_pipeline
from tabpilot.reflection import (
    RoundOutcome,
    accept,
    apply_changes,
    build_reflection_input,
    change_set_id,
    load,
    paired_decision,
    propose,
    raw_importances,
    reflection_budget,
    retrain_parents,
    should_stop,
    source_column,
    source_importances,
    validate_proposal,
    versions,
)
from tabpilot.reflection.propose import ReflectionContext
from tabpilot.schemas import (
    FEATURE_SPEC_ADAPTER,
    CandidateSpec,
    ClassWeight,
    LLMBudget,
    MetricName,
    ModelName,
    PlannerOutput,
    ProviderName,
    ReflectionProposal,
    TargetTransform,
    Task,
)
from tabpilot.schemas.experiment import LeaderboardEntry, ModelConfig, fold_statistics
from tabpilot.schemas.features import FeatureProvenance
from tabpilot.schemas.reflection import FeatureImportance, RemainingBudget

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
SPLIT = "a" * 64


def entry(eid: str, scores: list[float], *, model: str = "lightgbm", metric: MetricName = MetricName.ROC_AUC,
          status: str = "ok", comparable: bool = True, split_id: str = SPLIT, iteration: int = 0,
          feature_set: str = "baseline") -> LeaderboardEntry:  # fmt: skip
    stats: dict[str, Any] = {"mean": None, "std": None, "sem": None}
    if status == "ok":
        m, s, e = fold_statistics(scores)
        stats = {"mean": m, "std": s, "sem": e}
    return LeaderboardEntry(entry_id=eid, experiment_id=RID, iteration=iteration, model=model,
                            feature_set=feature_set, metric=metric_spec(metric),
                            fold_scores=scores if status == "ok" else [], status=status,
                            comparable=comparable, split_id=split_id, **stats)  # fmt: skip


# ---------------------------------------------------------------- T-RFL-01 acceptance
def decide(parent: list[float], child: list[float], metric: MetricName = MetricName.ROC_AUC) -> Any:
    return paired_decision(parent_id="p", child_id="c", parent_scores=parent, child_scores=child,
                           metric=metric)  # fmt: skip


@pytest.mark.unit
def test_T_RFL_01_clear_improvement_is_accepted() -> None:
    d = decide([0.80] * 5, [0.81, 0.81, 0.81, 0.80, 0.80])
    assert d.accepted
    assert d.folds_better == 3
    assert d.folds_required == 3
    assert d.rel_gain == pytest.approx(0.006 / 0.80)


@pytest.mark.unit
def test_T_RFL_01_noisy_improvement_on_two_of_five_folds_is_rejected() -> None:
    d = decide([0.80] * 5, [0.90, 0.90, 0.79, 0.79, 0.79])  # big mean gain, but only 2/5 folds
    assert not d.accepted
    assert "better in 2 of 5 folds" in d.reason


@pytest.mark.unit
def test_T_RFL_01_threshold_is_inclusive_and_relative() -> None:
    parent = [1.0] * 5
    exactly = decide(parent, [1.002] * 5)
    assert exactly.rel_gain == pytest.approx(0.002)
    assert exactly.accepted
    below = decide(parent, [1.0019] * 5)
    assert not below.accepted
    assert "below 0.2%" in below.reason


@pytest.mark.unit
@pytest.mark.parametrize(
    ("k", "better", "accepted"), [(5, 3, True), (5, 2, False), (15, 9, True), (15, 8, False)]
)
def test_T_RFL_01_majority_boundary(k: int, better: int, accepted: bool) -> None:
    parent = [0.5] * k
    child = [0.6] * better + [0.4999] * (k - better)
    d = decide(parent, child)
    assert d.folds_required == math.ceil(0.6 * k)
    assert d.folds_better == better
    assert d.accepted is accepted


@pytest.mark.unit
def test_T_RFL_01_loss_metrics_use_selection_direction() -> None:
    better = decide([0.50] * 5, [0.49] * 5, MetricName.LOG_LOSS)  # lower loss is better
    assert better.accepted
    assert all(x > 0 for x in better.fold_deltas)
    assert better.rel_gain == pytest.approx(0.01 / 0.50)  # relative to the loss magnitude
    worse = decide([0.50] * 5, [0.51] * 5, MetricName.LOG_LOSS)
    assert not worse.accepted
    assert worse.folds_better == 0
    rmse = decide([20.0] * 5, [19.9] * 5, MetricName.RMSE)
    assert rmse.accepted
    assert rmse.rel_gain == pytest.approx(0.1 / 20)


@pytest.mark.unit
def test_T_RFL_05_regression_and_unusable_children_keep_the_incumbent() -> None:
    parent = entry("0-lightgbm-aaaaaaaa", [0.80] * 5)
    worse = entry("1-lightgbm-bbbbbbbb", [0.79] * 5, iteration=1)
    assert not accept(parent, worse).accepted
    for child in (entry("1-lightgbm-cccccccc", [], status="failed", iteration=1),
                  entry("1-lightgbm-dddddddd", [0.99] * 5, comparable=False, iteration=1),
                  entry("1-lightgbm-eeeeeeee", [0.99] * 5, split_id="b" * 64, iteration=1),
                  entry("1-lightgbm-ffffffff", [0.99] * 3, iteration=1)):  # fmt: skip
        d = accept(parent, child)
        assert not d.accepted
        assert d.reason == "challenger is not ok or not comparable"
        assert d.fold_deltas == [0.0] * 5
    assert accept(parent, entry("1-lightgbm-99999999", [0.81] * 5, iteration=1)).accepted
    with pytest.raises(ValueError, match="same non-empty folds"):
        decide([0.8] * 5, [0.8] * 4)


# ---------------------------------------------------------------- T-RFL-04 budget, stopping
@pytest.mark.unit
def test_T_RFL_04_budget_per_round_and_skip() -> None:
    fresh = reflection_budget(total_s=1000, elapsed_s=300, reflection_used_s=0, initial_trials=40)
    assert (fresh.seconds, fresh.trials, fresh.skipped) == (150.0, 12, False)  # 15% share caps it
    second = reflection_budget(1000, 450, reflection_used_s=100, initial_trials=40)
    assert second.seconds == 50.0  # reflection share left
    near_stop = reflection_budget(1000, 560, 0, 40)  # 140 s left before 70% < 15% of total
    assert near_stop.skipped
    assert near_stop.skip_reason == "budget"
    assert near_stop.seconds == 0
    used_up = reflection_budget(1000, 300, reflection_used_s=150, initial_trials=40)
    assert used_up.skipped
    assert reflection_budget(1000, 300, 0, initial_trials=1).trials == 1


@pytest.mark.unit
@pytest.mark.parametrize(("outcome", "reason"), [
    (RoundOutcome(2, 2, accepted=True), "max rounds reached"),
    (RoundOutcome(1, 2, budget_skipped=True), "budget below minimum"),
    (RoundOutcome(1, 2, llm_unavailable=True), "LLM unavailable or LLM budget exhausted"),
    (RoundOutcome(1, 2, proposal_invalid=True), "proposal invalid after one repair"),
    (RoundOutcome(1, 2, proposal_empty=True), "proposal empty"),
    (RoundOutcome(1, 2, accepted=False), "no change accepted this round"),
    (RoundOutcome(1, 2, accepted=True), None),
])  # fmt: skip
def test_should_stop_rules(outcome: RoundOutcome, reason: str | None) -> None:
    assert should_stop(outcome) == (reason is not None, reason)


# ---------------------------------------------------------------- inputs
@pytest.mark.unit
@pytest.mark.parametrize(("column", "expected"), [
    ("lowcat__contract_one_year", "contract"), ("num__when__year", "when"),
    ("bool__amount__missing", "amount"), ("num__llm__ratio_a", "ratio_a"),
    ("highcat__high", "high"), ("num__unknown", None), ("num__amount", "amount"),
])  # fmt: skip
def test_source_column_mapping(column: str, expected: str | None) -> None:
    sources = ["contract", "when", "amount", "high", "ratio_a", "contract_type"]
    assert source_column(column, sources) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "model", [ModelName.LIGHTGBM, ModelName.LINEAR, ModelName.RANDOM_FOREST, ModelName.CATBOOST]
)
def test_importances_fold_into_source_columns_without_labels(model: ModelName) -> None:
    frame, y, schema = synthetic()
    cand = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=model, task=Task.BINARY, seed=1, n_jobs=1,
                                           params={"iterations": 50} if model == ModelName.CATBOOST else {}))  # fmt: skip
    pipe = build_pipeline(cand, schema, ModelRegistry(load_settings({}))).fit(frame, y)
    columns = list(pipe.named_steps["model"].feature_names_in_)
    raw = raw_importances(pipe.named_steps["model"], columns)
    assert raw
    imps = source_importances(raw, list(schema.columns))
    names = {i.name for i in imps}
    assert names <= set(schema.columns)  # one-hot columns like lowcat__low_a never appear (PI-16)
    assert sum(i.importance for i in imps) == pytest.approx(1.0)
    assert raw_importances(object(), columns) == {}
    assert raw_importances(pipe.named_steps["model"], ["x"]) == {}
    assert source_importances({"num__x": 0.0}, ["x"]) == []


def plan() -> PlannerOutput:
    return PlannerOutput.model_validate({
        "models": [{"name": "lightgbm", "priority": 1}, {"name": "linear", "priority": 2},
                   {"name": "random_forest", "priority": 3}],
        "tuning_allocation": {"lightgbm": 0.6, "linear": 0.15, "random_forest": 0.25},
        "rationale": "secret rationale",
    })  # fmt: skip


@pytest.mark.unit
def test_build_reflection_input_top_entries_and_no_rationale() -> None:
    entries = [entry(f"0-lightgbm-0000000{i}", [0.7 + i / 100] * 5) for i in range(7)]
    entries.append(entry("0-x-ffffffff", [], status="failed"))
    imps = [FeatureImportance(name=f"c{i}", importance=1 / 20) for i in range(20)]
    prov = [FeatureProvenance(id="b1", name="age", source="baseline", iteration=0, status="active"),
            FeatureProvenance(id="f1", name="ratio", source="llm_plan", iteration=0, status="active")]  # fmt: skip
    ri = build_reflection_input(task=Task.BINARY, metric=MetricName.ROC_AUC, entries=entries,
                                importances=imps, provenance=prov, plan=plan(),
                                remaining=RemainingBudget(seconds=100, trials=5, features_allowed=6),
                                warnings=["W-SMALL-DATA"])  # fmt: skip
    assert [r.entry_id for r in ri.leaderboard_top] == [
        f"0-lightgbm-0000000{i}" for i in (6, 5, 4, 3, 2)
    ]
    assert len(ri.top_features) == 15
    assert [p.name for p in ri.llm_features] == ["ratio"]
    assert ri.current_plan.rationale == ""
    text = ri.model_dump_json()
    assert "secret rationale" not in text
    assert "holdout" not in text


# ---------------------------------------------------------------- proposals (T-RFL-03)
def ctx(provider: Any = None, active: list[dict[str, Any]] | None = None, task: Task = Task.BINARY,
        target_min: float | None = None, **budget_kw: Any) -> ReflectionContext:  # fmt: skip
    _, _, schema = synthetic()
    pc = PlanContext(task=task, target="y", primary_metric=MetricName.ROC_AUC if task != Task.REGRESSION
                     else MetricName.RMSE, feature_types=dict(schema.columns),
                     available_models=[ModelName.LINEAR, ModelName.RANDOM_FOREST, ModelName.LIGHTGBM],
                     k=5, excluded=frozenset({"customer_id"}), target_min=target_min)  # fmt: skip
    specs = [FEATURE_SPEC_ADAPTER.validate_python(f) for f in active or []]
    ri = build_reflection_input(task=task, metric=pc.primary_metric, entries=[], importances=[],
                                provenance=[], plan=plan(),
                                remaining=RemainingBudget(seconds=100, trials=5, features_allowed=6))  # fmt: skip
    router = None
    if provider is not None:
        router = LLMRouter([provider], consent=ConsentStore.granted_for(list(ProviderName)))
    budget = BudgetTracker(LLMBudget(**{"max_wall_s": 100.0, **budget_kw}))
    return ReflectionContext(ri, pc, specs, router, budget)


def proposal(*changes: dict[str, Any], **kw: Any) -> ReflectionProposal:
    return ReflectionProposal.model_validate({"changes": list(changes), **kw})


def feat(
    name: str, op: str = "square", inputs: list[str] | None = None, **params: Any
) -> dict[str, Any]:
    return {"id": name[:16], "name": name, "op": op, "inputs": inputs or ["z"], "params": params}


@pytest.mark.unit
def test_T_RFL_03_prohibited_fields_are_not_in_the_schema() -> None:
    for key in ("task", "metric", "folds", "holdout", "seeds", "budget", "target"):
        with pytest.raises(ValueError, match="Extra inputs"):
            ReflectionProposal.model_validate({"changes": [], key: "x"})
    with pytest.raises(ValueError, match="at most 6 items"):
        proposal({"type": "add_features", "features": [feat(f"f{i}") for i in range(7)]})
    with pytest.raises(ValueError, match="does not match any of the expected tags"):
        proposal({"type": "request_holdout"})


@pytest.mark.unit
def test_T_RFL_03_validator_removes_out_of_bounds_changes() -> None:
    c = ctx(active=[feat("sq_z")])
    p = proposal(
        {"type": "add_features", "features": [feat("ok_feat", inputs=["amount"]),
                                              feat("bad", "ratio", ["low", "z"]),
                                              feat("evil", "ratio", ["http://x", "z"]),
                                              feat("sq_z")]},  # duplicate of an active feature
        {"type": "remove_features", "feature_names": ["sq_z", "ghost"]},
        {"type": "change_encoding", "columns": ["low", "nope"], "encoding": "ordinal"},
        {"type": "toggle_class_weight", "model": "xgboost", "value": "balanced"},
        {"type": "change_target_transform", "value": "log1p"},
        {"type": "reallocate_tuning", "allocation": {"lightgbm": 0.5, "catboost": 0.5}},
        {"type": "drop_model", "model": "catboost"},
    )  # fmt: skip
    validated, issues = validate_proposal(p, c)
    kinds = [ch.type for ch in validated.changes]
    assert kinds == ["add_features", "remove_features", "change_encoding", "reallocate_tuning"]
    assert [f.name for f in validated.changes[0].features] == ["ok_feat"]
    assert validated.changes[1].feature_names == ["sq_z"]
    assert validated.changes[2].columns == ["low"]
    assert validated.changes[3].allocation == {ModelName.LIGHTGBM: 0.5}
    text = " | ".join(issues)
    for expected in ("feature_type_mismatch", "unsafe_feature", "unknown feature names", "unknown columns",
                     "classification models in the plan only", "regression with a non-negative target only",
                     "models outside the plan", "must name a plan model"):  # fmt: skip
        assert expected in text


@pytest.mark.unit
def test_feature_cap_of_twelve_holds_with_reflection() -> None:
    eleven = [feat(f"a{i}", "clip", ["z"], lower_q=0.01 * (i % 10), upper_q=0.9) for i in range(11)]
    c = ctx(active=eleven)
    validated, issues = validate_proposal(proposal({"type": "add_features", "features": [
        feat("n1", inputs=["amount"]), feat("n2", "log1p", ["amount"])]}), c)  # fmt: skip
    assert [f.name for f in validated.changes[0].features] == ["n1"]  # 11 + 1 = 12 (A-010)
    assert any("capped" in i for i in issues)


@pytest.mark.unit
def test_dropping_every_model_keeps_one() -> None:
    p = proposal(
        *[{"type": "drop_model", "model": m} for m in ("lightgbm", "linear", "random_forest")]
    )
    validated, issues = validate_proposal(p, ctx())
    assert [c.model for c in validated.changes] == [ModelName.LIGHTGBM, ModelName.LINEAR]
    assert any("at least one model must remain" in i for i in issues)


@pytest.mark.unit
def test_regression_target_transform_and_class_weight_rules() -> None:
    reg = ctx(task=Task.REGRESSION, target_min=0.0)
    ok, _ = validate_proposal(proposal({"type": "change_target_transform", "value": "log1p"},
                                       {"type": "toggle_class_weight", "model": "linear", "value": "balanced"}), reg)  # fmt: skip
    assert [c.type for c in ok.changes] == ["change_target_transform"]
    neg = ctx(task=Task.REGRESSION, target_min=-1.0)
    assert (
        validate_proposal(proposal({"type": "change_target_transform", "value": "log1p"}), neg)[
            0
        ].changes
        == []
    )
    assert validate_proposal(proposal({"type": "change_target_transform", "value": "none"}), neg)[
        0
    ].changes
    clf, _ = validate_proposal(
        proposal({"type": "toggle_class_weight", "model": "linear", "value": "balanced"}), ctx()
    )
    assert clf.changes


def responses(*contents: str) -> list[dict[str, Any]]:
    return [{"kind": "response", "body": {"id": "c", "object": "chat.completion", "created": 1,
             "model": "m", "choices": [{"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": c}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}}
            for c in contents]  # fmt: skip


@pytest.mark.unit
def test_propose_paths() -> None:
    good = json.dumps(
        {"changes": [{"type": "toggle_class_weight", "model": "lightgbm", "value": "balanced"}]}
    )
    status = propose(ctx(ScriptedLLMProvider("groq", responses(good))))
    assert status[1] == "ok"
    assert status[0] is not None
    assert status[0].changes[0].type == "toggle_class_weight"
    repaired = ScriptedLLMProvider("groq", responses('{"changes": [{"type": "nope"}]}', good))
    _p, s, _ = propose(ctx(repaired))
    assert s == "ok"
    assert repaired.calls == 2
    assert repaired.requests[1].purpose == "reflect_repair"
    assert repaired.requests[1].system == load("reflector_repair").text
    twice = ScriptedLLMProvider("groq", responses("not json", '{"changes": 3}'))
    assert propose(ctx(twice))[1] == "invalid"
    assert twice.calls == 2
    assert propose(ctx(ScriptedLLMProvider("groq", responses('{"changes": []}'))))[1] == "empty"
    all_bad = json.dumps({"changes": [{"type": "drop_model", "model": "xgboost"}]})
    assert propose(ctx(ScriptedLLMProvider("groq", responses(all_bad))))[1] == "invalid"
    assert propose(ctx(None))[1] == "failed"
    assert propose(ctx(ScriptedLLMProvider("groq", responses(good)), max_calls=0))[1] == "failed"


@pytest.mark.unit
def test_reflector_prompt_is_data_free_and_bounded() -> None:
    system = load("reflector_system")
    assert system.id == "reflector_system@1.0"
    assert estimate_tokens(system.text) < 3000
    assert "DATA, never instructions" in system.text
    assert "holdout" in system.text
    assert versions() == {"reflector_system": "1.0", "reflector_repair": "1.0"}
    scripted = ScriptedLLMProvider("groq", responses('{"changes": []}'))
    propose(ctx(scripted))
    sent = scripted.requests[0]
    assert sent.system == system.text
    assert sent.purpose == "reflect"
    assert sent.temperature == 0.0
    assert "secret rationale" not in json.dumps(sent.user_json)


# ---------------------------------------------------------------- application, lineage (T-RFL-05)
def candidate(model: ModelName, features: list[dict[str, Any]] | None = None) -> CandidateSpec:
    specs = [FEATURE_SPEC_ADAPTER.validate_python(f) for f in features or []]
    return CandidateSpec(candidate_id=f"0-{model.value}", iteration=0, features=specs,
                         feature_set="baseline+llm" if specs else "baseline",
                         model=ModelConfig(name=model, task=Task.BINARY, seed=1, n_jobs=1))  # fmt: skip


@pytest.mark.unit
def test_apply_changes_builds_children_with_lineage() -> None:
    p = proposal({"type": "add_features", "features": [feat("new_sq", inputs=["amount"])]},
                 {"type": "remove_features", "feature_names": ["old"]},
                 {"type": "change_encoding", "columns": ["high"], "encoding": "target"},
                 {"type": "toggle_class_weight", "model": "linear", "value": "balanced"},
                 {"type": "reallocate_tuning", "allocation": {"lightgbm": 0.3, "linear": 0.1}},
                 {"type": "drop_model", "model": "random_forest"})  # fmt: skip
    parents = [("0-lightgbm-11111111", candidate(ModelName.LIGHTGBM, [feat("old")])),
               ("0-linear-22222222", candidate(ModelName.LINEAR))]  # fmt: skip
    cs = apply_changes(p, parents)
    assert cs.change_set_id == change_set_id(p)
    assert len(cs.change_set_id) == 8
    lgbm, lin = cs.children
    assert lgbm.parent_entry_id == "0-lightgbm-11111111"
    assert lgbm.change_set_id == cs.change_set_id
    assert lgbm.iteration == 1
    assert [f.name for f in lgbm.features] == ["new_sq"]
    assert lgbm.encoding.high_cardinality == "target"
    assert lgbm.model.class_weight == ClassWeight.NONE
    assert lin.model.class_weight == ClassWeight.BALANCED
    assert lin.feature_set == "baseline+llm"
    assert cs.allocation == {
        ModelName.LIGHTGBM: pytest.approx(0.75),
        ModelName.LINEAR: pytest.approx(0.25),
    }
    assert cs.dropped_models == [ModelName.RANDOM_FOREST]
    noop = apply_changes(proposal({"type": "toggle_class_weight", "model": "linear", "value": "none"}),
                         [("0-lightgbm-11111111", candidate(ModelName.LIGHTGBM))])  # fmt: skip
    assert noop.children == []  # nothing changes for this candidate
    reg = CandidateSpec(candidate_id="r", iteration=1, feature_set="baseline",
                        model=ModelConfig(name=ModelName.LINEAR, task=Task.REGRESSION, seed=1, n_jobs=1))  # fmt: skip
    child = apply_changes(
        proposal({"type": "change_target_transform", "value": "log1p"}),
        [("1-linear-33333333", reg)],
    )
    assert child.children[0].target_transform == TargetTransform.LOG1P
    assert child.children[0].iteration == 2


@pytest.mark.unit
def test_retrain_parents_top_two_families_plus_targeted() -> None:
    entries = [entry("0-lightgbm-00000001", [0.90] * 5), entry("0-lightgbm-00000002", [0.85] * 5),
               entry("0-random_forest-00000003", [0.88] * 5, model="random_forest"),
               entry("0-linear-00000004", [0.70] * 5, model="linear"),
               entry("0-xgboost-00000005", [], model="xgboost", status="failed")]  # fmt: skip
    base = retrain_parents(entries, MetricName.ROC_AUC, proposal())
    assert [e.entry_id for e in base] == ["0-lightgbm-00000001", "0-random_forest-00000003"]
    targeted = retrain_parents(entries, MetricName.ROC_AUC, proposal(
        {"type": "toggle_class_weight", "model": "linear", "value": "balanced"},
        {"type": "drop_model", "model": "random_forest"}))  # fmt: skip
    assert [e.entry_id for e in targeted] == ["0-lightgbm-00000001", "0-linear-00000004"]
