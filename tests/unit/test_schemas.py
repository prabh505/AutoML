"""P02 contract tests: round-trip, PRD examples, extra keys, bounds, discriminators (PRD §26)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from tabpilot.schemas import (
    CONTRACTS,
    FEATURE_SPEC_ADAPTER,
    UNION_CONTRACTS,
    ColumnInfo,
    ExperimentConfig,
    FinalResult,
    HoldoutResult,
    LeaderboardEntry,
    LLMRequest,
    MetricSpec,
    PlannerOutput,
    ReflectionInput,
    ReflectionProposal,
    RunConfig,
    RunResult,
    Seeds,
    SplitSpec,
    TrainingResult,
)
from tabpilot.schemas.features import (
    BinQuantileSpec,
    ClipSpec,
    EqualsSpec,
    GroupCountSpec,
    RatioSpec,
    RowAggregateSpec,
)
from tabpilot.schemas.report import NarrativeSection

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"
PRD_EXAMPLES = {
    "26.1": "DatasetMetadata", "26.2": "CompactProfile", "26.3": "PlannerInput",
    "26.4": "PlannerOutput", "26.6": "ModelConfig", "26.7": "ExperimentConfig",
    "26.8": "TrainingResult", "26.9": "EvaluationResult", "26.10": "ReflectionProposal",
    "26.11": "FinalReport", "26.12": "ArtifactManifest", "26.13": "RunEvent",
}  # fmt: skip
ALL = {**CONTRACTS, **UNION_CONTRACTS}


def fixture(name: str) -> dict[str, Any]:
    matches = sorted(FIXTURES.glob(f"*{name}.json"))
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def validate(name: str, data: Any) -> Any:
    target = ALL[name]
    if isinstance(target, TypeAdapter):
        return target.validate_python(data)
    return target.model_validate(data)


def mutate(name: str, **changes: Any) -> dict[str, Any]:
    return {**fixture(name), **changes}


@pytest.mark.unit
def test_every_contract_has_a_fixture() -> None:
    names = {p.stem.split("_")[-1] for p in FIXTURES.glob("*.json")}
    assert set(ALL) <= names


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(ALL))
def test_round_trip(name: str) -> None:
    """dump → load → equal, in Python and JSON modes."""
    obj = validate(name, fixture(name))
    if isinstance(obj, BaseModel):
        assert type(obj).model_validate(obj.model_dump()) == obj
        assert type(obj).model_validate_json(obj.model_dump_json()) == obj
    else:
        adapter = ALL[name]
        assert isinstance(adapter, TypeAdapter)
        assert adapter.validate_json(adapter.dump_json(obj)) == obj


@pytest.mark.unit
@pytest.mark.parametrize(("section", "name"), sorted(PRD_EXAMPLES.items()))
def test_prd_examples_validate(section: str, name: str) -> None:
    path = FIXTURES / f"prd_{section.replace('.', '_')}_{name}.json"
    assert CONTRACTS[name].model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(ALL))
def test_extra_top_level_key_rejected(name: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate(name, {**fixture(name), "unexpected": 1})


@pytest.mark.unit
def test_llm_plan_with_one_extra_nested_key_fails() -> None:
    """A malformed LLM plan with one extra key anywhere must fail (P02 self-review)."""
    plan = fixture("PlannerOutput")
    plan["imputation"] = {**plan["imputation"], "strategy_hint": "x"}
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(plan)
    plan = fixture("PlannerOutput")
    plan["features"][0]["params"]["scale"] = 2.0
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(plan)


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_contracts_are_frozen_except_run_state(name: str) -> None:
    obj = CONTRACTS[name].model_validate(fixture(name))
    field = next(iter(CONTRACTS[name].model_fields))
    if name == "RunState":
        obj.iteration = 1
        with pytest.raises(ValidationError):
            obj.iteration = 3  # validate_assignment enforces bounds
    else:
        with pytest.raises(ValidationError):
            setattr(obj, field, getattr(obj, field))


@pytest.mark.unit
def test_column_info_is_versioned_through_dataset_metadata() -> None:
    assert "schema_version" not in ColumnInfo.model_fields
    assert "schema_version" in CONTRACTS["DatasetMetadata"].model_fields


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(set(CONTRACTS) - {"ColumnInfo"}))
def test_schema_version_literal(name: str) -> None:
    model = CONTRACTS[name]
    version = model.model_fields["schema_version"].default
    assert isinstance(version, str)
    assert version.endswith("/1.0")
    with pytest.raises(ValidationError):
        model.model_validate({**fixture(name), "schema_version": "x/9.9"})


# ---------------------------------------------------------------- plan bounds (§12.3)
def _feature(i: int) -> dict[str, Any]:
    return {"id": f"f{i:02d}", "name": f"feat_{i}", "op": "square", "inputs": ["tenure"]}


@pytest.mark.unit
@pytest.mark.parametrize(("n", "ok"), [(0, True), (12, True), (13, False)])
def test_plan_feature_count(n: int, ok: bool) -> None:
    data = mutate("PlannerOutput", features=[_feature(i) for i in range(n)])
    if ok:
        PlannerOutput.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            PlannerOutput.model_validate(data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "length", "ok"), [("rationale", 800, True), ("rationale", 801, False)]
)
def test_plan_rationale_length(field: str, length: int, ok: bool) -> None:
    data = mutate("PlannerOutput", **{field: "x" * length})
    if ok:
        PlannerOutput.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            PlannerOutput.model_validate(data)


@pytest.mark.unit
@pytest.mark.parametrize(("length", "ok"), [(200, True), (201, False)])
def test_column_decision_reason_length(length: int, ok: bool) -> None:
    data = fixture("PlannerOutput")
    data["column_decisions"][0]["reason"] = "r" * length
    if ok:
        PlannerOutput.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            PlannerOutput.model_validate(data)


MODELS = ["linear", "random_forest", "lightgbm", "xgboost", "catboost", "hist_gb"]


@pytest.mark.unit
@pytest.mark.parametrize(("n", "ok"), [(0, False), (1, True), (5, True), (6, False)])
def test_plan_model_count(n: int, ok: bool) -> None:
    models = [{"name": m, "priority": min(i + 1, 5)} for i, m in enumerate(MODELS[:n])]
    data = mutate("PlannerOutput", models=models, tuning_allocation={})
    if ok:
        PlannerOutput.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            PlannerOutput.model_validate(data)


@pytest.mark.unit
@pytest.mark.parametrize(("folds", "ok"), [(3, True), (5, True), (4, False), (10, False)])
def test_plan_cv_folds(folds: int, ok: bool) -> None:
    data = mutate("PlannerOutput", cv_folds=folds)
    if ok:
        PlannerOutput.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            PlannerOutput.model_validate(data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("alloc", "ok"),
    [({"lightgbm": 0.5, "linear": 0.5}, True), ({"lightgbm": 0.8, "linear": 0.3}, False),
     ({"lightgbm": -0.1}, False), ({"gpt": 1.0}, False)],
)  # fmt: skip
def test_plan_tuning_allocation(alloc: dict[str, float], ok: bool) -> None:
    data = mutate("PlannerOutput", tuning_allocation=alloc)
    if ok:
        PlannerOutput.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            PlannerOutput.model_validate(data)


@pytest.mark.unit
def test_plan_rejects_unknown_model_and_duplicate_models() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(
            mutate("PlannerOutput", models=[{"name": "gpt", "priority": 1}])
        )
    dup = [{"name": "linear", "priority": 1}, {"name": "linear", "priority": 2}]
    with pytest.raises(ValidationError, match="unique"):
        PlannerOutput.model_validate(mutate("PlannerOutput", models=dup, tuning_allocation={}))


# ---------------------------------------------------------------- feature DSL (§14.3)
def spec(op: str, inputs: list[str], params: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"id": "f01", "name": "feat", "op": op, "inputs": inputs}
    if params is not None:
        data["params"] = params
    return data


@pytest.mark.unit
@pytest.mark.parametrize(
    ("op", "params", "cls"),
    [("ratio", {"eps": 1e-6}, RatioSpec), ("clip", {"lower_q": 0.0, "upper_q": 1.0}, ClipSpec),
     ("bin_quantile", {"n_bins": 5}, BinQuantileSpec), ("equals", {"value": 1}, EqualsSpec),
     ("row_mean", None, RowAggregateSpec), ("group_count", None, GroupCountSpec)],
)  # fmt: skip
def test_discriminator_selects_param_model(
    op: str, params: dict[str, Any] | None, cls: type
) -> None:
    arity = {"row_mean": 3, "ratio": 2}.get(op, 1)
    obj = FEATURE_SPEC_ADAPTER.validate_python(spec(op, [f"c{i}" for i in range(arity)], params))
    assert isinstance(obj, cls)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("op", "params", "ok"),
    [("ratio", {"eps": 1e-9}, True), ("ratio", {"eps": 1.0}, True),
     ("ratio", {"eps": 1e-10}, False), ("ratio", {"eps": 1.1}, False),
     ("ratio", {"eps": "1e-6"}, False),
     ("clip", {"lower_q": 0.2, "upper_q": 0.8}, True),
     ("clip", {"lower_q": 0.21, "upper_q": 0.9}, False),
     ("clip", {"lower_q": 0.1, "upper_q": 0.79}, False), ("clip", {"lower_q": 0.1}, False),
     ("bin_quantile", {"n_bins": 2}, True), ("bin_quantile", {"n_bins": 20}, True),
     ("bin_quantile", {"n_bins": 1}, False), ("bin_quantile", {"n_bins": 21}, False),
     ("bin_quantile", {"n_bins": 5.5}, False),
     ("equals", {"value": 3.5}, True), ("equals", {"value": "a"}, False),
     ("equals", {"value": True}, False), ("equals", {"value": float("inf")}, False),
     ("square", {}, True), ("square", {"power": 3}, False)],
)  # fmt: skip
def test_param_bounds(op: str, params: dict[str, Any], ok: bool) -> None:
    data = spec(op, ["a", "b"] if op == "ratio" else ["a"], params)
    if ok:
        FEATURE_SPEC_ADAPTER.validate_python(data)
    else:
        with pytest.raises(ValidationError):
            FEATURE_SPEC_ADAPTER.validate_python(data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("op", "n_inputs", "ok"),
    [("add", 2, True), ("add", 3, False), ("add", 1, False), ("log1p", 2, False),
     ("count_missing", 1, False), ("count_missing", 20, True), ("count_missing", 21, False),
     ("row_sum", 2, True), ("concat_categories", 3, True), ("concat_categories", 4, False),
     ("group_mean", 2, True), ("group_mean", 1, False), ("group_count", 1, True),
     ("group_count", 2, False), ("date_diff_days", 2, True), ("is_missing", 1, True)],
)  # fmt: skip
def test_input_arity(op: str, n_inputs: int, ok: bool) -> None:
    data = spec(op, [f"c{i}" for i in range(n_inputs)])
    if ok:
        FEATURE_SPEC_ADAPTER.validate_python(data)
    else:
        with pytest.raises(ValidationError):
            FEATURE_SPEC_ADAPTER.validate_python(data)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad", [{"op": "python"}, {"op": "eval"}, {"name": "Bad Name"}, {"name": "1abc"},
            {"name": "a" * 61}, {"id": "../x"}, {"inputs": []}, {"inputs": [""]}],
)  # fmt: skip
def test_unsafe_or_malformed_specs_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FEATURE_SPEC_ADAPTER.validate_python({**spec("square", ["a"]), **bad})


# ---------------------------------------------------------------- reflection (§19.3)
@pytest.mark.unit
def test_reflection_cannot_carry_prohibited_changes() -> None:
    for change in ({"type": "change_metric", "value": "log_loss"}, {"type": "set_folds", "k": 3},
                   {"type": "drop_model"}):  # fmt: skip
        with pytest.raises(ValidationError):
            ReflectionProposal.model_validate(mutate("ReflectionProposal", changes=[change]))
    with pytest.raises(ValidationError):
        ReflectionProposal.model_validate(mutate("ReflectionProposal", holdout="open"))
    add = {"type": "add_features", "features": [_feature(i) for i in range(7)]}
    with pytest.raises(ValidationError):
        ReflectionProposal.model_validate(mutate("ReflectionProposal", changes=[add]))


@pytest.mark.unit
def test_reflection_input_excludes_plan_rationale() -> None:
    data = fixture("ReflectionInput")
    data["current_plan"]["rationale"] = "LLM text"
    with pytest.raises(ValidationError, match="rationale"):
        ReflectionInput.model_validate(data)


# ---------------------------------------------------------------- consistency rules
@pytest.mark.unit
def test_leaderboard_statistics_must_match_fold_scores() -> None:
    """T-EVL-01 foundation: stored mean/std/sem are recomputable from fold scores."""
    entry = fixture("LeaderboardEntry")
    LeaderboardEntry.model_validate(entry)
    for field in ("mean", "std", "sem"):
        with pytest.raises(ValidationError, match=field):
            LeaderboardEntry.model_validate({**entry, field: entry[field] + 1e-6})


@pytest.mark.unit
def test_metric_direction_is_fixed_by_metric() -> None:
    with pytest.raises(ValidationError):
        MetricSpec.model_validate({"name": "rmse", "direction": "maximize"})
    spec_ = MetricSpec.model_validate({"name": "log_loss", "direction": "minimize"})
    assert spec_.selection_score(0.3) == -0.3


@pytest.mark.unit
def test_metric_must_be_allowed_for_task() -> None:
    """FR-TGT-03: binary cannot use RMSE; regression cannot use ROC-AUC."""
    with pytest.raises(ValidationError):
        RunConfig.model_validate(mutate("RunConfig", primary_metric="rmse"))
    cfg = mutate("ExperimentConfig", primary_metric={"name": "rmse", "direction": "minimize"})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


@pytest.mark.unit
def test_seeds_are_derived_from_root() -> None:
    """I-13 / §11.3."""
    assert Seeds.derive(7).model_dump() == {
        "root": 7, "split": 7, "model": 8, "optuna": 9, "profile": 10, "explain": 11
    }  # fmt: skip
    with pytest.raises(ValidationError):
        Seeds.model_validate({**Seeds.derive(7).model_dump(), "optuna": 1})


@pytest.mark.unit
def test_split_spec_partitions() -> None:
    base = fixture("SplitSpec")
    SplitSpec.model_validate(base)
    with pytest.raises(ValidationError, match="overlap"):
        SplitSpec.model_validate({**base, "holdout_indices": [1, 5, 10, 15]})
    with pytest.raises(ValidationError, match="cover"):
        SplitSpec.model_validate({**base, "n_valid": 21})
    with pytest.raises(ValidationError, match="fold ids"):
        SplitSpec.model_validate({**base, "k": 3})
    with pytest.raises(ValidationError, match="group"):
        SplitSpec.model_validate({**base, "strategy": "group_kfold"})


@pytest.mark.unit
def test_failures_carry_error_codes() -> None:
    """T-E2E-05 foundation: every failed run carries an E-code."""
    with pytest.raises(ValidationError, match="error code"):
        RunResult.model_validate(mutate("RunResult", status="failed", final=None))
    ok = mutate("RunResult", status="failed", final=None, error_code="E-NO-VALID-MODEL")
    assert RunResult.model_validate(ok).error_code is not None
    with pytest.raises(ValidationError):
        TrainingResult.model_validate(mutate("TrainingResult", status="failed", error=None))


def _set(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Set a dotted path (list indices allowed) in a copy-free way for test mutations."""
    node: Any = data
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[int(key)] if key.isdigit() else node[key]
    node[int(keys[-1]) if keys[-1].isdigit() else keys[-1]] = value
    return data


INVALID_CASES = [
    # (contract, dotted path, bad value, rule)
    ("ColumnInfo", "role", "excluded", "excluded column without reason (FR-ING-04)"),
    ("TaskDetection", "alternatives", [], "ambiguous detection lists alternatives"),
    ("CompactProfile", "mode", "minimal", "minimal profile has no quantiles"),
    ("PlanValidation", "status", "valid", "valid plan has no issues"),
    ("CandidateSpec", "feature_set", "baseline", "baseline candidate has no LLM features"),
    ("CandidateSpec", "features", [_feature(i) for i in range(13)], "A-010: ≤ 12 LLM features"),
    ("EnsembleResult", "members.0.weight", 0.9, "weights sum to 1"),
    ("EvaluationResult", "selection.winner", "0-xgboost-00000000", "winner must be eligible"),
    ("RunState", "final", "FinalResult", "final result requires frozen selection"),
    ("RunEvent", "origin", "llm_rationale", "llm_rationale origin pairing"),
    ("ArtifactManifest", "files.0.path", "../etc/passwd", "paths stay inside the bundle"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("name", "path", "value", "rule"), INVALID_CASES)
def test_consistency_rules(name: str, path: str, value: Any, rule: str) -> None:
    data = fixture(name)
    if value == "FinalResult":
        value = fixture("FinalResult")
    with pytest.raises(ValidationError):
        validate(name, _set(data, path, value))


@pytest.mark.unit
def test_narrative_and_holdout_rules() -> None:
    with pytest.raises(ValidationError):  # only fact-checked LLM narratives ship
        NarrativeSection(source="llm", fact_check="failed", text="x")
    holdout = fixture("FinalResult")["holdout"]
    with pytest.raises(ValidationError):  # ci95 must be [low, high]
        HoldoutResult.model_validate({**holdout, "final": {"ci95": [1, 0]}})
    assert FinalResult.model_validate(fixture("FinalResult")).is_ensemble is False


@pytest.mark.unit
def test_llm_request_schema_type_is_not_serialized() -> None:
    """PI-25: response_schema is usable in Python but excluded from JSON and the schema."""
    req = LLMRequest.model_validate({**fixture("LLMRequest"), "response_schema": PlannerOutput})
    assert req.response_schema is PlannerOutput
    assert "response_schema" not in req.model_dump()
    assert "response_schema" not in LLMRequest.model_json_schema()["properties"]
