"""P15 graph tests: routing, loop bounds, fallbacks, events, budgets (PRD §13).

Runs the real worker in-process (`tests/helpers/run_harness.py`) with the offline stub provider.
T-RFL-02 · T-LLM-05 · T-PRIV-03 · T-RUN-01 · T-RUN-03 · recover/fail paths · budget skips.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from tests.helpers.run_harness import CancelFlag, go

from tabpilot.llm.stub import MODES, StubProvider
from tabpilot.orchestration import RECURSION_LIMIT, build_graph
from tabpilot.orchestration import context as run_context
from tabpilot.orchestration import graph as run_graph_module
from tabpilot.orchestration import nodes as run_nodes
from tabpilot.orchestration import training as run_training
from tabpilot.reflection import RoundBudget
from tabpilot.resources import TimeBudget
from tabpilot.runtime import worker as run_worker
from tabpilot.schemas import ErrorCode, ModelName, ProviderName, Stage

BINARY = "binary_small.csv"


def records(outcome: Any, name: str) -> list[dict[str, Any]]:
    path = outcome.run_dir / "records" / name
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def llm_purposes(outcome: Any) -> list[str]:
    return [r["purpose"] for r in records(outcome, "llm_calls.jsonl")]


def experiment(outcome: Any) -> dict[str, Any]:
    data = json.loads((outcome.run_dir / "records" / "experiment.json").read_text("utf-8"))
    assert isinstance(data, dict)
    return data


# ---------------------------------------------------------------- structure (§13.2, §13.5)
@pytest.mark.unit
def test_graph_has_the_nodes_and_edges_of_the_prd() -> None:
    graph = build_graph().get_graph()
    nodes = set(graph.nodes) - {"__start__", "__end__"}
    assert nodes == {"plan", "validate_plan", "build_features", "train_candidates", "evaluate",
                     "reflect", "apply_changes", "ensemble", "finalize", "explain", "report",
                     "export", "recover", "fail"}  # fmt: skip
    edges = {(e.source, e.target) for e in graph.edges}
    for pair in [("__start__", "plan"), ("plan", "validate_plan"),
                 ("validate_plan", "build_features"), ("build_features", "train_candidates"),
                 ("apply_changes", "build_features"), ("ensemble", "finalize"),
                 ("finalize", "explain"), ("explain", "report"), ("report", "export"),
                 ("export", "__end__"), ("fail", "__end__"),
                 ("train_candidates", "evaluate"), ("train_candidates", "recover"),
                 ("evaluate", "reflect"), ("evaluate", "ensemble"), ("evaluate", "recover"),
                 ("reflect", "apply_changes"), ("reflect", "ensemble"),
                 ("recover", "train_candidates"), ("recover", "finalize"), ("recover", "fail")]:  # fmt: skip
        assert pair in edges, pair
    assert RECURSION_LIMIT == 40


# ---------------------------------------------------------------- the stub LLM path
@pytest.mark.integration
def test_stub_run_uses_the_llm_for_plan_reflection_and_narrative(tmp_path: Path) -> None:
    outcome = go(tmp_path, llm="stub")
    assert outcome.result.status == "completed"
    assert llm_purposes(outcome) == ["plan", "reflect", "narrate"]
    exp = experiment(outcome)
    assert exp["plan_source"] == "llm"
    assert len(exp["reflection_history"]) == 1
    report = json.loads((outcome.run_dir / "report" / "report.json").read_text("utf-8"))
    assert report["sections"]["narrative"]["source"] == "llm"
    assert report["sections"]["narrative"]["fact_check"] == "passed"
    assert all(e.origin == "system" for e in outcome.events)  # the LLM never creates events
    exported = outcome.of("bundle_exported")
    assert len(exported) == 1
    assert exported[0].data["sha256"] == outcome.result.final.bundle_sha256  # type: ignore[union-attr]


# ---------------------------------------------------------------- T-LLM-05 / failure modes
@pytest.mark.integration
@pytest.mark.parametrize("mode", [m for m in MODES if m != "ok"])
def test_T_LLM_05_every_stub_failure_mode_still_completes(tmp_path: Path, mode: str) -> None:
    outcome = go(tmp_path, llm="stub", stub_mode=mode)
    assert outcome.result.status == "completed", outcome.result
    assert outcome.result.final is not None
    assert (outcome.run_dir / outcome.result.final.bundle_path).is_file()  # type: ignore[arg-type]
    exp = experiment(outcome)
    assert exp["plan_source"] == "rule_based"  # the rule-based plan is the fallback (§12.8)
    report = json.loads((outcome.run_dir / "report" / "report.json").read_text("utf-8"))
    assert report["sections"]["narrative"]["source"] == "template"
    assert "planner_fallback" in outcome.types()


@pytest.mark.integration
def test_T_LLM_05_providers_without_keys_run_rule_based(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    outcome = go(tmp_path, llm="providers", providers=(ProviderName.GROQ, ProviderName.GEMINI))
    assert outcome.result.status == "completed"
    assert experiment(outcome)["plan_source"] == "rule_based"
    assert not any(t.startswith("llm_call") for t in outcome.types())  # nothing was attempted


# ---------------------------------------------------------------- T-PRIV-03 no provider calls
@pytest.mark.integration
def test_T_PRIV_03_llm_off_makes_no_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[Any] = []
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: attempts.append(addr))

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("an LLM component was used with llm_mode=off")

    monkeypatch.setattr(StubProvider, "complete", forbidden)
    monkeypatch.setattr(run_worker, "build_router", forbidden)
    monkeypatch.setattr(run_worker.LLMRouter, "complete", forbidden)
    outcome = go(tmp_path, llm="off")
    assert outcome.result.status == "completed"
    assert attempts == []  # the network spy saw no connection attempt
    assert not any(t.startswith("llm_") for t in outcome.types())
    assert not (outcome.run_dir / "records" / "llm_calls.jsonl").exists()  # no router, no record
    exp = experiment(outcome)
    assert exp["config"]["llm"]["enabled"] is False
    assert exp["config"]["prompt_template_versions"] == {}
    assert "reflection_skipped" in outcome.types()  # §19.6: no LLM → no reflection


# ---------------------------------------------------------------- T-RFL-02 loop bounds
def always_accept(real: Any) -> Any:
    def accept(parent: Any, child: Any) -> Any:
        decision = real(parent, child)
        return decision.model_copy(update={"accepted": True, "reason": "forced for the test"})

    return accept


@pytest.mark.integration
@pytest.mark.parametrize("max_rounds", [0, 1, 2])
def test_T_RFL_02_always_proposing_reflector_stops_at_max_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_rounds: int
) -> None:
    monkeypatch.setattr(run_nodes, "accept", always_accept(run_nodes.accept))
    outcome = go(tmp_path, llm="stub", rounds=max_rounds, models=(ModelName.LINEAR,))
    assert outcome.result.status == "completed"
    assert llm_purposes(outcome).count("reflect") == max_rounds  # one proposal per round, no more
    history = experiment(outcome)["reflection_history"]
    assert [h["round"] for h in history] == list(range(1, max_rounds + 1))
    assert max(len(outcome.of("reflection_proposed")), 0) == max_rounds
    reflect_starts = [e for e in outcome.of("stage_started") if e.stage == Stage.REFLECT]
    assert len(reflect_starts) <= max_rounds


@pytest.mark.integration
def test_T_RFL_02_rejected_round_stops_the_loop_even_with_rounds_left(tmp_path: Path) -> None:
    outcome = go(
        tmp_path, llm="stub", rounds=2, stub_mode="adversarial", models=(ModelName.LINEAR,)
    )
    assert outcome.result.status == "completed"
    assert llm_purposes(outcome).count("reflect") <= 1  # §19.6: stop when nothing was accepted


@pytest.mark.integration
def test_reflection_is_skipped_when_the_round_budget_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_nodes, "reflection_budget", lambda *a, **k: RoundBudget(0.0, 0, "budget")
    )
    outcome = go(tmp_path, llm="stub")
    skipped = outcome.of("reflection_skipped")
    assert [e.data["reason"] for e in skipped] == ["budget"]
    assert "reflect" not in llm_purposes(outcome)


# ---------------------------------------------------------------- recover and fail (§13.2)
@pytest.mark.integration
def test_training_error_recovers_with_a_fallback_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("simulated trainer crash")

    monkeypatch.setattr(run_nodes, "train_initial", boom)
    outcome = go(tmp_path)
    assert outcome.result.status == "completed"
    assert "stage_failed" in outcome.types()
    final = outcome.result.final
    assert final is not None
    assert final.selected_entry_id.startswith("0-linear")
    exp = experiment(outcome)
    assert exp["evaluation"]["selection"]["winner"] == final.selected_entry_id
    recover = [e for e in outcome.events if e.stage == Stage.RECOVER]
    assert recover  # the recover node ran


@pytest.mark.integration
def test_no_valid_model_anywhere_fails_with_e_no_valid_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("simulated trainer crash")

    monkeypatch.setattr(run_nodes, "train_initial", boom)
    monkeypatch.setattr(run_nodes, "run_cv", boom)
    outcome = go(tmp_path)
    assert outcome.result.status == "failed"
    assert outcome.result.error_code == ErrorCode.NO_VALID_MODEL
    assert "No model trained successfully" in (outcome.result.error_message or "")
    assert "run_failed" in outcome.types()
    assert outcome.result.final is None
    assert not list(outcome.run_dir.glob("*.zip"))  # nothing is offered


# ---------------------------------------------------------------- T-RUN-03 partial completion
def expire_training_after_first_model(monkeypatch: pytest.MonkeyPatch) -> None:
    real = TimeBudget.expired

    def expired(self: TimeBudget, stage: Stage) -> bool:
        contexts = list(run_context._REGISTRY.values())
        trained = bool(contexts and contexts[0].runs)
        return (stage == Stage.TRAIN_CANDIDATES and trained) or real(self, stage)

    monkeypatch.setattr(TimeBudget, "expired", expired)


@pytest.mark.integration
def test_T_RUN_03_time_budget_hit_gives_a_partial_result_when_a_model_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expire_training_after_first_model(monkeypatch)
    outcome = go(tmp_path, tuning_mode="time", trials=1, ensemble=False,
                 models=(ModelName.LINEAR, ModelName.LIGHTGBM, ModelName.RANDOM_FOREST))  # fmt: skip
    result = outcome.result
    assert result.status == "completed_partial"
    assert result.stopped_after_stage == Stage.TRAIN_CANDIDATES
    assert result.final is not None
    assert (outcome.run_dir / result.final.bundle_path).is_file()  # type: ignore[arg-type]
    board = json.loads((outcome.run_dir / "leaderboard.json").read_text("utf-8"))
    statuses = sorted(e["status"] for e in board)
    assert statuses == ["ok", "skipped_budget", "skipped_budget"]
    report = json.loads((outcome.run_dir / "report" / "report.json").read_text("utf-8"))
    assert report["status"] == "completed_partial"
    assert any("status completed_partial" in item for item in report["sections"]["limitations"])


@pytest.mark.integration
def test_T_RUN_03_time_budget_hit_without_any_valid_model_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = TimeBudget.expired
    monkeypatch.setattr(TimeBudget, "expired",
                        lambda self, stage: stage == Stage.TRAIN_CANDIDATES or real(self, stage))  # fmt: skip
    outcome = go(tmp_path, tuning_mode="time", trials=1)
    assert outcome.result.status == "failed"
    assert outcome.result.error_code == ErrorCode.NO_VALID_MODEL


# ---------------------------------------------------------------- cancellation in the graph
@pytest.mark.integration
def test_cancel_during_training_stops_the_graph_without_a_bundle(tmp_path: Path) -> None:
    flag = CancelFlag()

    def cancel_when_training(event: Any) -> None:
        if event.type.value == "candidate_started":
            flag.set()

    outcome = go(tmp_path, cancel=flag, on_event=cancel_when_training)
    assert outcome.result.status == "cancelled"
    assert outcome.result.stopped_after_stage == Stage.TRAIN_CANDIDATES
    assert "run_cancelled" in outcome.types()
    assert "bundle_exported" not in outcome.types()
    assert "holdout_evaluated" not in outcome.types()
    assert not list(outcome.run_dir.glob("*.zip"))


# ---------------------------------------------------------------- T-RUN-01 events reflect actions
@pytest.mark.integration
def test_T_RUN_01_events_are_emitted_after_the_action_they_describe(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def check(event: Any) -> None:
        ctx = next(iter(run_context._REGISTRY.values()), None)
        kind = event.type.value
        if kind == "selection_frozen":
            seen["frozen"] = ctx.freeze.frozen and ctx.final_model is not None
        elif kind == "holdout_evaluated":
            seen["holdout"] = ctx.holdout is not None and len(ctx.guard.access_log) == 1
        elif kind == "report_rendered":
            seen["report"] = (ctx.run_dir / "report" / "report.html").is_file()
        elif kind == "bundle_exported":
            seen["bundle"] = (ctx.run_dir / event.data["file"]).is_file()
        elif kind == "candidate_completed":
            seen.setdefault("candidates", []).append(event.data["entry_id"] in ctx.runs)
        elif kind == "ensemble_decided":
            seen["ensemble"] = ctx.freeze.frozen is False
        elif kind == "leaderboard_updated":
            seen.setdefault("boards", []).append(event.data["n"])

    outcome = go(tmp_path, "binary_basic.csv", "Churn", llm="stub", on_event=check)
    assert outcome.result.status == "completed"
    assert seen["frozen"]
    assert seen["holdout"]
    assert seen["report"]
    assert seen["bundle"]
    assert seen["ensemble"]  # the ensemble decision precedes the freeze
    assert seen["candidates"]
    assert all(seen["candidates"])
    assert seen["boards"] == sorted(seen["boards"])  # the leaderboard only grows
    kinds = outcome.types()
    assert kinds.index("selection_frozen") < kinds.index("holdout_evaluated")
    assert kinds.index("holdout_evaluated") < kinds.index("report_rendered")
    assert kinds.index("report_rendered") < kinds.index("bundle_exported")
    assert [e.seq for e in outcome.events] == list(range(len(outcome.events)))
    assert kinds[-1] == "run_completed"
    stages_completed = [e.stage.value for e in outcome.of("stage_completed")]
    for stage in ("ingest", "split", "profile", "plan", "validate_plan", "build_features",
                  "train_candidates", "evaluate", "finalize", "explain", "report", "export"):  # fmt: skip
        assert stage in stages_completed


# ---------------------------------------------------------------- budgets and the reserved slice
@pytest.mark.integration
def test_report_uses_the_template_when_the_reserved_slice_is_nearly_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = TimeBudget.remaining

    def remaining(self: TimeBudget, stage: Stage) -> float:
        return 5.0 if stage == Stage.REPORT else real(self, stage)

    monkeypatch.setattr(TimeBudget, "remaining", remaining)
    outcome = go(tmp_path, llm="stub")
    assert outcome.result.status == "completed"
    assert "narrate" not in llm_purposes(outcome)  # §29.3: no LLM when the slice is short
    report = json.loads((outcome.run_dir / "report" / "report.json").read_text("utf-8"))
    assert report["sections"]["narrative"]["source"] == "template"


@pytest.mark.integration
def test_ensemble_is_skipped_when_its_budget_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = TimeBudget.expired
    monkeypatch.setattr(TimeBudget, "expired",
                        lambda self, stage: stage == Stage.ENSEMBLE or real(self, stage))  # fmt: skip
    outcome = go(tmp_path)
    skipped = [e for e in outcome.of("stage_skipped") if e.stage == Stage.ENSEMBLE]
    assert [e.data["reason"] for e in skipped] == ["budget"]
    assert "ensemble_decided" not in outcome.types()


@pytest.mark.integration
def test_disabled_ensemble_and_oof_option(tmp_path: Path) -> None:
    outcome = go(tmp_path, ensemble=False, include_oof=True)
    skipped = [e for e in outcome.of("stage_skipped") if e.stage == Stage.ENSEMBLE]
    assert [e.data["reason"] for e in skipped] == ["disabled"]
    assert outcome.result.final is not None
    assert not outcome.result.final.is_ensemble


@pytest.mark.integration
def test_graph_checkpoint_retry_resumes_once_after_an_unexpected_node_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")  # unregistered types would be blocked
    calls = {"n": 0}
    real = run_nodes.explain_node

    def flaky(state: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return real(state)

    monkeypatch.setitem(run_graph_module.NODES, "explain", flaky)
    outcome = go(tmp_path)
    assert outcome.result.status == "completed"
    assert calls["n"] == 2  # retried once from the last checkpoint (§13.6)
    assert outcome.types().count("selection_frozen") == 1  # finalize was not repeated
    assert outcome.types().count("holdout_evaluated") == 1  # the holdout opens once (L-7)
    assert "unregistered type" not in caplog.text  # every checkpointed class is registered


@pytest.mark.integration
def test_run_training_helpers_reserve_trials_for_every_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = run_context.TimeBudget
    monkeypatch.setattr(run_context, "TimeBudget", lambda total, _cap: real(total, 3))
    outcome = go(tmp_path, models=(ModelName.LINEAR, ModelName.LIGHTGBM, ModelName.RANDOM_FOREST),
                 trials=3, ensemble=False)  # fmt: skip
    board = json.loads((outcome.run_dir / "leaderboard.json").read_text("utf-8"))
    assert all(e["status"] == "ok" for e in board)  # trial 0 reserved for each planned model
    assert len(records(outcome, "trials.jsonl")) == 3
    assert run_training.initial_candidates  # the module under test is the one the graph uses


# ---------------------------------------------------------------- U2: KI-028 (A-009), KI-029
@pytest.mark.integration
def test_A_009_top_families_are_also_trained_without_llm_features(tmp_path: Path) -> None:
    """A-009: with LLM features in the plan, each of the top families has a baseline twin that
    reuses the sibling's tuned parameters; without features there are no twins.
    """
    outcome = go(tmp_path, llm="stub", models=(ModelName.LINEAR, ModelName.LIGHTGBM), rounds=0)
    board = json.loads((outcome.run_dir / "leaderboard.json").read_text("utf-8"))
    iteration0 = [e for e in board if e["iteration"] == 0]
    by_model = {}
    for e in iteration0:
        by_model.setdefault(e["model"], set()).add(e["feature_set"])
    assert by_model == {"linear": {"baseline", "baseline+llm"},
                        "lightgbm": {"baseline", "baseline+llm"}}  # fmt: skip
    twins = [e for e in iteration0 if e["feature_set"] == "baseline"]
    siblings = {e["model"]: e for e in iteration0 if e["feature_set"] == "baseline+llm"}
    for twin in twins:
        assert twin["status"] == "ok"
        assert twin["split_id"] == siblings[twin["model"]]["split_id"]  # same folds
        assert twin["params"] == siblings[twin["model"]]["params"]  # same tuned parameters
    off = go(tmp_path / "off", llm="off", models=(ModelName.LINEAR, ModelName.LIGHTGBM))
    off_board = json.loads((off.run_dir / "leaderboard.json").read_text("utf-8"))
    assert {e["feature_set"] for e in off_board} == {"baseline"}
    assert len([e for e in off_board if e["iteration"] == 0]) == 2  # no twins without features


@pytest.mark.integration
def test_A_009_twin_count_is_limited_to_the_profile_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tabpilot.schemas import ExecutionProfileName  # noqa: PLC0415

    monkeypatch.setitem(run_training.TWIN_FAMILIES, ExecutionProfileName.LOCAL, 1)
    outcome = go(tmp_path, llm="stub", models=(ModelName.LINEAR, ModelName.LIGHTGBM), rounds=0)
    board = json.loads((outcome.run_dir / "leaderboard.json").read_text("utf-8"))
    twins = [e for e in board if e["feature_set"] == "baseline"]
    assert len(twins) == 1


@pytest.mark.integration
def test_reflector_receives_importances_and_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§19.2: the reflector payload of a real run has the incumbent's top source-column
    importances (no one-hot labels, PI-16) and the profile warnings.
    """
    sent: list[dict[str, Any]] = []
    real = StubProvider.complete

    def spy(self: StubProvider, req: Any) -> Any:
        if req.purpose == "reflect":
            sent.append(dict(req.user_json))
        return real(self, req)

    monkeypatch.setattr(StubProvider, "complete", spy)
    outcome = go(tmp_path, "binary_imbalanced.csv", "is_fraud", llm="stub", rounds=1,
                 models=(ModelName.LINEAR,))  # fmt: skip
    assert outcome.result.status == "completed"
    (payload,) = sent
    top = payload["top_features"]
    assert top
    assert len(top) <= 15
    names = {f["name"] for f in top}
    columns = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "channel", "sq_stub"}
    assert names <= columns
    assert abs(sum(f["importance"] for f in top) - 1.0) < 1e-6 or len(top) == 15
    assert any("IMBALANCE" in w.upper() for w in payload["warnings"])
    for label in ("store", "phone", "web"):  # the `channel` values never appear as names
        assert label not in names


@pytest.mark.integration
def test_binary_run_without_a_positive_label_uses_the_default_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `RunConfig` may leave `positive_label` empty; §10.1's default is then used (and
    recorded), not a crash."""
    from tabpilot import cli  # noqa: PLC0415

    real = cli.build_config
    monkeypatch.setattr(
        cli,
        "build_config",
        lambda *a, **k: real(*a, **k).model_copy(update={"positive_label": None}),
    )
    outcome = go(tmp_path, "binary_basic.csv", "Churn")
    assert outcome.result.status == "completed"
    assert experiment(outcome)["config"]["positive_label"] == "Yes"


# ---------------------------------------------------------------- U2 KI-043
@pytest.mark.integration
def test_U2_KI_043_a_kept_ensemble_is_a_leaderboard_row_and_the_final_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np  # noqa: PLC0415

    from tabpilot.ensembling import ensemble as ens  # noqa: PLC0415

    monkeypatch.setattr(
        ens, "hill_climb", lambda preds, *a, **k: np.full(len(preds), 1 / len(preds))
    )
    real = ens.paired_decision
    monkeypatch.setattr(ens, "paired_decision", lambda **k: real(**k).model_copy(
        update={"accepted": True, "reason": "forced for the test"}))  # fmt: skip
    outcome = go(tmp_path, models=(ModelName.LINEAR, ModelName.LIGHTGBM), trials=1)
    assert outcome.result.status == "completed"
    board = json.loads((outcome.run_dir / "leaderboard.json").read_text("utf-8"))
    rows = [e for e in board if e["feature_set"] == "ensemble"]
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "ensemble"
    assert row["status"] == "ok"
    assert len(row["fold_scores"]) == len(board[0]["fold_scores"])  # same folds as the singles
    assert row["split_id"] == board[0]["split_id"]
    final = outcome.result.final
    assert final is not None
    assert final.is_ensemble
    assert final.selected_entry_id == row["entry_id"]  # resolvable on the leaderboard
    selection = json.loads((outcome.run_dir / "selection.json").read_text("utf-8"))
    assert selection["winner"] != row["entry_id"]  # the single-model rule is unchanged
    html = (outcome.run_dir / "report" / "report.html").read_text("utf-8")
    assert "configurations were compared in total" in html
    facts = json.loads((outcome.run_dir / "report" / "facts.json").read_text("utf-8"))
    compared = next(f for f in facts.values() if f["key"] == "n_models_compared")
    assert compared["value"] == len(board) - 1  # the ensemble is not counted as a model type
