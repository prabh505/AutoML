"""The LangGraph run graph (PRD §13.2, §13.5, §13.6).

```
plan → validate_plan → build_features → train_candidates → evaluate
evaluate → reflect → apply_changes → build_features …   (≤ max_rounds, recursion limit 40)
evaluate → ensemble → finalize → explain → report → export
train_candidates / evaluate → recover → train_candidates | finalize | fail
```

Nodes decide the next step and store it in `ctx.route`; the edge functions read it. State is
checkpointed in memory (`MemorySaver`) after every node, so an unexpected node error is retried
once from the last completed node (§13.6). Cancellation and E-codes are not retried.
"""

from __future__ import annotations

import sys
from enum import Enum
from functools import cache
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from tabpilot import schemas
from tabpilot.export import BundleTooLargeError, RawRowError
from tabpilot.orchestration import nodes
from tabpilot.orchestration.context import context
from tabpilot.schemas import RunState, TabPilotError
from tabpilot.training import Cancelled

RECURSION_LIMIT = 40
NODE_RETRIES = 1

NODES = {
    "plan": nodes.plan_node,
    "validate_plan": nodes.validate_plan_node,
    "build_features": nodes.build_features_node,
    "train_candidates": nodes.train_candidates_node,
    "evaluate": nodes.evaluate_node,
    "reflect": nodes.reflect_node,
    "apply_changes": nodes.apply_changes_node,
    "ensemble": nodes.ensemble_node,
    "finalize": nodes.finalize_node,
    "explain": nodes.explain_node,
    "report": nodes.report_node,
    "export": nodes.export_node,
    "recover": nodes.recover_node,
    "fail": nodes.fail_node,
}
ROUTES = {
    "train_candidates": ("evaluate", "recover"),
    "evaluate": ("reflect", "ensemble", "recover"),
    "reflect": ("apply_changes", "ensemble"),
    "recover": ("train_candidates", "finalize", "fail"),
}
EDGES = (("plan", "validate_plan"), ("validate_plan", "build_features"),
         ("build_features", "train_candidates"), ("apply_changes", "build_features"),
         ("ensemble", "finalize"), ("finalize", "explain"), ("explain", "report"),
         ("report", "export"))  # fmt: skip


@cache
def checkpoint_types() -> list[tuple[str, str]]:
    """(module, class) of every contract and enum a `RunState` can hold."""
    found: list[tuple[str, str]] = []
    prefix = f"{schemas.__name__}."
    # `tabpilot.schemas` imports every contract module, so they are all in `sys.modules`
    # (no dynamic import, T-SEC-01).
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(prefix) or module is None:
            continue
        for cls_name, obj in vars(module).items():
            if isinstance(obj, type) and obj.__module__ == name \
                    and issubclass(obj, BaseModel | Enum):  # fmt: skip
                found.append((name, cls_name))
    return sorted(found)


def _route(state: RunState) -> str:
    return context(state.run_id).route


def build_graph() -> Any:
    """The compiled graph with an in-memory checkpointer."""
    graph = StateGraph(RunState)
    for name, node in NODES.items():
        graph.add_node(name, node)
    graph.add_edge(START, "plan")
    for source, target in EDGES:
        graph.add_edge(source, target)
    for source, targets in ROUTES.items():
        graph.add_conditional_edges(source, _route, {t: t for t in targets})
    graph.add_edge("export", END)
    graph.add_edge("fail", END)
    # The checkpoint lives in this process's memory and holds only our own `RunState` objects.
    # Their classes are registered explicitly, so a LangGraph that blocks unregistered types
    # (its announced default) still resumes the retry.
    serde = JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types())
    return graph.compile(checkpointer=MemorySaver(serde=serde))


def run_graph(state: RunState) -> RunState:
    """Run the graph for `state.run_id` (its `RunContext` must be registered)."""
    app = build_graph()
    config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": state.run_id}}
    payload: RunState | None = state
    for attempt in range(NODE_RETRIES + 1):
        try:
            out = app.invoke(payload, config=config)
            return RunState.model_validate(out)
        except (Cancelled, TabPilotError, BundleTooLargeError, RawRowError):
            raise
        except Exception:
            if attempt == NODE_RETRIES:
                raise
            payload = None  # resume the same thread from its last completed node
    raise RuntimeError("unreachable")
