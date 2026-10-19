# ADR-0004: Cooperative time limits in-process; DSL spec names may not shadow raw columns

| Field | Value |
| --- | --- |
| Status | proposed |
| Date | 2026-10-19 |
| Prompt | U2 (P07, P09); found by the U1 re-audit (KI-019) |
| Decision Log | DEC-033 (`CONTEXT.md` §9) |
| PRD refs | §17.4, §17.3, §13.7, §14.4, §14.5 (name row), T-TUN-01, T-FE-03 |
| Amendment | none (proposed; becomes an amendment if the owner accepts) |

## Context

1. **Watchdog.** §17.4 says the per-trial limit is enforced by a "cooperative check per fold + watchdog". In the P09 in-process CLI, a watchdog thread cannot interrupt a running native fit (LightGBM, XGBoost, CatBoost, or the scikit-learn C code). It could only set a flag that the code already checks at fold boundaries. A single slow fold can therefore overrun a trial or allocation deadline by up to one fold's fit time.
2. **DSL names.** §14.5 says a feature name that collides with an existing column gets the suffix `__2`. Later specs refer to earlier specs by `name` (depth-2 composition, §14.4). If a spec were allowed to be named, say, `age` while a raw column `age` exists, then a later spec's input `age` would be ambiguous.

## Decision

1. Time limits stay cooperative in-process: they are checked after every fold, both in CV and in tuning. The default-config fallback run is deadline-bounded too. The worst overrun is one fold's fit time, which `test_slow_fold_overrun_is_bounded_to_one_fold` measures. The hard wall-clock kill at 130% of the budget (§13.7) is the P15 worker's job: it runs in a separate process that can be terminated, and that is the real watchdog.
2. `validate_spec` rejects a spec whose `name` equals an existing raw column (`duplicate_feature`, T-FE-03). The `__2` suffix still applies, as §14.5 says, to the prefixed output column (`llm__<name>`) when it collides with an existing column (`DSLFeatureTransformer`).

## Alternatives

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| In-process watchdog thread | Matches the §17.4 wording | Cannot stop native code; adds threads with no effect | Ineffective |
| Boosting callbacks that check the deadline per iteration | Stops LightGBM/XGBoost/CatBoost mid-fit | No effect for linear/RF; partial models complicate early stopping and refit | Adds complexity before the P15 worker, which solves it for all models |
| Suffix raw-column name collisions (`age` → `age__2`) | Literal reading of §14.5 | References by name become ambiguous; the LLM would see a feature renamed behind its back | Ambiguity risk |

## Consequences

- Invariants: none changed. I-12 (limits enforced in code) holds with a bounded one-fold overrun until P15 adds the hard kill.
- Tests: `test_slow_fold_overrun_is_bounded_to_one_fold`, `test_T_TUN_01_time_mode_stops_at_allocation`, and T-FE-03 (`duplicate_feature`).
- Prompts: P15 must implement the worker kill at 130% (§13.7). P11's plan validator reports a rejected name as a repair issue.
- Risk: a run on hardware where one fold takes minutes can exceed its stage share by that fold's time.

## Status

Proposed by the implementer. Both items change behavior described in the PRD, so the owner must accept or reject them (CLAUDE.md §4).
