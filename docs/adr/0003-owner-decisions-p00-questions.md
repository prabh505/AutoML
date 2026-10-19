# ADR-0003: Owner decisions on the P00 open questions (Q-04…Q-16)

| Field | Value |
| --- | --- |
| Status | accepted (owner, 2026-10-19) |
| Date | 2026-10-19 |
| Prompt | P00 follow-up |
| Decision Log | DEC-007 … DEC-019 |
| PRD refs | §8.6, §8.7, §9.4, §11.1, §11.3, §11.7, §13.4, §13.5, §13.7, §14.3, §14.4, §19.4, §23.7, §27.2, §29.1, §29.3, §29.7, US-08, D-02, D-13, D-15 |
| Amendment | A-002 … A-010 |

## Context

P00 recorded 39 PRD issues (`CONTEXT.md` §11). Thirteen of them needed an owner decision (Q-04…Q-16). The owner answered every question on 2026-10-19. Two answers were confirmed with a follow-up question because, taken literally, they conflicted with the PRD's structure:

- Q-04 "90%": a single 0.90 exclusion level would sit below the 0.95 warning level.
- Q-06 "1.5–2×": stage shares are percentages that must still sum to 100%.

The owner also said that budgets are deliberately generous for now and will be tightened after the product works.

## Decision

| ID | Question | Decision (owner) | Amendment |
| --- | --- | --- | --- |
| DEC-007 | Q-04 leakage thresholds (PI-07) | Classification (univariate AUC): **near-deterministic ≥ 0.95 → excluded by default**; **strong ≥ 0.90 → warn only**. Regression (R²): **near-deterministic ≥ 0.90**, **strong ≥ 0.80**. The `leaky_binary.csv` copy keeps 0.5% noise (AUC ≈ 0.995 → excluded). The fixture's legitimate strong predictor must fall in AUC [0.90, 0.95) for T-LEAK-06. Owner intends to revisit later. | A-002 (overrides §11.7 thresholds; settles D-15) |
| DEC-008 | Q-05 holdout basis (PI-06) | Every small-data rule uses **valid rows before splitting (n_valid)**. n_valid ≥ 2,000 → 20% holdout + single K-fold. Otherwise → no holdout + repeated 3×5 CV, ≤ 20 tuning trials per model (PI-28). W-SMALL-DATA stays on DEV rows < 500. | A-003 (clarifies §11.3, §11.5, §8.10) |
| DEC-009 | Q-06 time budgets (PI-01…PI-05, PI-27) | **Budgets and timeouts doubled**; the §29.3 ledger governs conflicts. Exact values are in the table below. | A-004 |
| DEC-010 | Q-07 numeric quantiles to LLM (PI-15) | Allowed in standard mode, including min/max of feature columns. The privacy notice must say numeric ranges are sent. Minimal mode removes them. FR-PRF-03 "no cell values" is scoped as T-PRIV-01 (identifier/categorical values). | A-005 |
| DEC-011 | Q-08 re-run after completion (PI-22) | Data is deleted after export as §24.7 says. "Re-run with changes" needs a re-upload; the UI keeps the previous settings. | none (PRD behavior kept) |
| DEC-012 | Q-09 missing indicators (PI-11) | Indicator added at **≥ 5% missing**. 30–95% only triggers W-HIGH-MISSING. | A-006 (overrides the §8.6 row) |
| DEC-013 | Q-10 `log1p` on negatives (PI-12) | No clipping. Static validation flags the spec when the profile min is < 0. At transform, negatives become NaN and are counted (§14.5). | A-007 (overrides the §14.3 note) |
| DEC-014 | Q-11 target > 50% missing (PI-09) | New warning **W-TARGET-MISSING** requiring user confirmation. E-TARGET-TOO-FEW-ROWS is used only for < 50 rows. | A-008 (overrides the §8.7 row) |
| DEC-015 | Q-12 with/without LLM features (PI-23) | In iteration 0, the **top 2 model families by plan priority (hosted) or top 3 (local)** are trained both with and without LLM features. Other families use the plan's features. | A-009 |
| DEC-016 | Q-13 LLM feature cap (PI-32) | At most **12 active LLM features per candidate at all times**. A reflection round adds ≤ 6 and must stay within 12. | A-010 |
| DEC-017 | Q-14 per-call token limit (PI-21) | Keep the 6,000-token per-call input cap. The compact profile is truncated (§9.4 algorithm, smaller `max_chars`) until the request fits. | none |
| DEC-018 | Q-15 worker spike timing (PI-33) | A small spike in P01 checks that a `spawn` child process starts, communicates, and can be killed in the build environment. The result goes to `CONTEXT.md`. The real host is still checked at deploy (D-01/D-02). | none |
| DEC-019 | Q-16 license (PI-34, D-13) | MIT; P01 adds `LICENSE`. | none |

### A-004 exact values (DEC-009)

| Item | PRD value | New value |
| --- | --- | --- |
| Hosted time budget default / max (§29.1, §13.7) | 480 s / 900 s | **960 s / 1,800 s** |
| Local time budget default / max (§29.1) | 600 s / 3,600 s | **1,200 s / 7,200 s** |
| UI budget slider (§27.1–§27.2) | 120–900 s | **240 s – active profile max**, step 30 s |
| Ledger: ingest + split + profile (§29.3) | ≤ 5% (min 30 s) | ≤ 5%, at least 60 s; profiling total ≤ max(5%, 60 s), with group caps scaled to fit (PI-27) |
| Ledger: planning | ≤ 5% | ≤ 5% + roll-forward, node deadline capped at 120 s; past the deadline → rule-based plan |
| Ledger: initial training + tuning | 50% | **40%** |
| Ledger: reflection | ≤ 20% | **≤ 15% across all rounds**. Per round: min(25% of total, remaining reflection share, time left before the reserved slice). Skipped if the remaining non-reserved time is < 15% of total |
| Ledger: ensemble | ≤ 5% (node timeout said 10%) | ≤ 5%; node timeout = its allocation |
| Ledger: reserved (finalize, explain, report, export) | 15% | **30%**, never consumed by earlier stages |
| Stop point for training / reflection / ensemble (§13.7) | "at 100%" | **at 70%** (start of the reserved slice) |
| Reserved-slice nodes | — | timeout = min(node timeout, time left to 100%). If short: explain → gain importance; report → template narrative |
| Hard wall-clock kill | 130% | 130% (unchanged) |
| Node timeouts (§13.4) | plan 60, validate 10, features 30, evaluate 10, reflect 60, apply 10, explain 60/300, report 60, export 60, recover 5 s; finalize ≤ 2× fold fit + 30 s | plan 120, validate 20, features 60, evaluate 20, reflect 120, apply 20, explain **120 / 600**, report 120, export 120, recover 10 s; finalize ≤ 2× mean fold fit + 60 s |
| SHAP cap (§29.7) | 60 / 300 s | 120 / 600 s |
| Profiling per-group cap (§9.5); leakage scan cap (§9.2) | 10 s hosted / 60 s local; 30 s | 20 s / 120 s; 60 s |
| Per-LLM-call timeout (§23.1, §23.7) | 45 s (10–120) | **90 s (20–240)** |
| Per-trial timeout (§17.3) | min(60 s, 25% of model allocation) / 300 s | min(120 s, 25% of model allocation) / 600 s |
| LLM wall time per run | 20% of run budget | 20% (unchanged, so it doubles with the budget) |
| Trial counts, LLM call counts, RSS limits, cancel/heartbeat timings | — | unchanged (not time budgets) |

SM-2, T-PERF-01 and G-14 still measure the PRD target, "≤ 10 min p90 for 10K × ≤ 30 columns". Until budgets are tightened, those measurements run with an explicit `time_budget_s = 480` (Known Issue KI-001).

## Alternatives

Each alternative was offered to the owner with its trade-offs (see the option lists in the session record). The recommended defaults were taken for Q-05, Q-07–Q-11, Q-13–Q-16. Owner-specific choices were made for Q-04, Q-06 and Q-12.

## Consequences

- Invariants: none changed. A-002 excludes more columns by default, and every exclusion stays visible and re-includable (FR-PRF-02).
- Tests:
  - T-LEAK-05 and T-LEAK-06 use the A-002 thresholds.
  - T-RUN-03 tests the 70% stop.
  - T-TUN-01 uses the doubled per-trial timeout.
  - T-PERF-01 runs with `time_budget_s = 480` (KI-001).
- Prompts:
  - P02 (W-TARGET-MISSING; config values).
  - P03 (fixture legitimate predictor AUC in [0.90, 0.95)).
  - P04 (target rule).
  - P05 (n_valid basis).
  - P06 (thresholds, profiling caps).
  - P07 (log1p, 5% indicators).
  - P09 (ledger).
  - P11 (A/B variants, token fit).
  - P12 (feature cap, reflection share).
  - P13 (SHAP cap).
  - P15 (stop points, timeouts).
  - P16 (slider).
  - P01 (license, worker spike, config values).
- Risks: more false-positive leak exclusions (A-002); default runs take about twice as long (A-004), so the hosted demo may exceed the 10-minute expectation until it is retuned.

## Status

Accepted by the owner on 2026-10-19.
