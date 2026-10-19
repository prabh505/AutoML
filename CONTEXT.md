# CONTEXT.md — TabPilot Project Memory

> **This is the project's memory.** Any AI or human picking up this project reads this file first. It must always describe the true current state.
>
> **Rules for editing**
> 1. Update it at the end of every prompt (CLAUDE.md SOP-FINISH).
> 2. Sections marked **append-only** are never rewritten. Add new entries at the bottom with date and prompt ID.
> 3. Sections marked **replace** are overwritten to reflect the current truth.
> 4. Record facts, not intentions. "Implemented and tested" means tests exist and passed in this session.
> 5. Keep entries short. Link to files, ADRs, and PRD sections instead of copying them.

---

## 1. Project identity  *(replace only if the product changes)*

| Field | Value |
| --- | --- |
| Product | TabPilot — LLM-assisted AutoML for tabular data (binary, multiclass, regression) |
| Core principle | The LLM plans, proposes, and explains. Deterministic, validated ML code computes and decides. |
| Owner | Prabhpreet Singh |
| Spec | `docs/PRD.md` v1.0 (40 sections) — immutable; changes only via §10 PRD Amendments |
| Build plan | `docs/PROMPTS.md` (P00–P20 + utility prompts U1–U4) |
| Rules | `CLAUDE.md` |
| Language / runtime | Python 3.11, uv, Streamlit single app, worker process per run |
| Deployment target | Streamlit Community Cloud (free tier, unverified resources — D-01); Docker image as portable fallback |

---

## 2. Source-of-truth map  *(replace)*

| Need | Where |
| --- | --- |
| What to build | `docs/PRD.md` |
| Approved changes to the spec | §9 Decision Log, §10 PRD Amendments (this file) |
| Requirement → prompt → test mapping | `docs/TRACEABILITY.md` (created in P00) |
| Architecture decisions | `docs/adr/` |
| Data contracts (generated) | `docs/schemas/` (generated in P02) |
| Security review | `docs/security/REVIEW.md` (P17) |
| Release evidence | `docs/RELEASE_CHECKLIST.md` (P20) |
| Human deployment steps | `docs/DEPLOY.md` (P20) |

---

## 3. Current state  *(replace)*

| Field | Value |
| --- | --- |
| Last completed prompt | **P00** Project charter and context bootstrap (documentation only) |
| Next prompt | **P01** Repository scaffold, toolchain, CI skeleton |
| Current milestone | Milestone 0 — Foundations |
| Test suite | Not yet created (P01 creates it) |
| Last green commit | — (no code or tests yet; P00 commit is documentation only) |
| Blockers | None for P01. Owner answers needed before P03 (Q-04), P05 (Q-05), P09 (Q-06); see §12 |

---

## 4. Prompt ledger  *(replace status cells; keep rows)*

Status values: `pending` · `in_progress` · `complete` · `complete_with_issues` · `blocked`

| ID | Title | PRD sections | Milestone / gate | Status | Commit | Date |
| --- | --- | --- | --- | --- | --- | --- |
| P00 | Project charter and context bootstrap | All (§1–§40) | M0 | complete | see `git log --grep '^P00:'` | 2026-10-19 |
| P01 | Repository scaffold, toolchain, CI skeleton | §25, §34.1–34.2, §34.5 | M0 | pending | | |
| P02 | Data contracts, configuration, logging | §26, §8.9, §14.3, §28.2, §29.1, §30.2–30.3 | M0 | pending | | |
| P03 | Deterministic fixtures and test harness | §31.1–31.4 | M0 | pending | | |
| P04 | Ingestion, validation, feasibility | §8, §29.4–29.5, FR-ING-*, FR-TGT-01 | M1 | pending | | |
| P05 | Task detection, metrics, splitting, holdout guard | §10, §11.1–11.6 | M1 | pending | | |
| P06 | Profiling, leakage scan, compact profile | §9, §11.7 | M1 | pending | | |
| P07 | Feature engine: baseline + DSL + runtime package | §14.1–14.6 | M1 | pending | | |
| P08 | Preprocessing, model zoo, CV runner | §15, §16, §11.4, §11.8 | M1 | pending | | |
| P09 | Tuning, evaluation, selection, holdout, budgets, rule-based CLI | §17, §18, §11.10, §12.8, §29.3, §29.6, §30.1 | **Gate A** | pending | | |
| P10 | LLM provider router | §23, §6.4, §24.5 | M2 | pending | | |
| P11 | LLM planner and plan validator | §12 | M2 | pending | | |
| P12 | Reflection and ensembling | §19, §20 | M2 | pending | | |
| P13 | Explainability and reporting | §21 | M2 | pending | | |
| P14 | Artifact export and reproducibility | §22 | M2 | pending | | |
| P15 | Orchestration and run runtime | §13, §28, §6, §7, §24.7 | **Gate B** | pending | | |
| P16 | Streamlit user interface | §27, §3.2, §24.5 | **Gate C** | pending | | |
| P17 | Security and privacy hardening audit | §24, §30.3 | M3 | pending | | |
| P18 | Containerization, CI/CD, performance validation | §34, §29 | M3 | pending | | |
| P19 | Benchmark harness | §32 | M3 | pending | | |
| P20 | Release readiness | §38, §35, §36, §37, §40 | **Gate D** | pending | | |

Utility prompts (run any time): U1 Independent audit · U2 Course correction · U3 Session resume / AI handoff · U4 Change request.

---

## 5. Milestone gates  *(replace status cells)*

| Gate | After | Pass condition (summary) | Status | Evidence |
| --- | --- | --- | --- | --- |
| A | P09 | Rule-based AutoML runs headless via CLI on binary, multiclass, regression fixtures; leakage tests (T-LEAK-01..08) green; leaderboard equals stored results | not started | |
| B | P15 | Full agent run (stub LLM and no-LLM) via worker + LangGraph produces a valid bundle; cancel/timeout/cleanup work; T-E2E-01..03, 05 green | not started | |
| C | P16 | User can complete the §3.2 journey in the Streamlit app locally; T-UI-01..04 green | not started | |
| D | P20 | All release gates G-01..G-18 have recorded evidence or an explicit, accepted waiver | not started | |

---

## 6. Architecture snapshot  *(replace when architecture changes)*

- **UI process** (Streamlit pages + `RunManager`) spawns **one worker process per run**. Worker: ingest → split (seal holdout) → profile (DEV only) → LangGraph graph: plan → validate_plan → build_features → train_candidates → evaluate → [reflect → apply_changes]* → ensemble → finalize (holdout once) → explain → report → export.
- LLM access only through `tabpilot.llm.LLMRouter` (Groq → Gemini → OpenRouter; Ollama locally). LLM sees compact profile, experiment summary, facts table only.
- LLM features are JSON **Feature DSL** specs executed by first-party code. No generated code runs.
- Custom estimators/transformers that get pickled live in **`tabpilot_runtime`** (A-001).

### Module status  *(replace rows)*

| Module | Owner prompt | Status | Notes |
| --- | --- | --- | --- |
| `tabpilot.schemas` | P02 | not started | |
| `tabpilot.config`, `tabpilot.logging` | P02 | not started | |
| `tabpilot.ingestion` | P04 | not started | |
| `tabpilot.resources` (estimator) | P04 / P09 | not started | |
| `tabpilot.evaluation.metrics` | P05 | not started | |
| `tabpilot.splitting` | P05 | not started | |
| `tabpilot.profiling` | P06 | not started | |
| `tabpilot.features` + `tabpilot_runtime` | P07 | not started | |
| `tabpilot.preprocessing`, `tabpilot.models`, `tabpilot.training` | P08 | not started | |
| `tabpilot.tuning`, `tabpilot.evaluation`, `tabpilot.tracking`, `planning.fallback` | P09 | not started | |
| `tabpilot.llm` | P10 | not started | |
| `tabpilot.planning` | P11 | not started | |
| `tabpilot.reflection`, `tabpilot.ensembling` | P12 | not started | |
| `tabpilot.explain`, `tabpilot.reporting` | P13 | not started | |
| `tabpilot.export` | P14 | not started | |
| `tabpilot.orchestration`, `tabpilot.runtime` | P15 | not started | |
| `tabpilot.ui` | P16 | not started | |
| `benchmark/` | P19 | not started | |

---

## 7. Tech stack and pinned versions  *(replace; filled in P01)*

| Package | Version | Why |
| --- | --- | --- |
| python | 3.11.x | PRD §34.2 |
| *(filled by P01 from uv.lock)* | | |

---

## 8. Key invariants  *(do not edit without a Decision Log entry)*

See `CLAUDE.md` §3 (I-1 … I-14). Any change to an invariant requires a Decision Log entry with status `accepted` by the owner.

---

## 9. Decision log  *(append-only)*

Format: `ID | Date | Prompt | Decision | Rationale | Status (proposed/accepted/superseded) | Ref`

| ID | Date | Prompt | Decision | Rationale | Status | Ref |
| --- | --- | --- | --- | --- | --- | --- |
| DEC-001 | 2026-10-19 | kit | Baseline decisions C-01 … C-17 of PRD §2.8 are in force as written | Already resolved in the spec | accepted | PRD §2.8 |
| DEC-002 | 2026-10-19 | kit | Open decisions D-01 … D-16 use the PRD's recommended defaults until the owner decides otherwise | Avoid blocking on non-critical questions | accepted | PRD §37 |
| DEC-003 | 2026-10-19 | kit | Introduce package `tabpilot_runtime` for all custom classes that are pickled into `model.joblib` (see A-001) | Pickle stores module paths; without this, the exported model cannot load without the full `tabpilot` package, which breaks R1 and the standalone `pipeline.py` promise | accepted | A-001 |
| DEC-004 | 2026-10-19 | kit | Build order: the deterministic rule-based fallback planner (PRD §12.8) is implemented in P09, before the LLM planner (P11) | Makes Gate A a complete, LLM-free AutoML (ablation A2) and keeps the fallback path tested from day one; no behavior change | accepted | PROMPTS P09 |
| DEC-005 | 2026-10-19 | kit | Build order: explainability (P13) and export (P14) are built before orchestration (P15) | Orchestration integrates finished components instead of stubs; no behavior change | accepted | PROMPTS |
| DEC-006 | 2026-10-19 | P00 | Traceability conventions: each FR/T/NFR/G ID has exactly one owning prompt plus optional contributors; test meanings follow the §31.3 catalog, with enumerated topics mapped to range IDs in order, except T-PRE-03, which FR-PRE-01 assigns explicitly (PI-19) | Makes ownership auditable by U1; documentation only, no behavior change | accepted | `docs/TRACEABILITY.md` |

---

## 10. PRD amendments  *(append-only)*

Each amendment overrides the PRD only for the exact item named.

| ID | PRD ref | Amendment | Reason | Status |
| --- | --- | --- | --- | --- |
| A-001 | §15.1, §20.5, §22.2, §22.4, §25.1 | Add `src/tabpilot_runtime/` — a small package with **no dependency on `tabpilot`** (only numpy, pandas, scikit-learn, and the model libraries). It contains every custom class that can appear inside a fitted pipeline: `ColumnNameMapper`, `TypeCaster`, `BaselineFeatureTransformer`, `DSLFeatureTransformer` and DSL op implementations, `ColumnSelector`, frequency/rare-category encoders, the early-stopping estimator wrapper, and `VotingEnsemble`. `tabpilot` imports these from `tabpilot_runtime`. The export bundle ships a copy of `tabpilot_runtime/` next to `pipeline.py`, so `model.joblib` loads with only the pinned third-party packages installed. "Standalone `pipeline.py`" therefore means "standalone bundle". | Pickle references classes by module path | accepted |

---

## 11. PRD issues found  *(append-only; P00 seeds this)*

| ID | PRD ref | Issue | Proposed resolution | Status |
| --- | --- | --- | --- | --- |
| PI-01 | §29.3 vs §19.4 | Reflection time. §29.3: "Reflection (incl. retraining) ≤ 20%". §19.4: "Maximum additional runtime: 25% of the original total budget per round". With `max_rounds` = 2 that allows up to 50%, and the §29.3 shares (5+5+50+20+5+15) would exceed 100%. | The §29.3 ledger governs: **all** reflection rounds together use ≤ 20% (plus roll-forward). Each round gets min(25% of total, remaining reflection share, remaining − reserved 15%). | proposed → Q-06 |
| PI-02 | §13.5, §19.4 | Reflection minimum. Both say reflection is skipped when "remaining budget" < 20% of total, but neither says whether "remaining" includes the reserved 15% slice. If it does, reflection can start with only 5% usable time. | "Remaining" means remaining **non-reserved** time (time to the 85% mark). Skip reflection if that is < 20% of the total. | proposed → Q-06 |
| PI-03 | §13.4 vs §29.3 | Ensemble node timeout is "10% of budget" (§13.4), but the ledger share is "Ensemble ≤ 5%" (§29.3). | Ensemble timeout = its ledger allocation (5% + roll-forward). | proposed → Q-06 |
| PI-04 | §13.4, §13.7 vs §29.3 | Reserved slice. 15% of the 480 s default = 72 s. That slice must cover finalize (refit ≤ 2× fold fit + 30 s), explain (60 s), report (60 s) and export (60 s), which is ≥ 210 s at the node-timeout ceilings. §13.7 says training stops "at 100%" and then jumps to finalize, but §29.3 says the reserved 15% "is never consumed by earlier stages". | Training, reflection and ensemble stop at the 85% mark. Reserved-slice nodes use min(node timeout, time left to 100%). When time is short, explain falls back to gain importance and the report to the template narrative. The 130% hard kill stays as the backstop. T-RUN-03 tests the 85% stop. | proposed → Q-06 |
| PI-05 | §29.3 vs §23.7, §13.4, §12.5 | Planning share "≤ 5%" is 24 s at the hosted default. But one planner call has a 45 s per-call timeout (§23.7), up to 3 attempts × 3 providers (§23.5), and up to 2 planner calls (§12.5), while the plan node timeout is 60 s (§13.4). | Plan node deadline = its ledger share + roll-forward from ingest/profile, capped by the 60 s node timeout. The router stops when the deadline would be exceeded, and the planner falls back to the rule-based plan (§12.9). The LLM wall-time ceiling (20%, §23.7) stays a separate global limit. | proposed → Q-06 |
| PI-06 | §11.1 vs §11.3, §11.5 | Holdout row basis. §11.1 bases the holdout on all valid rows ("rows ≥ holdout_min_rows (2,000)"). §11.3 uses "DEV rows < 2,000 (no holdout)". With 2,000–2,499 valid rows, a holdout is created, DEV has 1,600–1,999 rows, and §11.3 then calls for repeated CV with no holdout — a contradiction. | Evaluate every small-data threshold on **valid rows before splitting** (n_valid): n_valid ≥ 2,000 → 20% holdout + single K-fold; otherwise no holdout + repeated 3×5 CV. W-SMALL-DATA stays on DEV rows (§9.3). | proposed → Q-05 |
| PI-07 | FR-PRF-02 AC, §31.2, T-LEAK-05 vs §11.7 | `leaky_binary.csv` "contains a copy of the target with 0.5% noise". The univariate AUC of a binary copy with flip rate p is ≈ 1 − p = 0.995 (computed analytically in P00). That is below the near-deterministic threshold "≥ 0.999", so the column is only "strong" and **T-LEAK-05 cannot pass as specified**. | Keep the §11.7 thresholds (D-15 default). Generate the copy with ≤ 0.05% flips (AUC ≈ 0.9995). The strong-but-legitimate column (AUC 0.95–0.99, P03) still covers T-LEAK-06. Alternative (owner): lower the classification threshold to 0.99. | proposed → Q-04 |
| PI-08 | T-LEAK-08 (§31.3) | The wording is unclear: "Feature = target mean of rows *not* in training fold would score high if leaked". It doesn't say the row count, the category cardinality, or whether "≤ 0.55 (5 seeds)" applies per seed or to the mean. Also, the rule-based plan disables target encoding (§12.8), so the canary must force it. | Pure-noise binary target, n_dev = 2,000, one categorical with 500 levels (> 50, so target encoding applies) + 3 noise numerics. Candidate forced to `high_cardinality = target`. Assert OOF ROC-AUC ≤ 0.55 for **each** of seeds 0–4. Positive control: an encoder fitted on all of DEV outside the folds must score > 0.55, which proves the test can fail. | proposed, no behavior change (P08 adopts; non-blocking) |
| PI-09 | §8.9, §8.7 | E-codes. All 16 specific codes used in the PRD appear in §8.9, and the wildcards (E-FILE-\*, E-PARSE-\*, E-TARGET-\*) resolve to them. But the six E-TARGET-\* codes have no message template ("Specific per §8.7"). E-TARGET-TOO-FEW-ROWS is also used two ways: as a hard error (< 50 rows) and as a "warning requiring confirmation" (> 50% target missing). | P02 writes what/where/fix templates for each E-TARGET-\* code from §8.7. > 50% missing target becomes W-TARGET-MISSING with required confirmation. E-TARGET-TOO-FEW-ROWS is kept for < 50 rows only. | proposed → Q-11 |
| PI-10 | §9.3 vs §8.2, §8.5–§8.7, §11.1 | W-ENCODING-FALLBACK, W-COERCED, W-INF, W-TARGET-MISSING and W-NO-HOLDOUT are used but not catalogued (§9.3 lists 10 of the 15 W-codes). | The P02 `WarningCode` enum covers all 15 (list in `docs/TRACEABILITY.md` App. B). | proposed, no behavior change (P02) |
| PI-11 | §8.6 vs §14.2, §12.8 | Missing indicators. §8.6: "Missing rate 30–95% → Warning; missing indicator added by default". §14.2 and §12.8: indicators for missing ≥ 5%. | Missing indicator at ≥ 5% (baseline transform and fallback plan agree). 30% only triggers the W-HIGH-MISSING warning. | proposed → Q-09 |
| PI-12 | §14.3 vs §14.5 | `log1p` with negatives. §14.3: "requires profile min ≥ 0; else clipped at 0 and flagged". §14.5: "log1p/sqrt of negative → NaN with counter". | No silent clipping. Static validation flags the spec when the profile min is < 0 (recorded in provenance). At transform, negative inputs → NaN with a counter (§14.5), handled by the imputer downstream. | proposed → Q-10 |
| PI-13 | §11.2 L-2, §7.1 vs §21.4, §25.4 rule 2 | Holdout access. L-2 and §7.1 let the "post-freeze Explainer" read HOLD for error analysis. §21.4 says error analysis is "computed on DEV OOF predictions (no holdout use)", and §25.4 rule 2 lets only `evaluation.holdout` and `orchestration.nodes.finalize` import the guard. The module that **constructs** the guard (worker pre-graph setup; P09 CLI) is not allowed either. | The Explainer never touches HOLD (§21.4 wins; this is stricter). Contract rule 2 allows `splitting` (definition), `evaluation.holdout`, `orchestration.nodes.finalize`, and one named construction site (the run-context builder used by the P09 CLI and the P15 worker). P01 encodes this. | proposed, no behavior change (P01/P05) |
| PI-14 | I-1, T-LEAK-03 vs §11.2 L-7, FR-EVL-02 | The holdout is "opened exactly once", yet it must score both the final pipeline **and** the reference baseline. | One `open()` call returns the HOLD indices, and the Finalizer scores both models from that single access. Any second `open()` raises. | proposed, no behavior change (P05/P09) |
| PI-15 | FR-PRF-03 AC vs §9.4, §26.2 | FR-PRF-03 says the compact profile "contains no cell values from the dataset (only aggregates)". But `q` = [min, p5, …, max] includes the exact min and max (and often the median) of numeric columns, which are literal cell values. T-PRIV-01 only checks identifier and categorical columns. | Follow §9.4: numeric order statistics of feature columns are allowed aggregates in standard mode, and minimal mode removes them. Read the FR-PRF-03 AC within T-PRIV-01's scope. The privacy notice states that numeric ranges are sent. | proposed → Q-07 |
| PI-16 | §19.2, §21.7, §21.4 vs §24.4, I-6 | Category labels can reach the LLM through derived names. ReflectionInput's "top 15 features … (names)" uses post-preprocessing names, and one-hot names embed labels (e.g. `contract_Month-to-month`). The narrator's facts table includes "feature names", and §21.4 worst segments are category values. §24.4 says category labels are never sent. | Importances in ReflectionInput are aggregated to source columns (as §21.3 does for charts). The facts table sent to the LLM excludes category values: segments are referenced by column and rank, and labels are rendered only locally. P12/P13 add a T-PRIV-01-style assertion on those payloads. | proposed — required by I-6 (P12/P13) |
| PI-17 | §22.5, T-ART-06 | Parity fixture. (a) CatBoost native categoricals (§15.2) keep no recoverable label list in the model, so "use only categories present in the fitted encoder" doesn't work for CatBoost pipelines. (b) In low-dimensional, all-categorical data a synthetic row can match a real DEV/HOLD row by chance, which fails T-ART-06 or genuinely reproduces a row. | (a) Take CatBoost-column categories from the frequent-label map that the pipeline's rare-grouping step already stores in `model.joblib`. (b) Re-sample (seeded) any synthetic row whose hash matches a DEV/HOLD row hash. | proposed, no behavior change (P14) |
| PI-18 | §22.2, §22.4, §25.1, §25.4, T-ART-01, T-SEC-01 vs A-001 | A-001 knock-on text: §22.2 doesn't list `tabpilot_runtime/`. §22.4 says "Imports only packages listed in `requirements.txt`" and asks for op code "copied from the library's reference implementation". §25.1 and §25.4 lack the package and its contract. | Read these as amended by A-001 (see ADR-0002): the bundle adds `tabpilot_runtime/`, `pipeline.py` imports it (no copying), the tree and contract include it, T-ART-01 lists it, and T-SEC-01 scans it. | resolved by A-001 |
| PI-19 | §27.4, §1.6/SM-7, FR-PRE-01 vs §31.3 | Test-ID meaning conflicts. §27.4: "T-UI-03 checks every system event type has a corresponding state change", but in §31.3 T-UI-03 is "LLM rationale labeled" and the event check is T-RUN-01. SM-7 / §1.6 measure "fact-check pass rate (T-RPT-03)", but §31.3 T-RPT-03 is "causal phrase rejected". FR-PRE-01 names batch-vs-row consistency T-PRE-03, while §31.3 lists it fifth. | §31.3 is canonical except T-PRE-03 = batch vs row (FR-PRE-01 explicit; others reorder, see TRACEABILITY). The event/state check is T-RUN-01. SM-7 is measured from run records and supported by T-RPT-02..04. | proposed, no behavior change (DEC-006) |
| PI-20 | §31.3 | Some range IDs have no per-ID definition: T-ING-09..13, T-TGT-07..10, T-PRF-01..04 (three topics for four IDs), T-PLN-01..02, T-PLN-03..08. | The owning prompt assigns per-ID meanings in test docstrings and updates TRACEABILITY. For T-PLN-03..08, use the plan-fixture outcomes in `fixtures/MANIFEST.yaml` (P03). | proposed, no behavior change |
| PI-21 | §29.7 vs §9.4, §23.7, P11 | Token limits. "LLM input tokens per call ≤ 6,000 (estimated)", but a planner request = system prompt with embedded JSON Schema (P11 target ≤ ~3,000 tokens) + compact profile (≤ 14,000 chars ≈ 3,500 tokens) + DSL catalog + constraints, which can exceed 6,000. Worst-case calls per run (plan+repair, 2×(reflect+repair), narrate+repair) can also exceed the 40,000 input tokens/run. | P11 measures the real request size. If the estimate is > 6,000, shrink the compact profile with the existing `max_chars` truncation, not the limits. Exhausting the per-run budget ends LLM use and falls back (FR-LLM-02). | proposed → Q-14 |
| PI-22 | §3.2 stage 7 vs §24.7 | "Re-run with changes" after results needs the data, but §24.7 deletes "Raw input and `normalized.parquet` … immediately after export". | Re-run requires re-upload; the UI keeps the previous configuration and says why. (Alternative: keep normalized input until idle TTL.) | proposed → Q-08 |
| PI-23 | US-08 vs §12, §16, §39 | US-08 wants "with LLM features" vs "without" leaderboard rows for the best model family, but no section says which candidates get both variants. That affects budget and the §19.4 retraining scope. | In iteration 0, train the baseline-only variant for the highest-priority model family only. All other families use the plan's features. | proposed → Q-12 |
| PI-24 | §18.1 | `entry_id = {iteration}-{model}-{feature_set_hash[:8]}` ignores encoding, class weight and target transform. Two candidates of one model with the same features but different preprocessing in one iteration would collide. | The hash covers the full candidate configuration (features + preprocessing + class weight + target transform); the format stays the same. | proposed, no behavior change (P09) |
| PI-25 | §23.1, §13.3, §26 | Pydantic v2 details. Fields `model_id`, `model_ids` and `model_size_bytes` hit the protected `model_` namespace (warnings). `LLMRequest.response_schema: type[BaseModel]` is not JSON-serializable, which breaks the JSON Schema export step. `RunState.warnings: list[Warning]` shadows the builtin `Warning`. | `protected_namespaces=()` on affected models; exclude `response_schema` from serialization/schema export (or carry the schema name); name the class `RunWarning`. | proposed, no behavior change (P02) |
| PI-26 | §34.2 vs §16.1, §23.3, §30.2, FR-RUN-03 | Dependencies. YAML config files need PyYAML, which is not in §34.2. §30.2 allows `structlog`, also not listed. FR-RUN-03 needs an OS file lock. | P01 adds `pyyaml` with a Decision Log entry. P02 uses the stdlib JSON logging formatter unless `structlog` is approved. P15 uses stdlib `fcntl` (Linux/macOS targets). | proposed, no behavior change (P01/P02/P15) |
| PI-27 | FR-PRF-01 vs §29.3, §9.5, §9.2 | Profiling time. FR-PRF-01 gives **profiling** "≤ 5% of run budget or ≤ 30 s, whichever is larger". §29.3 gives 5% to ingest + split + profile together. The §9.5 per-group caps (10 s hosted × ~10 groups) plus the 30 s leakage-scan cap can add up to more. | Profile total ≤ max(5% budget, 30 s), with per-group caps scaled down to fit (the leakage scan counts inside it). | proposed, no behavior change (P06) |
| PI-28 | §8.10 vs §11.5, §17.3 | Small-data tuning cap: "50–200 rows … tuning trials capped at 20" vs "below 2,000 rows … ≤ 20" vs "≤ 20 trials per model". | ≤ 20 trials **per model** whenever there is no holdout (n_valid < 2,000). | proposed, no behavior change (P09) |
| PI-29 | §11.3, §8.7 | With n_valid ≥ 2,000 and a class of 2–4 rows, the stratified holdout can leave < 2 rows of that class in DEV, so K-fold is impossible. §11.3 only handles "smallest class count < K". | Compute K reduction on DEV counts after the holdout. If DEV's smallest class is < 2 → E-TARGET-CLASS-TOO-SMALL with the §8.7 remove-rows offer. | proposed, no behavior change (P05) |
| PI-30 | §22.2 vs §26.11, §30.1 | `metadata/facts.json` (§26.11 `facts_ref`) and the §30.1 records ("included (sanitized) in the bundle's metadata/") are missing from the §22.2 contents table that T-ART-01 checks. | The bundle file list = §22.2 + `metadata/facts.json` + sanitized §30.1 records + `tabpilot_runtime/` (A-001). | proposed, no behavior change (P14) |
| PI-31 | §26.13, US-07, §12.9, §19.4, §23.9 | Event types are named across many sections (`stage_completed`, `candidate_completed`, `planner_fallback`, `reflection_skipped`, `llm_*`, …) but never catalogued. T-RUN-01 needs a closed list. | P02 defines `RunEvent.type` as an enum. Prompts that add events extend it. | proposed, no behavior change (P02) |
| PI-32 | §14.4, §29.7, §12.2 | "≤ 12 per plan; ≤ 6 new in a reflection round" lets active LLM features reach 24 after two rounds. There is no total cap. | Active LLM features per candidate ≤ 12 at all times; a reflection that adds features must stay within 12. | proposed → Q-13 |
| PI-33 | §37.1 D-02 vs PROMPTS P15 | D-02 (worker process viability) is "required before implementation … spike in Phase 1", but the build plan first spikes it in P15 (risk R-16, late discovery). | Optional early spike (a few lines in P01 or P09). Otherwise keep the P15 "spike first" step. | proposed → Q-15 |
| PI-34 | §37.1 D-13 vs PROMPTS P01 | The repository license is "required before implementation", but P01 has no LICENSE deliverable. | P01 adds an MIT `LICENSE` (D-13 default, DEC-002). | proposed → Q-16 |
| PI-35 | §25.1, §14.7, T-SEC-08 vs D-09 | §25.1 lists `sandbox/`, but D-09 defaults to "Not built in v1", and P01 forbids placeholder code. | P01 does not create `tabpilot/sandbox/`. T-SEC-08 asserts `import tabpilot.sandbox` fails in every build. Contract rule 4 stays as a forbidden-module rule. | proposed, no behavior change (P01/P17) |
| PI-36 | FR-RUN-01 | Cites "§27.5" for "progress events are emitted only by code that performed the action". That rule is in §27.4; §27.5 is responsiveness/accessibility. | Read as §27.4. | doc reference error |
| PI-37 | §10.1 | Task detection doesn't cover float targets whose values are all integers (e.g. 1.0, 2.0), or string targets with > 50 distinct values. | Floats with all-integer values follow the integer rules. Strings with > 50 distinct → E-TARGET-TOO-MANY-CLASSES (§8.7). | proposed, no behavior change (P05) |
| PI-38 | FR-ING-06 AC, US-F1, §31.3 | Requirements with no test ID: FR-ING-06 "Each sample completes a run within the SM-2 budget on REF-HOSTED" (T-E2E-01..03 use fixtures, not samples). E-FILE-TYPE, E-FILE-EMPTY, E-FILE-CORRUPT, E-PARQUET-NESTED, E-TOO-MANY-COLUMNS/ROWS and E-EXPORT-PARITY have no dedicated T-ID, although US-F1 needs each E-code triggered by a fixture. | Owning prompts add tests named after the E-code (TRACEABILITY App. A). P16 adds a local stub-LLM run of each sample. P20 records the live smoke run (G-14). | proposed, no behavior change |
| PI-39 | §27.1–§27.2 vs §29.1 | Time-budget slider is fixed at 120–900 s, but the local profile allows up to 3,600 s. | Slider max = the active profile's maximum. | proposed, no behavior change (P16) |

---

## 12. Open questions for the owner  *(replace)*

| ID | Question | Blocking? | Default if unanswered |
| --- | --- | --- | --- |
| Q-01 | Verify Streamlit Community Cloud resources (D-01) with a probe app | Before P18 perf calibration | Keep 1 GB design assumption |
| Q-02 | Provider order and model IDs for hosted (D-03) | Before deployment (P20) | Groq first; IDs from provider pages at deploy time |
| Q-03 | Sample and benchmark dataset licenses (D-05) | Before bundling samples (P16) and P19 | Replace any non-redistributable dataset with an OpenML equivalent |
| Q-04 | PI-07: `leaky_binary.csv` copy at 0.5% noise cannot reach the 0.999 near-deterministic threshold. Change the fixture noise or the threshold? | **Yes, before P03** | Fixture copy with ≤ 0.05% flips; thresholds unchanged |
| Q-05 | PI-06: base the 2,000-row holdout/repeated-CV threshold on valid rows before splitting? | **Yes, before P05** | Yes: n_valid basis for all small-data rules |
| Q-06 | PI-01…PI-05: reconcile the time-budget ledger (reflection share across rounds, reflection minimum, ensemble timeout, 85% stop for the reserved slice, planning deadline) | **Yes, before P09** (ledger), P12, P15 | As proposed in PI-01…PI-05 |
| Q-07 | PI-15: may numeric min/max/quantiles of feature columns be sent to the LLM in standard mode? | Before P06 | Yes (§9.4); disclosed in the privacy notice; minimal mode removes them |
| Q-08 | PI-22: "Re-run with changes" after completion — require re-upload, or keep normalized input until idle TTL? | Before P16 | Require re-upload |
| Q-09 | PI-11: missing-indicator threshold, 5% or 30%? | Before P07 | 5% |
| Q-10 | PI-12: `log1p` on negative inputs — NaN + counter, or clip at 0? | Before P07 | NaN + counter, flagged at validation |
| Q-11 | PI-09: treat > 50% missing target as a warning with confirmation (W-TARGET-MISSING) instead of E-TARGET-TOO-FEW-ROWS? | Before P02 | Yes |
| Q-12 | PI-23: which model families get both "with" and "without LLM features" rows (US-08)? | Before P11 | Highest-priority family only |
| Q-13 | PI-32: total cap on active LLM features across reflection rounds? | Before P12 | 12 at all times |
| Q-14 | PI-21: if the planner request exceeds 6,000 estimated tokens, shrink the profile or raise the limit? | Before P11 | Shrink the profile via `max_chars` |
| Q-15 | PI-33: run the D-02 worker-process spike early (P01/P09) instead of at P15? | No | Keep the P15 "spike first" step |
| Q-16 | PI-34 / D-13: add an MIT `LICENSE` in P01? | Before P01 (non-blocking) | Yes, MIT |

---

## 13. Known issues and tech debt  *(append-only; mark resolved, don't delete)*

| ID | Date | Prompt | Issue | Severity | Status |
| --- | --- | --- | --- | --- | --- |
| *(none yet)* | | | | | |

---

## 14. Unverified items  *(append-only; mark verified when done)*

Things that could not be verified in a session (no Docker, no network, no provider key, etc.). CI or the owner must verify them.

| ID | Date | Prompt | Item | Why unverified | How to verify | Status |
| --- | --- | --- | --- | --- | --- | --- |
| *(none yet)* | | | | | | |

---

## 15. Environment notes  *(replace)*

| Item | Value |
| --- | --- |
| Setup | `uv sync --all-extras` |
| Full check | `make check` |
| Tests only | `uv run pytest -m "not perf and not live and not slow"` |
| Network in AI sessions | Unknown — record per session in Unverified Items if installs or provider calls fail |
| Docker in AI sessions | Unknown — perf and container tests may need CI |

---

## 16. Audit log  *(append-only; written by U1)*

| Date | Prompt audited | Verdict | Findings (IDs) | Fixed by |
| --- | --- | --- | --- | --- |
| *(none yet)* | | | | |

---

## 17. Change log  *(append-only)*

| Date | Prompt | Change |
| --- | --- | --- |
| 2026-10-19 | kit | Created CLAUDE.md, CONTEXT.md, docs/PROMPTS.md; PRD placed at docs/PRD.md |
| 2026-10-19 | P00 | Added `docs/TRACEABILITY.md` (FR 36/36, T 136/136, NFR 27/27, G 18/18 mapped; E-code and W-code appendices), `docs/adr/0000-template.md`, `0001-baseline-decisions.md`, `0002-runtime-package.md`; recorded PRD issues PI-01…PI-39 (§11) and open questions Q-04…Q-16 (§12); DEC-006 |
| 2026-10-19 | P00 | Ownership assigned where no prompt, or more than one, claimed an item: T-SEC-05 → P17 (no prompt listed it individually); T-SEC-08 and FR-FE-03 → P17; NFR-DEF-01, NFR-POR-02, NFR-COST-01 → P20; T-FE-08 → P08 (fold restart lives in the CV runner); T-EVL-06 → P13 (P09 contributes); T-LLM-05 → P15 (P11 contributes); T-TGT-11 → P11 (P05 contributes); T-LEAK-03 → P09 (P05 guard-level); T-LEAK-07 → P08 (P07 DSL ops); T-SEC-02 → P13 (P11 plan part); T-RFL-02 → P15; T-ENS-04 → P12 (P14 contributes) |

---

## 18. Handoff note  *(replace at the end of every session)*

**Latest (P00, 2026-10-19):** P00 is complete. It is documentation only: no code, config or dependency files exist yet.

- **Next:** start **P01** (repository scaffold). Read `docs/TRACEABILITY.md` for who owns what.
- **P01 must also apply:**
  - PI-13: import-contract rule 2 allows the guard's definition module and one construction site.
  - PI-18: add the `tabpilot_runtime` package and its contract (A-001 / ADR-0002).
  - PI-26: `pyyaml` needs a Decision Log entry.
  - PI-35: do not create `tabpilot/sandbox/`.
  - Q-16: MIT `LICENSE`.
- **Owner decisions needed before later prompts:** Q-04 (before P03, leaky-fixture noise), Q-05 (before P05, holdout row basis), Q-06 (before P09, time-budget ledger). Each has a recommended default in §12. If the owner does not answer, the prompt that hits the question must stop and ask (Deviation Protocol), not assume.
- **No-behavior-change items:** these PI entries are marked "proposed, no behavior change". The owning prompt may adopt them, recording a Decision Log entry with status `accepted`.
- **Other inputs:** ADR-0001 summarizes C-01…C-17 and D-01…D-16; ADR-0002 records A-001. `docs/adr/0000-template.md` is the ADR template.
