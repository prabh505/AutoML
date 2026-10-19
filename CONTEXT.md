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
| Last completed prompt | **P01** Repository scaffold, toolchain, CI skeleton |
| Next prompt | **P02** Data contracts, configuration, logging |
| Current milestone | Milestone 0 — Foundations |
| Test suite | `make check` green: ruff, format, mypy, import-linter (5 contracts), pytest 8 passed, coverage 100% (10 statements) |
| Last green commit | P01 commit (`git log --grep '^P01:'`) |
| Blockers | None. Owner answered Q-04…Q-16 (ADR-0003) |

---

## 4. Prompt ledger  *(replace status cells; keep rows)*

Status values: `pending` · `in_progress` · `complete` · `complete_with_issues` · `blocked`

| ID | Title | PRD sections | Milestone / gate | Status | Commit | Date |
| --- | --- | --- | --- | --- | --- | --- |
| P00 | Project charter and context bootstrap | All (§1–§40) | M0 | complete | see `git log --grep '^P00:'` | 2026-10-19 |
| P01 | Repository scaffold, toolchain, CI skeleton | §25, §34.1–34.2, §34.5 | M0 | complete | see `git log --grep '^P01:'` | 2026-10-19 |
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
| Package skeleton (all §25.1 packages + `tabpilot_runtime`) | P01 | complete | docstrings with responsibilities/forbidden; `tabpilot.ui.main()` shell |
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
| python | 3.11.17 (session) | PRD §34.2; `requires-python >=3.11,<3.12` |
| uv | 0.11.32 | Lockfile + env manager (PRD §34.1) |
| streamlit | 1.66.0 | UI and hosting target; `st.fragment` available for P16 |
| pandas | 3.0.6 | Dataframes; note pandas 3 (copy-on-write, `str` dtype default) — P04 string-first parsing must account for it |
| numpy | 2.4.6 | Arrays; newest release supporting 3.11 |
| pyarrow | 25.0.1 | Parquet reading (§8.1) |
| scikit-learn | 1.9.1 | Pipelines, `TargetEncoder`, `OneHotEncoder(min_frequency, max_categories)`, `set_output`, `StratifiedGroupKFold` (all verified to import) |
| lightgbm | 4.7.0 | Primary model (§16.1); imports on 3.11 |
| optuna | 5.0.0 | Tuning (§17); imports on 3.11 |
| shap | 0.51.0 | Explainability (§21.3); newest supporting 3.11 (0.54 needs 3.12) |
| langgraph | 1.2.16 | Run graph (§13) |
| pydantic | 2.14.1 | Contracts (§26) |
| openai | 3.35.1 | OpenAI-compatible client for Groq/Gemini/OpenRouter/Ollama (§23.2) |
| charset-normalizer | 3.5.2 | Encoding detection (§8.2) |
| psutil | 7.2.2 | RSS monitor (§29.6) |
| joblib | 1.6.0 | Model serialization (§22.2) |
| jinja2 | 3.1.6 | Report + `pipeline.py` templates |
| matplotlib | 3.11.2 | Report figures (§21.2) |
| python-ulid | 4.0.1 | `run_id` (§0.4) |
| pyyaml | 6.0.3 | Config YAML loading (DEC-020) |
| xgboost / catboost (extras) | 2.1.4 / 1.2.10 | Optional models, local profile (C-04); xgboost 3.x needs 3.12 |
| mlflow / flaml (extras) | 3.17.1 / 2.7.0 | Benchmark + local tracking (C-11, §32) |
| dev: pytest / pytest-cov / pytest-timeout / pytest-socket / hypothesis | 9.1.1 / — / — / 0.8.1 / 6.170.2 | Tests; network blocked by default |
| dev: ruff / mypy / import-linter / pip-audit / pre-commit | 0.17.1 / 2.4.0 / 2.15 / 2.10.1 / — | Lint+format, types, contracts, audit |
| dev: pandas-stubs, types-PyYAML, types-psutil | 3.0.5 / — / — | mypy stubs (DEC-020) |

Exact pins for every package: `uv.lock`; hosted subset: `requirements.txt` (exported, checked in CI).

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
| DEC-007 | 2026-10-19 | P00 follow-up | Q-04: leakage thresholds — classification near-deterministic AUC ≥ 0.95 (excluded), strong ≥ 0.90 (warn); regression R² ≥ 0.90 / ≥ 0.80 | Owner choice; fixture keeps 0.5% noise | accepted (owner) | ADR-0003, A-002 |
| DEC-008 | 2026-10-19 | P00 follow-up | Q-05: all small-data rules use valid rows before split (n_valid ≥ 2,000 → holdout + single K-fold; else repeated 3×5 CV, ≤ 20 trials/model) | Removes §11.1/§11.3 contradiction | accepted (owner) | ADR-0003, A-003 |
| DEC-009 | 2026-10-19 | P00 follow-up | Q-06: time budgets and timeouts doubled; §29.3 ledger governs (shares 5/5/40/15/5/30, stop at 70%, hosted default 960 s) | Owner: generous first, tighten later | accepted (owner) | ADR-0003, A-004 |
| DEC-010 | 2026-10-19 | P00 follow-up | Q-07: numeric quantiles incl. min/max of feature columns allowed in standard mode; disclosed in privacy notice | Planner quality; minimal mode removes | accepted (owner) | ADR-0003, A-005 |
| DEC-011 | 2026-10-19 | P00 follow-up | Q-08: re-run after completion requires re-upload (data deleted after export per §24.7) | Keeps PRD privacy behavior | accepted (owner) | ADR-0003 |
| DEC-012 | 2026-10-19 | P00 follow-up | Q-09: missing indicators at ≥ 5% missing | §14.2/§12.8 agree | accepted (owner) | ADR-0003, A-006 |
| DEC-013 | 2026-10-19 | P00 follow-up | Q-10: log1p negatives → NaN + counter, flagged at validation; no clipping | No silent value change | accepted (owner) | ADR-0003, A-007 |
| DEC-014 | 2026-10-19 | P00 follow-up | Q-11: target > 50% missing → W-TARGET-MISSING with confirmation; E-TARGET-TOO-FEW-ROWS only for < 50 rows | One meaning per code | accepted (owner) | ADR-0003, A-008 |
| DEC-015 | 2026-10-19 | P00 follow-up | Q-12: iteration 0 trains top 2 (hosted) / top 3 (local) plan-priority families with and without LLM features | Owner: top 2–3 | accepted (owner) | ADR-0003, A-009 |
| DEC-016 | 2026-10-19 | P00 follow-up | Q-13: ≤ 12 active LLM features per candidate at all times | Bounded cost | accepted (owner) | ADR-0003, A-010 |
| DEC-017 | 2026-10-19 | P00 follow-up | Q-14: keep 6,000-token per-call cap; shrink compact profile to fit | No limit change | accepted (owner) | ADR-0003 |
| DEC-018 | 2026-10-19 | P00 follow-up | Q-15: small worker-process (spawn) spike in P01 | Early R-16 signal | accepted (owner) | ADR-0003 |
| DEC-019 | 2026-10-19 | P00 follow-up | Q-16: MIT license (D-13) | Recommended default | accepted (owner) | ADR-0003 |
| DEC-020 | 2026-10-19 | P01 | Dependencies beyond §34.2: `pyyaml` (runtime; config YAML loading, PI-26) and dev-only type stubs `pandas-stubs`, `types-PyYAML`, `types-psutil` | Needed for config loading and strict mypy; no runtime behavior change | accepted | PI-26 |
| DEC-021 | 2026-10-19 | P01 | Scaffold details: (1) no `tabpilot/sandbox/` (D-09, PI-35); §25.4 rule 4 is enforced by a smoke test now and T-SEC-08 later, because import-linter can't target a missing module; (2) contract R2 forbids all of `tabpilot.splitting` outside `evaluation` and `orchestration` until P05 creates `splitting.guard` and narrows it (PI-13); (3) `config`/`logging` are optional layers until P02 creates them; (4) ruff excludes Markdown, because ruff 0.17 formats code blocks in `.md` and would rewrite the immutable PRD; (5) `tabpilot.config`/`tabpilot.logging` modules are left to P02 (no placeholder files) | Implementation details; no behavior change | accepted | ADR n/a |
| DEC-022 | 2026-10-19 | P01 | D-02 spike (DEC-018) in this Linux container: spawn child started and sent its first message in 0.05 s, heartbeats over `multiprocessing.Queue` worked, cooperative stop exited 0, a SIGTERM-ignoring child was killed by SIGKILL (exit -9) instantly. The only remaining child was the multiprocessing `resource_tracker`, so P15's no-orphan check (T-RUN-02) must exclude it | Process mode is viable here; Streamlit Cloud still unverified (Q-01/D-01) | accepted | R-16 |

---

## 10. PRD amendments  *(append-only)*

Each amendment overrides the PRD only for the exact item named.

| ID | PRD ref | Amendment | Reason | Status |
| --- | --- | --- | --- | --- |
| A-001 | §15.1, §20.5, §22.2, §22.4, §25.1 | Add `src/tabpilot_runtime/` — a small package with **no dependency on `tabpilot`** (only numpy, pandas, scikit-learn, and the model libraries). It contains every custom class that can appear inside a fitted pipeline: `ColumnNameMapper`, `TypeCaster`, `BaselineFeatureTransformer`, `DSLFeatureTransformer` and DSL op implementations, `ColumnSelector`, frequency/rare-category encoders, the early-stopping estimator wrapper, and `VotingEnsemble`. `tabpilot` imports these from `tabpilot_runtime`. The export bundle ships a copy of `tabpilot_runtime/` next to `pipeline.py`, so `model.joblib` loads with only the pinned third-party packages installed. "Standalone `pipeline.py`" therefore means "standalone bundle". | Pickle references classes by module path | accepted |
| A-002 | §11.7, §9.3, FR-PRF-02 | Leakage levels: classification AUC near-deterministic ≥ 0.95 (excluded by default, re-includable), strong ≥ 0.90 (warn); regression R² ≥ 0.90 / ≥ 0.80. | Owner (Q-04, ADR-0003) | accepted |
| A-003 | §11.1, §11.3, §11.5, §8.10 | Holdout and small-data rules use n_valid (valid rows before split): ≥ 2,000 → 20% holdout + single K-fold; else no holdout + repeated 3×5 CV, ≤ 20 tuning trials per model. | Owner (Q-05) | accepted |
| A-004 | §29.1, §29.3, §29.7, §13.4, §13.5, §13.7, §9.2, §9.5, §17.3, §19.4, §23.7, §27.2 | Budgets and timeouts doubled; ledger shares 5/5/40/15/5/30 (reserved 30%); training/reflection/ensemble stop at 70%; exact values in ADR-0003. SM-2/T-PERF-01 measured with `time_budget_s=480` (KI-001). | Owner (Q-06) | accepted |
| A-005 | FR-PRF-03 AC, §9.4 | Numeric quantiles incl. min/max of feature columns are permitted aggregates in standard mode; "no cell values" is scoped per T-PRIV-01; privacy notice discloses numeric ranges. | Owner (Q-07) | accepted |
| A-006 | §8.6 | Missing indicator added at ≥ 5% missing; 30–95% triggers W-HIGH-MISSING only. | Owner (Q-09) | accepted |
| A-007 | §14.3 (`log1p`) | No clipping; flagged at static validation if profile min < 0; negatives → NaN + counter. | Owner (Q-10) | accepted |
| A-008 | §8.7 | Target > 50% missing → W-TARGET-MISSING requiring confirmation (not an E-code). | Owner (Q-11) | accepted |
| A-009 | US-08, §12, §16 | Iteration 0 trains top 2 (hosted) / top 3 (local) plan-priority families both with and without LLM features. | Owner (Q-12) | accepted |
| A-010 | §14.4, §29.7 | ≤ 12 active LLM features per candidate at all times (reflection adds ≤ 6 within the cap). | Owner (Q-13) | accepted |

---

## 11. PRD issues found  *(append-only; P00 seeds this)*

| ID | PRD ref | Issue | Proposed resolution | Status |
| --- | --- | --- | --- | --- |
| PI-01 | §29.3 vs §19.4 | Reflection time. §29.3: "Reflection (incl. retraining) ≤ 20%". §19.4: "Maximum additional runtime: 25% of the original total budget per round". With `max_rounds` = 2 that allows up to 50%, and the §29.3 shares (5+5+50+20+5+15) would exceed 100%. | The §29.3 ledger governs: **all** reflection rounds together use ≤ 20% (plus roll-forward). Each round gets min(25% of total, remaining reflection share, remaining − reserved 15%). | accepted (A-004) |
| PI-02 | §13.5, §19.4 | Reflection minimum. Both say reflection is skipped when "remaining budget" < 20% of total, but neither says whether "remaining" includes the reserved 15% slice. If it does, reflection can start with only 5% usable time. | "Remaining" means remaining **non-reserved** time (time to the 85% mark). Skip reflection if that is < 20% of the total. | accepted (A-004) |
| PI-03 | §13.4 vs §29.3 | Ensemble node timeout is "10% of budget" (§13.4), but the ledger share is "Ensemble ≤ 5%" (§29.3). | Ensemble timeout = its ledger allocation (5% + roll-forward). | accepted (A-004) |
| PI-04 | §13.4, §13.7 vs §29.3 | Reserved slice. 15% of the 480 s default = 72 s. That slice must cover finalize (refit ≤ 2× fold fit + 30 s), explain (60 s), report (60 s) and export (60 s), which is ≥ 210 s at the node-timeout ceilings. §13.7 says training stops "at 100%" and then jumps to finalize, but §29.3 says the reserved 15% "is never consumed by earlier stages". | Training, reflection and ensemble stop at the 85% mark. Reserved-slice nodes use min(node timeout, time left to 100%). When time is short, explain falls back to gain importance and the report to the template narrative. The 130% hard kill stays as the backstop. T-RUN-03 tests the 85% stop. | accepted (A-004) |
| PI-05 | §29.3 vs §23.7, §13.4, §12.5 | Planning share "≤ 5%" is 24 s at the hosted default. But one planner call has a 45 s per-call timeout (§23.7), up to 3 attempts × 3 providers (§23.5), and up to 2 planner calls (§12.5), while the plan node timeout is 60 s (§13.4). | Plan node deadline = its ledger share + roll-forward from ingest/profile, capped by the 60 s node timeout. The router stops when the deadline would be exceeded, and the planner falls back to the rule-based plan (§12.9). The LLM wall-time ceiling (20%, §23.7) stays a separate global limit. | accepted (A-004) |
| PI-06 | §11.1 vs §11.3, §11.5 | Holdout row basis. §11.1 bases the holdout on all valid rows ("rows ≥ holdout_min_rows (2,000)"). §11.3 uses "DEV rows < 2,000 (no holdout)". With 2,000–2,499 valid rows, a holdout is created, DEV has 1,600–1,999 rows, and §11.3 then calls for repeated CV with no holdout — a contradiction. | Evaluate every small-data threshold on **valid rows before splitting** (n_valid): n_valid ≥ 2,000 → 20% holdout + single K-fold; otherwise no holdout + repeated 3×5 CV. W-SMALL-DATA stays on DEV rows (§9.3). | accepted (A-003) |
| PI-07 | FR-PRF-02 AC, §31.2, T-LEAK-05 vs §11.7 | `leaky_binary.csv` "contains a copy of the target with 0.5% noise". The univariate AUC of a binary copy with flip rate p is ≈ 1 − p = 0.995 (computed analytically in P00). That is below the near-deterministic threshold "≥ 0.999", so the column is only "strong" and **T-LEAK-05 cannot pass as specified**. | Keep the §11.7 thresholds (D-15 default). Generate the copy with ≤ 0.05% flips (AUC ≈ 0.9995). The strong-but-legitimate column (AUC 0.95–0.99, P03) still covers T-LEAK-06. Alternative (owner): lower the classification threshold to 0.99. | accepted (A-002) |
| PI-08 | T-LEAK-08 (§31.3) | The wording is unclear: "Feature = target mean of rows *not* in training fold would score high if leaked". It doesn't say the row count, the category cardinality, or whether "≤ 0.55 (5 seeds)" applies per seed or to the mean. Also, the rule-based plan disables target encoding (§12.8), so the canary must force it. | Pure-noise binary target, n_dev = 2,000, one categorical with 500 levels (> 50, so target encoding applies) + 3 noise numerics. Candidate forced to `high_cardinality = target`. Assert OOF ROC-AUC ≤ 0.55 for **each** of seeds 0–4. Positive control: an encoder fitted on all of DEV outside the folds must score > 0.55, which proves the test can fail. | proposed, no behavior change (P08 adopts; non-blocking) |
| PI-09 | §8.9, §8.7 | E-codes. All 16 specific codes used in the PRD appear in §8.9, and the wildcards (E-FILE-\*, E-PARSE-\*, E-TARGET-\*) resolve to them. But the six E-TARGET-\* codes have no message template ("Specific per §8.7"). E-TARGET-TOO-FEW-ROWS is also used two ways: as a hard error (< 50 rows) and as a "warning requiring confirmation" (> 50% target missing). | P02 writes what/where/fix templates for each E-TARGET-\* code from §8.7. > 50% missing target becomes W-TARGET-MISSING with required confirmation. E-TARGET-TOO-FEW-ROWS is kept for < 50 rows only. | accepted (A-008) |
| PI-10 | §9.3 vs §8.2, §8.5–§8.7, §11.1 | W-ENCODING-FALLBACK, W-COERCED, W-INF, W-TARGET-MISSING and W-NO-HOLDOUT are used but not catalogued (§9.3 lists 10 of the 15 W-codes). | The P02 `WarningCode` enum covers all 15 (list in `docs/TRACEABILITY.md` App. B). | proposed, no behavior change (P02) |
| PI-11 | §8.6 vs §14.2, §12.8 | Missing indicators. §8.6: "Missing rate 30–95% → Warning; missing indicator added by default". §14.2 and §12.8: indicators for missing ≥ 5%. | Missing indicator at ≥ 5% (baseline transform and fallback plan agree). 30% only triggers the W-HIGH-MISSING warning. | accepted (A-006) |
| PI-12 | §14.3 vs §14.5 | `log1p` with negatives. §14.3: "requires profile min ≥ 0; else clipped at 0 and flagged". §14.5: "log1p/sqrt of negative → NaN with counter". | No silent clipping. Static validation flags the spec when the profile min is < 0 (recorded in provenance). At transform, negative inputs → NaN with a counter (§14.5), handled by the imputer downstream. | accepted (A-007) |
| PI-13 | §11.2 L-2, §7.1 vs §21.4, §25.4 rule 2 | Holdout access. L-2 and §7.1 let the "post-freeze Explainer" read HOLD for error analysis. §21.4 says error analysis is "computed on DEV OOF predictions (no holdout use)", and §25.4 rule 2 lets only `evaluation.holdout` and `orchestration.nodes.finalize` import the guard. The module that **constructs** the guard (worker pre-graph setup; P09 CLI) is not allowed either. | The Explainer never touches HOLD (§21.4 wins; this is stricter). Contract rule 2 allows `splitting` (definition), `evaluation.holdout`, `orchestration.nodes.finalize`, and one named construction site (the run-context builder used by the P09 CLI and the P15 worker). P01 encodes this. | partly applied in P01 (contract R2); P05 narrows to `splitting.guard` |
| PI-14 | I-1, T-LEAK-03 vs §11.2 L-7, FR-EVL-02 | The holdout is "opened exactly once", yet it must score both the final pipeline **and** the reference baseline. | One `open()` call returns the HOLD indices, and the Finalizer scores both models from that single access. Any second `open()` raises. | proposed, no behavior change (P05/P09) |
| PI-15 | FR-PRF-03 AC vs §9.4, §26.2 | FR-PRF-03 says the compact profile "contains no cell values from the dataset (only aggregates)". But `q` = [min, p5, …, max] includes the exact min and max (and often the median) of numeric columns, which are literal cell values. T-PRIV-01 only checks identifier and categorical columns. | Follow §9.4: numeric order statistics of feature columns are allowed aggregates in standard mode, and minimal mode removes them. Read the FR-PRF-03 AC within T-PRIV-01's scope. The privacy notice states that numeric ranges are sent. | accepted (A-005) |
| PI-16 | §19.2, §21.7, §21.4 vs §24.4, I-6 | Category labels can reach the LLM through derived names. ReflectionInput's "top 15 features … (names)" uses post-preprocessing names, and one-hot names embed labels (e.g. `contract_Month-to-month`). The narrator's facts table includes "feature names", and §21.4 worst segments are category values. §24.4 says category labels are never sent. | Importances in ReflectionInput are aggregated to source columns (as §21.3 does for charts). The facts table sent to the LLM excludes category values: segments are referenced by column and rank, and labels are rendered only locally. P12/P13 add a T-PRIV-01-style assertion on those payloads. | proposed — required by I-6 (P12/P13) |
| PI-17 | §22.5, T-ART-06 | Parity fixture. (a) CatBoost native categoricals (§15.2) keep no recoverable label list in the model, so "use only categories present in the fitted encoder" doesn't work for CatBoost pipelines. (b) In low-dimensional, all-categorical data a synthetic row can match a real DEV/HOLD row by chance, which fails T-ART-06 or genuinely reproduces a row. | (a) Take CatBoost-column categories from the frequent-label map that the pipeline's rare-grouping step already stores in `model.joblib`. (b) Re-sample (seeded) any synthetic row whose hash matches a DEV/HOLD row hash. | proposed, no behavior change (P14) |
| PI-18 | §22.2, §22.4, §25.1, §25.4, T-ART-01, T-SEC-01 vs A-001 | A-001 knock-on text: §22.2 doesn't list `tabpilot_runtime/`. §22.4 says "Imports only packages listed in `requirements.txt`" and asks for op code "copied from the library's reference implementation". §25.1 and §25.4 lack the package and its contract. | Read these as amended by A-001 (see ADR-0002): the bundle adds `tabpilot_runtime/`, `pipeline.py` imports it (no copying), the tree and contract include it, T-ART-01 lists it, and T-SEC-01 scans it. | resolved by A-001 |
| PI-19 | §27.4, §1.6/SM-7, FR-PRE-01 vs §31.3 | Test-ID meaning conflicts. §27.4: "T-UI-03 checks every system event type has a corresponding state change", but in §31.3 T-UI-03 is "LLM rationale labeled" and the event check is T-RUN-01. SM-7 / §1.6 measure "fact-check pass rate (T-RPT-03)", but §31.3 T-RPT-03 is "causal phrase rejected". FR-PRE-01 names batch-vs-row consistency T-PRE-03, while §31.3 lists it fifth. | §31.3 is canonical except T-PRE-03 = batch vs row (FR-PRE-01 explicit; others reorder, see TRACEABILITY). The event/state check is T-RUN-01. SM-7 is measured from run records and supported by T-RPT-02..04. | proposed, no behavior change (DEC-006) |
| PI-20 | §31.3 | Some range IDs have no per-ID definition: T-ING-09..13, T-TGT-07..10, T-PRF-01..04 (three topics for four IDs), T-PLN-01..02, T-PLN-03..08. | The owning prompt assigns per-ID meanings in test docstrings and updates TRACEABILITY. For T-PLN-03..08, use the plan-fixture outcomes in `fixtures/MANIFEST.yaml` (P03). | proposed, no behavior change |
| PI-21 | §29.7 vs §9.4, §23.7, P11 | Token limits. "LLM input tokens per call ≤ 6,000 (estimated)", but a planner request = system prompt with embedded JSON Schema (P11 target ≤ ~3,000 tokens) + compact profile (≤ 14,000 chars ≈ 3,500 tokens) + DSL catalog + constraints, which can exceed 6,000. Worst-case calls per run (plan+repair, 2×(reflect+repair), narrate+repair) can also exceed the 40,000 input tokens/run. | P11 measures the real request size. If the estimate is > 6,000, shrink the compact profile with the existing `max_chars` truncation, not the limits. Exhausting the per-run budget ends LLM use and falls back (FR-LLM-02). | accepted (DEC-017) |
| PI-22 | §3.2 stage 7 vs §24.7 | "Re-run with changes" after results needs the data, but §24.7 deletes "Raw input and `normalized.parquet` … immediately after export". | Re-run requires re-upload; the UI keeps the previous configuration and says why. (Alternative: keep normalized input until idle TTL.) | accepted (DEC-011) |
| PI-23 | US-08 vs §12, §16, §39 | US-08 wants "with LLM features" vs "without" leaderboard rows for the best model family, but no section says which candidates get both variants. That affects budget and the §19.4 retraining scope. | In iteration 0, train the baseline-only variant for the highest-priority model family only. All other families use the plan's features. | accepted (A-009) |
| PI-24 | §18.1 | `entry_id = {iteration}-{model}-{feature_set_hash[:8]}` ignores encoding, class weight and target transform. Two candidates of one model with the same features but different preprocessing in one iteration would collide. | The hash covers the full candidate configuration (features + preprocessing + class weight + target transform); the format stays the same. | proposed, no behavior change (P09) |
| PI-25 | §23.1, §13.3, §26 | Pydantic v2 details. Fields `model_id`, `model_ids` and `model_size_bytes` hit the protected `model_` namespace (warnings). `LLMRequest.response_schema: type[BaseModel]` is not JSON-serializable, which breaks the JSON Schema export step. `RunState.warnings: list[Warning]` shadows the builtin `Warning`. | `protected_namespaces=()` on affected models; exclude `response_schema` from serialization/schema export (or carry the schema name); name the class `RunWarning`. | proposed, no behavior change (P02) |
| PI-26 | §34.2 vs §16.1, §23.3, §30.2, FR-RUN-03 | Dependencies. YAML config files need PyYAML, which is not in §34.2. §30.2 allows `structlog`, also not listed. FR-RUN-03 needs an OS file lock. | P01 adds `pyyaml` with a Decision Log entry. P02 uses the stdlib JSON logging formatter unless `structlog` is approved. P15 uses stdlib `fcntl` (Linux/macOS targets). | accepted (DEC-020); structlog/fcntl choices remain for P02/P15 |
| PI-27 | FR-PRF-01 vs §29.3, §9.5, §9.2 | Profiling time. FR-PRF-01 gives **profiling** "≤ 5% of run budget or ≤ 30 s, whichever is larger". §29.3 gives 5% to ingest + split + profile together. The §9.5 per-group caps (10 s hosted × ~10 groups) plus the 30 s leakage-scan cap can add up to more. | Profile total ≤ max(5% budget, 30 s), with per-group caps scaled down to fit (the leakage scan counts inside it). | accepted (A-004) |
| PI-28 | §8.10 vs §11.5, §17.3 | Small-data tuning cap: "50–200 rows … tuning trials capped at 20" vs "below 2,000 rows … ≤ 20" vs "≤ 20 trials per model". | ≤ 20 trials **per model** whenever there is no holdout (n_valid < 2,000). | accepted (A-003) |
| PI-29 | §11.3, §8.7 | With n_valid ≥ 2,000 and a class of 2–4 rows, the stratified holdout can leave < 2 rows of that class in DEV, so K-fold is impossible. §11.3 only handles "smallest class count < K". | Compute K reduction on DEV counts after the holdout. If DEV's smallest class is < 2 → E-TARGET-CLASS-TOO-SMALL with the §8.7 remove-rows offer. | proposed, no behavior change (P05) |
| PI-30 | §22.2 vs §26.11, §30.1 | `metadata/facts.json` (§26.11 `facts_ref`) and the §30.1 records ("included (sanitized) in the bundle's metadata/") are missing from the §22.2 contents table that T-ART-01 checks. | The bundle file list = §22.2 + `metadata/facts.json` + sanitized §30.1 records + `tabpilot_runtime/` (A-001). | proposed, no behavior change (P14) |
| PI-31 | §26.13, US-07, §12.9, §19.4, §23.9 | Event types are named across many sections (`stage_completed`, `candidate_completed`, `planner_fallback`, `reflection_skipped`, `llm_*`, …) but never catalogued. T-RUN-01 needs a closed list. | P02 defines `RunEvent.type` as an enum. Prompts that add events extend it. | proposed, no behavior change (P02) |
| PI-32 | §14.4, §29.7, §12.2 | "≤ 12 per plan; ≤ 6 new in a reflection round" lets active LLM features reach 24 after two rounds. There is no total cap. | Active LLM features per candidate ≤ 12 at all times; a reflection that adds features must stay within 12. | accepted (A-010) |
| PI-33 | §37.1 D-02 vs PROMPTS P15 | D-02 (worker process viability) is "required before implementation … spike in Phase 1", but the build plan first spikes it in P15 (risk R-16, late discovery). | Optional early spike (a few lines in P01 or P09). Otherwise keep the P15 "spike first" step. | accepted (DEC-018) |
| PI-34 | §37.1 D-13 vs PROMPTS P01 | The repository license is "required before implementation", but P01 has no LICENSE deliverable. | P01 adds an MIT `LICENSE` (D-13 default, DEC-002). | resolved in P01 (MIT `LICENSE`) |
| PI-35 | §25.1, §14.7, T-SEC-08 vs D-09 | §25.1 lists `sandbox/`, but D-09 defaults to "Not built in v1", and P01 forbids placeholder code. | P01 does not create `tabpilot/sandbox/`. T-SEC-08 asserts `import tabpilot.sandbox` fails in every build. Contract rule 4 stays as a forbidden-module rule. | applied in P01 (DEC-021); T-SEC-08 in P17 |
| PI-36 | FR-RUN-01 | Cites "§27.5" for "progress events are emitted only by code that performed the action". That rule is in §27.4; §27.5 is responsiveness/accessibility. | Read as §27.4. | doc reference error |
| PI-37 | §10.1 | Task detection doesn't cover float targets whose values are all integers (e.g. 1.0, 2.0), or string targets with > 50 distinct values. | Floats with all-integer values follow the integer rules. Strings with > 50 distinct → E-TARGET-TOO-MANY-CLASSES (§8.7). | proposed, no behavior change (P05) |
| PI-38 | FR-ING-06 AC, US-F1, §31.3 | Requirements with no test ID: FR-ING-06 "Each sample completes a run within the SM-2 budget on REF-HOSTED" (T-E2E-01..03 use fixtures, not samples). E-FILE-TYPE, E-FILE-EMPTY, E-FILE-CORRUPT, E-PARQUET-NESTED, E-TOO-MANY-COLUMNS/ROWS and E-EXPORT-PARITY have no dedicated T-ID, although US-F1 needs each E-code triggered by a fixture. | Owning prompts add tests named after the E-code (TRACEABILITY App. A). P16 adds a local stub-LLM run of each sample. P20 records the live smoke run (G-14). | proposed, no behavior change |
| PI-39 | §27.1–§27.2 vs §29.1 | Time-budget slider is fixed at 120–900 s, but the local profile allows up to 3,600 s. | Slider max = the active profile's maximum. | proposed, no behavior change (P16) |
| PI-40 | §23.3 | The `providers.yaml` snippet is not valid YAML: `model_ids: [${TABPILOT_OR_MODEL_1}, ${TABPILOT_OR_MODEL_2}]` fails to parse (unquoted `${` inside a flow sequence). Found by the P01 smoke test. | Quote every `${ENV}` placeholder (done in `config/providers.yaml`); P02's interpolation works on the quoted strings. | resolved in P01 |

---

## 12. Open questions for the owner  *(replace)*

| ID | Question | Blocking? | Default if unanswered |
| --- | --- | --- | --- |
| Q-01 | Verify Streamlit Community Cloud resources (D-01) with a probe app | Before P18 perf calibration | Keep 1 GB design assumption |
| Q-02 | Provider order and model IDs for hosted (D-03) | Before deployment (P20) | Groq first; IDs from provider pages at deploy time |
| Q-03 | Sample and benchmark dataset licenses (D-05) | Before bundling samples (P16) and P19 | Replace any non-redistributable dataset with an OpenML equivalent |

Q-04 … Q-16 were answered by the owner on 2026-10-19 (DEC-007 … DEC-019, ADR-0003).

---

## 13. Known issues and tech debt  *(append-only; mark resolved, don't delete)*

| ID | Date | Prompt | Issue | Severity | Status |
| --- | --- | --- | --- | --- | --- |
| KI-001 | 2026-10-19 | P00 follow-up | A-004 doubles the default time budget (hosted 960 s), so a default hosted run can exceed the SM-2 "≤ 10 min" target. SM-2 / T-PERF-01 / G-14 are measured with `time_budget_s=480` until the owner retunes budgets | minor | open (owner retune after MVP) |

---

## 14. Unverified items  *(append-only; mark verified when done)*

Things that could not be verified in a session (no Docker, no network, no provider key, etc.). CI or the owner must verify them.

| ID | Date | Prompt | Item | Why unverified | How to verify | Status |
| --- | --- | --- | --- | --- | --- | --- |
| UV-001 | 2026-10-19 | P01 | `.github/workflows/ci.yml` runs on GitHub Actions | Actions cannot run inside this session; the same `make setup` / `make check` / `make requirements-check` commands passed locally | Check the first Actions run on `main` is green | verified 2026-10-19: run 39243450886 green (39 s); note: `actions/checkout@v4` and `setup-uv@v6` trigger a Node 20 deprecation annotation (bump in P18) |

---

## 15. Environment notes  *(replace)*

| Item | Value |
| --- | --- |
| Setup | `uv sync --all-extras` |
| Full check | `make check` |
| Tests only | `uv run pytest -m "not perf and not live and not slow"` |
| Network in AI sessions | 2026-10-19 session: PyPI reachable (`uv sync --all-extras` worked); provider calls not attempted |
| Docker in AI sessions | Not checked yet (P18) |
| Python in AI sessions | `/usr/bin/python3.11` (3.11.17) present; default `python3` is 3.13, so always use `uv run` |
| Worker processes | spawn context works in the session container (DEC-022) |

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
| 2026-10-19 | P00 follow-up | Owner answered Q-04…Q-16: DEC-007…DEC-019, amendments A-002…A-010, ADR-0003, KI-001; PI statuses updated. Work mode: each prompt is committed and pushed directly to `main`, then the next starts (stop at gates, failures, or new questions) |
| 2026-10-19 | P01 | Scaffold: `pyproject.toml` (deps, extras, ruff/mypy/pytest/coverage/import-linter config), `uv.lock`, `requirements.txt` + `scripts/export_requirements.sh`, `Makefile`, `config/{profiles,models,providers}.yaml`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `.gitignore`, `.env.example`, MIT `LICENSE`, package tree with docstrings, `streamlit_app.py` + UI shell, smoke + UI tests; DEC-020…022; PI-40; UV-001 |

---

## 18. Handoff note  *(replace at the end of every session)*

**Latest (P01, 2026-10-19):** P01 is complete and pushed to `main`. `make check` is green (8 tests).

- **Next:** P02 (data contracts, configuration, logging). It creates `tabpilot.config` and `tabpilot.logging`; the optional layers in contract R1 then become real.
- **P02 must apply:**
  - A-008: add W-TARGET-MISSING and give E-TARGET-TOO-FEW-ROWS its single meaning.
  - PI-09: E-TARGET-\* message templates.
  - PI-10: the full list of 15 W-codes.
  - PI-25: Pydantic `model_` namespace, `response_schema` serialization, `RunWarning` name.
  - PI-31: event-type enum.
  - DEC-021: config values live in `config/*.yaml`; placeholders are quoted strings (PI-40).
  - Time values come from A-004 / ADR-0003.
- **Environment:** Python 3.11 at `/usr/bin/python3.11`; always run tools through `uv run --frozen` or `make`. Never let ruff touch Markdown (excluded in `pyproject.toml`).
- **CI:** the first GitHub Actions run on `main` is green (UV-001 verified). Owner decisions are all recorded (ADR-0003).
- **Work mode:** commit and push each prompt to `main`, then continue (stop at gates, failures, or new questions).
