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
| Last completed prompt | — (none yet) |
| Next prompt | **P00** Project charter and context bootstrap |
| Current milestone | Milestone 0 — Foundations |
| Test suite | Not yet created |
| Last green commit | — |
| Blockers | None |

---

## 4. Prompt ledger  *(replace status cells; keep rows)*

Status values: `pending` · `in_progress` · `complete` · `complete_with_issues` · `blocked`

| ID | Title | PRD sections | Milestone / gate | Status | Commit | Date |
| --- | --- | --- | --- | --- | --- | --- |
| P00 | Project charter and context bootstrap | All (§1–§40) | M0 | pending | | |
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
| *(P00 adds entries)* | | | | |

---

## 12. Open questions for the owner  *(replace)*

| ID | Question | Blocking? | Default if unanswered |
| --- | --- | --- | --- |
| Q-01 | Verify Streamlit Community Cloud resources (D-01) with a probe app | Before P18 perf calibration | Keep 1 GB design assumption |
| Q-02 | Provider order and model IDs for hosted (D-03) | Before deployment (P20) | Groq first; IDs from provider pages at deploy time |
| Q-03 | Sample and benchmark dataset licenses (D-05) | Before bundling samples (P16) and P19 | Replace any non-redistributable dataset with an OpenML equivalent |

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

---

## 18. Handoff note  *(replace at the end of every session)*

**Latest:** Project not started. Begin with **P00** in `docs/PROMPTS.md`. Read `CLAUDE.md` first. The PRD is complete and approved; DEC-003/A-001 (runtime package) and DEC-004/005 (build order) are the only changes so far.
