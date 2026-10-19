# ADR-0001: Baseline decisions — source reconciliation (C-01…C-17) and open-decision defaults (D-01…D-16)

| Field | Value |
| --- | --- |
| Status | accepted |
| Date | 2026-10-19 |
| Prompt | P00 |
| Decision Log | DEC-001, DEC-002 |
| PRD refs | §2.8, §37 |
| Amendment | none |

## Context

The PRD reconciles conflicts in the concept PRD in §2.8 (C-01…C-17). It also lists open decisions in §37 (D-01…D-16), each with a recommended default. The project kit already records both in force (DEC-001, DEC-002). This ADR summarizes them in one place so later prompts can cite one record. The PRD remains the full text. This ADR does not change or extend it.

## Decision

### Source reconciliation (§2.8) — in force as written (DEC-001)

| ID | Resolution in force |
| --- | --- |
| C-01 | Two execution profiles: `hosted` 20 MB / 100K rows / 200 cols, `local` 50 MB / 500K rows / 500 cols, both gated by the memory estimator (§29.1, §29.4). Raising hosted limits needs evidence (NFR-PERF-04). |
| C-02 | No generated code in P0. LLM features are JSON Feature-DSL specs run by first-party code (§14.3). Arbitrary Python is P2, local-only, in a hardened container (§14.7). |
| C-03 | Sampling only within training folds for tuning trials (§17.6) and for bounded profiling statistics (§9.5). Final refit uses all DEV rows. |
| C-04 | Default zoo: linear, random forest, LightGBM. XGBoost and CatBoost are optional: on by default for `local`, off for `hosted` (§16). |
| C-05 | `reflection.max_rounds` default 1, hard cap 2 (§19). |
| C-06 | The Kaggle top-50% target is report-only (SM-6), not a release gate. |
| C-07 | Reproducibility levels R1 (required), R2 and R3 (targets) (§22.6). |
| C-08 | The backup host is the Docker image (§34.8). HF Spaces is open (D-07). |
| C-09 | Upload-limit conflict resolved by the C-01 profiles. |
| C-10 | LLM call budget default 12, configurable up to 25 (§23.7). |
| C-11 | Hosted runs write JSON run records (§30). MLflow is for benchmark and local runs only. |
| C-12 | Per-provider privacy notice and per-session consent. The compact profile excludes raw values (§24.4–§24.5). |
| C-13 | Datetimes become calendar features, with a temporal-structure warning. Ordered validation is P1 (§11.3). |
| C-14 | SM-2 is measured on REF-HOSTED, ≤ 30 columns, excluding cold start, default budget, p90 of 10 runs. |
| C-15 | Deletion at run end, plus idle TTL, plus startup/periodic sweeper (§24.7). |
| C-16 | Sealed holdout for datasets with ≥ 2,000 rows, evaluated once after selection (§11). See PI-06 on the row-count basis. |
| C-17 | Paired-fold acceptance: relative gain ≥ 0.2% **and** better in a majority of folds (§19.5). |

### Open decisions (§37) — recommended defaults in force until the owner decides (DEC-002)

| ID | Topic | Default in force | Needed before | Tracked as |
| --- | --- | --- | --- | --- |
| D-01 | Streamlit Community Cloud resources | Keep the 1 GB design assumption; verify with a probe app | P18 calibration / P20 deploy | Q-01 |
| D-02 | Worker as child process | `multiprocessing` spawn process; thread-mode fallback documented | P15 (spike first; see Q-15) | Q-15 |
| D-03 | Hosted provider order and model IDs | Groq first; model IDs chosen from provider pages at deploy time | P20 deploy | Q-02 |
| D-04 | Include Gemini free tier | Include with explicit per-provider consent | P10 / P16 | — |
| D-05 | Sample and benchmark dataset licenses | Replace any non-redistributable dataset with an OpenML equivalent | P16, P19 | Q-03 |
| D-06 | Holdout threshold and fraction | 2,000 rows / 20% | P05 | PI-06 / Q-05 |
| D-07 | Backup host | Docker image ready; choose a host when needed | When needed | — |
| D-08 | PDF report | HTML + Markdown only | — | — |
| D-09 | P2 arbitrary-Python sandbox | Not built in v1 | — | PI-35 |
| D-10 | Headerless CSV | Assume a header | — | — |
| D-11 | Gemini transport | OpenAI-compatible endpoint if the JSON-mode adapter test passes; else the `google-genai` SDK, isolated in one adapter file | P10 | — |
| D-12 | Refit-on-all-data export | DEV-refit only | — | — |
| D-13 | Repository license | MIT | P01 | Q-16 |
| D-14 | Precomputed sample results when the slot is busy | Yes, labeled "precomputed on {date}" | P16 | — |
| D-15 | Leakage thresholds | §11.7 defaults | P06 | PI-07 / Q-04 |
| D-16 | Category-label sharing / name anonymization | Labels never sent; names sent | — | — |

## Alternatives

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Block implementation until every D-xx is decided | No rework risk | Stalls all prompts on questions that don't affect early milestones | The PRD gives safe defaults, and §37 marks most items deferrable |
| Re-derive C-xx resolutions per prompt | — | Inconsistent interpretations across sessions | One record is cheaper and auditable |

## Consequences

- Invariants: none changed. C-02, C-03, C-12 and C-16 are the source of invariants I-1, I-2, I-6 and I-7.
- Requirements and tests: none changed.
- Changing any default above needs a Decision Log entry with status `accepted` by the owner. If the change affects behavior, it also needs a PRD amendment (U4).
- P00 found PRD issues that touch D-06, D-13 and D-15 (PI-06, PI-34, PI-07). They are in `CONTEXT.md` §11 and §12.

## Status

Accepted (project kit, DEC-001 and DEC-002). Summarized by P00.
