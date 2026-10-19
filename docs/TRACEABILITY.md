# TabPilot — Traceability Matrix

> Maps every requirement, test, non-functional requirement, and release gate in `docs/PRD.md` to the prompt in `docs/PROMPTS.md` that owns it.
> Created in P00 (2026-10-19). P20 adds implementation status and evidence columns.

## Conventions

- **Owner**: the one prompt responsible for making the item true and tested. Every ID has exactly one owner.
- **Contributors**: prompts that build part of it (a helper, fixture, UI surface, or later re-verification). They do not own it.
- **Milestone**: the milestone gate by which the item must be green: **A** (after P09), **B** (after P15), **C** (after P16), **D** (after P20).
- **Release gate**: the `G-*` gate in PRD §38 whose evidence uses the item. "—" means the item doesn't feed a release gate directly.
- Test meanings follow the §31.3 catalog. Where §31.3 lists several topics for one ID range, they map to the IDs in order. Exceptions and ranges without per-ID topics are recorded in `CONTEXT.md` §11 (PI-19, PI-20).
- PRD issues that affect an item are cited as `PI-nn` (see `CONTEXT.md` §11).

Coverage (checked by a scratch script that extracts every ID from the PRD, with ranges expanded): FR 36/36, T 136/136, NFR 27/27, G 18/18.

---

## 1. Functional requirements (PRD §4)

| FR | Title | PRD sections | Owner | Contributors | Test IDs | Milestone | Release gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FR-ING-01 | File upload and format acceptance | §4.1, §8.1, §24.2 | P04 | P16 (upload page) | T-ING-01, T-ING-02, T-SEC-04 | A | G-02 |
| FR-ING-02 | Parsing with encoding and delimiter detection | §8.2, §8.3 | P04 | P03 (parsing fixtures) | T-ING-03, T-ING-04, T-ING-05, T-ING-06, T-ING-07 | A | G-02 |
| FR-ING-03 | Column-name normalization | §8.4 | P04 | P14 (`pipeline.py` applies map) | T-ING-08, T-ART-04 | A | G-02 |
| FR-ING-04 | Type inference and column roles | §8.5, §8.6 | P04 | P16 (override controls) | T-ING-09, T-ING-10, T-ING-11, T-ING-12, T-ING-13 | A | G-02 |
| FR-ING-05 | Resource feasibility check | §29.4, §29.5 | P04 | P18 (calibration) | T-RES-01, T-RES-02 | A | G-02 |
| FR-ING-06 | Sample datasets | §4.1, §37 D-05, D-14 | P16 | P20 (live smoke run) | T-E2E-01, T-E2E-02, T-E2E-03 (fixtures only; see PI-38) | C | G-01, G-14 |
| FR-TGT-01 | Target selection and validation | §8.7 | P04 | — | T-TGT-01, T-TGT-02, T-TGT-03, T-TGT-04, T-TGT-05, T-TGT-06 | A | G-02 |
| FR-TGT-02 | Task detection and confirmation | §10.1 | P05 | P16 (confirmation UI, T-UI-02) | T-TGT-07, T-TGT-08, T-TGT-09, T-TGT-10 | A | G-01 |
| FR-TGT-03 | Metric selection | §10.2–§10.4 | P05 | P11 (plan cannot change metric) | T-TGT-11 | A | G-06 |
| FR-SPL-01 | Holdout sealing and fold construction | §11 | P05 | P09 (run-level guard use) | T-LEAK-01, T-LEAK-02, T-LEAK-03, T-LEAK-04 | A | G-06 |
| FR-PRF-01 | Profiling engine | §9 | P06 | — | T-PRF-01, T-PRF-02, T-PRF-03, T-PRF-04 | A | — |
| FR-PRF-02 | Leakage and structure warnings | §9.3, §11.7 | P06 | P03 (`leaky_binary.csv`) | T-LEAK-05, T-LEAK-06 | A | — |
| FR-PRF-03 | Compact profile and planner-input inspection | §9.4 | P06 | P11 (`preview_planner_input`), P16 (inspector) | T-PRIV-01, T-PRIV-02 | B | G-04 |
| FR-PLN-01 | LLM planning | §12 | P11 | P10 (router) | T-PLN-01, T-PLN-02 | B | G-03 |
| FR-PLN-02 | Plan validation and repair | §12.6 | P11 | P03 (plan fixtures), P07 (DSL validator) | T-PLN-03, T-PLN-04, T-PLN-05, T-PLN-06, T-PLN-07, T-PLN-08 | B | G-03 |
| FR-PLN-03 | Rule-based fallback plan | §12.8, §12.9 | P09 (DEC-004) | P11 (fallback triggers), P15 (end-to-end) | T-PLN-09, T-LLM-05 | A | G-03, G-12 |
| FR-FE-01 | Deterministic baseline transformations | §14.2, §15 | P07 | P08 (pipeline step) | T-FE-01 | A | G-07 |
| FR-FE-02 | LLM-proposed DSL features | §14.3–§14.6 | P07 | P08 (fold restart, T-FE-08), P11 | T-FE-02, T-FE-03, T-FE-04, T-FE-05, T-FE-06, T-FE-07, T-FE-08, T-LEAK-07, T-SEC-01 | A | G-05, G-07 |
| FR-FE-03 | Arbitrary-Python features (P2, deferred: D-09) | §14.7 | P17 (verifies it is absent from every build) | P01 (contract rule 4) | T-SEC-08 | D | G-07 |
| FR-PRE-01 | Preprocessing pipeline | §15 | P08 | P07 (runtime transformers) | T-PRE-01, T-PRE-02, T-PRE-03, T-PRE-04, T-PRE-05 | A | G-05 |
| FR-TRN-01 | Model zoo training with CV and OOF | §16, §11.4 | P08 | — | T-TRN-01, T-TRN-02, T-TRN-03, T-TRN-04, T-TRN-05 | A | G-01 |
| FR-TUN-01 | Hyperparameter optimization | §17 | P09 | — | T-TUN-01, T-TUN-02, T-TUN-03, T-TUN-04 | A | G-08 |
| FR-EVL-01 | Leaderboard and selection | §18 | P09 | P16 (display) | T-EVL-01, T-EVL-02, T-EVL-03, T-EVL-04, T-EVL-05 | A | G-06, G-09 |
| FR-EVL-02 | One-time holdout evaluation | §11.2, §18.3, §18.5 | P09 | P13 (report section) | T-LEAK-03, T-EVL-06 | A | G-06, G-13 |
| FR-RFL-01 | Bounded reflection | §19 | P12 | P15 (graph loop, T-RFL-02) | T-RFL-01, T-RFL-02, T-RFL-03, T-RFL-04, T-RFL-05 | B | G-08 |
| FR-ENS-01 | Validated ensembling | §20 | P12 | P14 (exported wrapper) | T-ENS-01, T-ENS-02, T-ENS-03, T-ENS-04 | B | — |
| FR-EXP-01 | Explainability | §21.3–§21.5 | P13 | — | T-EXP-01, T-EXP-02, T-EXP-03 | B | — |
| FR-RPT-01 | Report generation | §21 | P13 | P10 (router) | T-RPT-01, T-RPT-02, T-RPT-03, T-RPT-04 | B | G-13 |
| FR-ART-01 | Artifact bundle export | §22 | P14 | — | T-ART-01, T-ART-02, T-ART-03, T-ART-04, T-ART-05, T-ART-06 | B | G-10, G-11 |
| FR-ART-02 | Standalone `pipeline.py` | §22.4, A-001 | P14 | — | T-ART-03, T-ART-04 | B | G-10 |
| FR-LLM-01 | Provider configuration and routing | §23 | P10 | P16 (provider settings, consent) | T-LLM-01, T-LLM-02, T-LLM-03, T-LLM-04, T-LLM-05, T-LLM-06 | B | G-12 |
| FR-LLM-02 | LLM budget accounting | §23.7 | P10 | — | T-LLM-07 | B | G-08 |
| FR-RUN-01 | Run lifecycle: events, cancellation, timeout | §13.7, §28, §27.4 | P15 | P16 (run page) | T-RUN-01, T-RUN-02, T-RUN-03, T-RUN-04 | B | G-08 |
| FR-RUN-02 | Cleanup | §24.7 | P15 | P16 (Delete now), P17 (audit) | T-SEC-06, T-SEC-07 | B | G-15 |
| FR-RUN-03 | Concurrency guard on hosted | §4.8, §28 | P15 | P16 (waiting state) | T-RUN-05 | B | — |
| FR-BEN-01 | Benchmark harness | §32 | P19 | — | T-BEN-01, T-BEN-02 | D | G-16 |

---

## 2. Tests (PRD §31.3, ranges expanded)

| Test ID | Verifies (per §31.3) | Requirement | Owner | Contributors | Milestone | Release gate |
| --- | --- | --- | --- | --- | --- | --- |
| T-ING-01 | Size limit checked before parse (25 MB on hosted rejected; parser spy not called) | FR-ING-01 | P04 | — | A | G-02 |
| T-ING-02 | Extension/magic mismatch: ZIP-as-CSV → E-ARCHIVE-UNSUPPORTED | FR-ING-01 | P04 | P03 (fixture) | A | G-02 |
| T-ING-03 | Encoding detection | FR-ING-02 | P04 | P03 (fixtures) | A | G-02 |
| T-ING-04 | Delimiter detection | FR-ING-02 | P04 | P03 (fixtures) | A | G-02 |
| T-ING-05 | Decimal comma | FR-ING-02 | P04 | P03 (fixtures) | A | G-02 |
| T-ING-06 | Malformed lines (bounded skip / E-PARSE-MALFORMED) | FR-ING-02 | P04 | P03 (fixtures) | A | G-02 |
| T-ING-07 | Quoted embedded newlines | FR-ING-02 | P04 | P03 (fixtures) | A | G-02 |
| T-ING-08 | Name normalization + dedupe; idempotent | FR-ING-03 | P04 | — | A | G-02 |
| T-ING-09 | Type-inference rules on `quality_issues.csv` + coercion counts (per-ID split defined by P04, PI-20) | FR-ING-04 | P04 | P03 (fixture) | A | G-02 |
| T-ING-10 | Type-inference rules on `quality_issues.csv` + coercion counts (per-ID split defined by P04, PI-20) | FR-ING-04 | P04 | P03 (fixture) | A | G-02 |
| T-ING-11 | Type-inference rules on `quality_issues.csv` + coercion counts (per-ID split defined by P04, PI-20) | FR-ING-04 | P04 | P03 (fixture) | A | G-02 |
| T-ING-12 | Type-inference rules on `quality_issues.csv` + coercion counts (per-ID split defined by P04, PI-20) | FR-ING-04 | P04 | P03 (fixture) | A | G-02 |
| T-ING-13 | Type-inference rules on `quality_issues.csv` + coercion counts (per-ID split defined by P04, PI-20) | FR-ING-04 | P04 | P03 (fixture) | A | G-02 |
| T-TGT-01 | E-TARGET-CONSTANT | FR-TGT-01 | P04 | P03 (fixtures) | A | G-02 |
| T-TGT-02 | E-TARGET-ALL-MISSING | FR-TGT-01 | P04 | P03 (fixtures) | A | G-02 |
| T-TGT-03 | E-TARGET-TOO-FEW-ROWS | FR-TGT-01 | P04 | P03 (fixtures) | A | G-02 |
| T-TGT-04 | E-TARGET-CLASS-TOO-SMALL | FR-TGT-01 | P04 | P03 (fixtures) | A | G-02 |
| T-TGT-05 | E-TARGET-TOO-MANY-CLASSES | FR-TGT-01 | P04 | P03 (fixtures) | A | G-02 |
| T-TGT-06 | E-TARGET-UNSUPPORTED-TYPE | FR-TGT-01 | P04 | P03 (fixtures) | A | G-02 |
| T-TGT-07 | Task detection: expected task/confidence table (per-ID split defined by P05, PI-20) | FR-TGT-02 | P05 | P16 (UI confirm) | A | G-01 |
| T-TGT-08 | Task detection: expected task/confidence table (per-ID split defined by P05, PI-20) | FR-TGT-02 | P05 | P16 (UI confirm) | A | G-01 |
| T-TGT-09 | Task detection: expected task/confidence table (per-ID split defined by P05, PI-20) | FR-TGT-02 | P05 | P16 (UI confirm) | A | G-01 |
| T-TGT-10 | Task detection: expected task/confidence table (per-ID split defined by P05, PI-20) | FR-TGT-02 | P05 | P16 (UI confirm) | A | G-01 |
| T-TGT-11 | Metric fixed for run: ExperimentConfig metric unchanged after a plan recommends another | FR-TGT-03 | P11 | P05 (metric registry) | B | G-06 |
| T-LEAK-01 | Holdout sealed: `HoldoutGuard.open` before freeze raises; access log empty until finalize | FR-SPL-01 | P05 | — | A | G-06 |
| T-LEAK-02 | Tuning never reads HOLD (GuardedFrame raises) | FR-SPL-01, FR-TUN-01 | P09 | P03 (GuardedFrame) | A | G-06 |
| T-LEAK-03 | Holdout opened once per run (access log length 1) | FR-SPL-01, FR-EVL-02 | P09 | P05 (guard level) | A | G-06 |
| T-LEAK-04 | Group integrity: no group in two folds or in DEV∩HOLD | FR-SPL-01 | P05 | P03 (fixture) | A | G-06 |
| T-LEAK-05 | Near-deterministic column excluded with notice (`leaky_binary.csv`; see PI-07) | FR-PRF-02 | P06 | P03 (fixture) | A | — |
| T-LEAK-06 | Strong-leak column kept and warned | FR-PRF-02 | P06 | P03 (fixture) | A | — |
| T-LEAK-07 | Fold-local fitting: spies record fit indices ⊆ training fold for every stateful step | FR-FE-02, FR-PRE-01 | P08 | P07 (DSL ops), P03 (FitSpy) | A | G-05 |
| T-LEAK-08 | Canary: pure-noise data + high-card target encoding, CV AUC ≤ 0.55 (5 seeds; see PI-08) | FR-PRE-01 | P08 | — | A | G-05 |
| T-PRF-01 | Profile stats, time caps, truncation (per-ID split defined by P06, PI-20) | FR-PRF-01 | P06 | — | A | — |
| T-PRF-02 | Profile stats, time caps, truncation (per-ID split defined by P06, PI-20) | FR-PRF-01 | P06 | — | A | — |
| T-PRF-03 | Profile stats, time caps, truncation (per-ID split defined by P06, PI-20) | FR-PRF-01 | P06 | — | A | — |
| T-PRF-04 | Profile stats, time caps, truncation (per-ID split defined by P06, PI-20) | FR-PRF-01 | P06 | — | A | — |
| T-PRIV-01 | No raw values in compact profile | FR-PRF-03, NFR-PRIV-01 | P06 | P12, P13 (same assertion on reflection/narrative payloads, PI-16) | A | G-04 |
| T-PRIV-02 | Inspector JSON bytes == captured request data | FR-PRF-03, NFR-PRIV-01 | P11 | P16 (UI inspector) | B | G-04 |
| T-PRIV-03 | No-LLM mode makes zero provider requests | FR-PLN-03, NFR-PRIV-01 | P15 | — | B | G-04 |
| T-PRIV-04 | Logs contain no fixture cell values or key patterns (e2e scan) | NFR-PRIV-01, NFR-SEC-03 | P15 | P02 (scrubber), P10 (router logs) | B | — |
| T-PLN-01 | Planner happy path: recorded responses validate | FR-PLN-01 | P11 | P03 (llm_responses) | B | G-03 |
| T-PLN-02 | Planner happy path: recorded responses validate | FR-PLN-01 | P11 | P03 (llm_responses) | B | G-03 |
| T-PLN-03 | Invalid/unsafe/contradictory plans → repaired or fallback (per-ID split = plan fixtures in MANIFEST.yaml, PI-20) | FR-PLN-02 | P11 | P03 (plan fixtures) | B | G-03 |
| T-PLN-04 | Invalid/unsafe/contradictory plans → repaired or fallback (per-ID split = plan fixtures in MANIFEST.yaml, PI-20) | FR-PLN-02 | P11 | P03 (plan fixtures) | B | G-03 |
| T-PLN-05 | Invalid/unsafe/contradictory plans → repaired or fallback (per-ID split = plan fixtures in MANIFEST.yaml, PI-20) | FR-PLN-02 | P11 | P03 (plan fixtures) | B | G-03 |
| T-PLN-06 | Invalid/unsafe/contradictory plans → repaired or fallback (per-ID split = plan fixtures in MANIFEST.yaml, PI-20) | FR-PLN-02 | P11 | P03 (plan fixtures) | B | G-03 |
| T-PLN-07 | Invalid/unsafe/contradictory plans → repaired or fallback (per-ID split = plan fixtures in MANIFEST.yaml, PI-20) | FR-PLN-02 | P11 | P03 (plan fixtures) | B | G-03 |
| T-PLN-08 | Invalid/unsafe/contradictory plans → repaired or fallback (per-ID split = plan fixtures in MANIFEST.yaml, PI-20) | FR-PLN-02 | P11 | P03 (plan fixtures) | B | G-03 |
| T-PLN-09 | Fallback plan determinism: same profile → identical plan JSON | FR-PLN-03 | P09 | — | A | G-03 |
| T-FE-01 | Baseline features: expected columns and values | FR-FE-01 | P07 | — | A | G-07 |
| T-FE-02 | DSL ops | FR-FE-02 | P07 | — | A | G-07 |
| T-FE-03 | Static validation | FR-FE-02 | P07 | — | A | G-07 |
| T-FE-04 | NaN semantics | FR-FE-02 | P07 | — | A | G-07 |
| T-FE-05 | Degenerate features | FR-FE-02 | P07 | — | A | G-07 |
| T-FE-06 | Dependency skipping | FR-FE-02 | P07 | — | A | G-07 |
| T-FE-07 | Per-feature timeout | FR-FE-02 | P07 | — | A | G-07 |
| T-FE-08 | Feature failure in any fold → removed, candidate re-run from fold 1 | FR-FE-02 | P08 | P07 (failure reporting) | A | G-07 |
| T-PRE-01 | Pipeline construction | FR-PRE-01 | P08 | — | A | — |
| T-PRE-02 | Unknown categories at inference | FR-PRE-01 | P08 | — | A | — |
| T-PRE-03 | Batch vs row-by-row consistency (FR-PRE-01 AC; PI-19) | FR-PRE-01 | P08 | — | A | — |
| T-PRE-04 | Feature ordering | FR-PRE-01 | P08 | — | A | — |
| T-PRE-05 | Missing input columns → PredictionSchemaError | FR-PRE-01 | P08 | — | A | — |
| T-TRN-01 | Each model × task trains | FR-TRN-01 | P08 | — | A | — |
| T-TRN-02 | Fallback params on fold-1 failure | FR-TRN-01 | P08 | — | A | — |
| T-TRN-03 | Failure isolation | FR-TRN-01, NFR-REL-03 | P08 | — | A | — |
| T-TRN-04 | Early stopping uses inner split | FR-TRN-01 | P08 | — | A | — |
| T-TRN-05 | RF skipped on resource estimate | FR-TRN-01 | P08 | — | A | — |
| T-TUN-01 | Budget enforcement: elapsed ≤ allocation × 1.1 + 15 s | FR-TUN-01 | P09 | — | A | G-08 |
| T-TUN-02 | Trial-mode determinism | FR-TUN-01 | P09 | — | A | — |
| T-TUN-03 | Failed trials recorded; study continues | FR-TUN-01 | P09 | — | A | — |
| T-TUN-04 | Within-fold subsampling leaves validation folds unchanged | FR-TUN-01 | P09 | — | A | — |
| T-EVL-01 | Leaderboard mean/std equal recomputation from stored fold scores | FR-EVL-01, FR-TRN-01 | P09 | — | A | G-09 |
| T-EVL-02 | Selection rule | FR-EVL-01 | P09 | — | A | G-06 |
| T-EVL-03 | Tie-breaks | FR-EVL-01 | P09 | — | A | G-06 |
| T-EVL-04 | Failed-entry exclusion | FR-EVL-01 | P09 | — | A | G-06 |
| T-EVL-05 | Comparability enforcement | FR-EVL-01 | P09 | — | A | G-06 |
| T-EVL-06 | Holdout reported separately; CV table has no holdout numbers | FR-EVL-02 | P13 | P09 (holdout record) | B | G-13 |
| T-RFL-01 | Paired-fold acceptance rule on synthetic deltas | FR-RFL-01 | P12 | — | B | — |
| T-RFL-02 | Termination: adversarial reflector → ≤ max_rounds (graph level) | FR-RFL-01 | P15 | P12 (round budget) | B | G-08 |
| T-RFL-03 | Prohibited changes rejected | FR-RFL-01 | P12 | — | B | — |
| T-RFL-04 | Budget skip | FR-RFL-01 | P12 | — | B | — |
| T-RFL-05 | Incumbent preserved on regression | FR-RFL-01 | P12 | — | B | — |
| T-ENS-01 | Cross-fitted scoring | FR-ENS-01 | P12 | P14 (bundle with ensemble) | B | — |
| T-ENS-02 | Keep/discard rule | FR-ENS-01 | P12 | P14 (bundle with ensemble) | B | — |
| T-ENS-03 | Determinism | FR-ENS-01 | P12 | P14 (bundle with ensemble) | B | — |
| T-ENS-04 | Export wrapper parity | FR-ENS-01 | P12 | P14 (bundle with ensemble) | B | — |
| T-EXP-01 | SHAP per model type | FR-EXP-01 | P13 | — | B | — |
| T-EXP-02 | Fallback to permutation importance | FR-EXP-01, NFR-EXP-01 | P13 | — | B | — |
| T-EXP-03 | Timeout path | FR-EXP-01 | P13 | — | B | — |
| T-RPT-01 | All report sections present | FR-RPT-01 | P13 | — | B | G-13 |
| T-RPT-02 | Fact-check catches invented number | FR-RPT-01, NFR-EXP-02 | P13 | — | B | — |
| T-RPT-03 | Causal phrase rejected | FR-RPT-01 | P13 | — | B | — |
| T-RPT-04 | Template fallback | FR-RPT-01 | P13 | — | B | — |
| T-ART-01 | Bundle structure + manifest SHA-256 | FR-ART-01 | P14 | — | B | G-10 |
| T-ART-02 | R1 parity: reload + predict == expected | FR-ART-01, NFR-REP-01 | P14 | — | B | G-11 |
| T-ART-03 | `pipeline.py` lint + `verify` in clean venv | FR-ART-02 | P14 | P18 (CI job) | B | G-10 |
| T-ART-04 | Original column names accepted at predict | FR-ART-02, FR-ING-03 | P14 | — | B | — |
| T-ART-05 | R2 retrain parity ≤ 1e-6 on fixtures | FR-ART-02, NFR-REP-02 | P14 | — | B | — |
| T-ART-06 | No raw rows in bundle (row-hash scan) | FR-ART-01 | P14 | — | B | — |
| T-LLM-01 | Adapters parse recorded responses | FR-LLM-01 | P10 | P03 (llm_responses) | B | — |
| T-LLM-02 | 429 → wait or next provider | FR-LLM-01 | P10 | — | B | — |
| T-LLM-03 | 5xx → backoff | FR-LLM-01 | P10 | — | B | — |
| T-LLM-04 | 401 → skip provider | FR-LLM-01 | P10 | — | B | — |
| T-LLM-05 | All providers down → rule-based run completes | FR-LLM-01, FR-PLN-03, NFR-REL-02 | P15 | P11 (planner fallback) | B | G-03, G-12 |
| T-LLM-06 | Fallback does not alter request (same payload hash) | FR-LLM-01 | P10 | — | B | — |
| T-LLM-07 | Calls/tokens never exceed budget ceilings | FR-LLM-02 | P10 | — | B | G-08 |
| T-RUN-01 | Each system event type matches a state change | FR-RUN-01 | P15 | — | B | — |
| T-RUN-02 | Cancel ≤ 15 s, no orphans | FR-RUN-01 | P15 | — | B | — |
| T-RUN-03 | Global timeout → partial result | FR-RUN-01 | P15 | — | B | G-08 |
| T-RUN-04 | Worker crash → E-WORKER-LOST + cleanup | FR-RUN-01 | P15 | — | B | — |
| T-RUN-05 | Slot lock: second concurrent start → SlotBusy | FR-RUN-03 | P15 | — | B | — |
| T-SEC-01 | AST scan: no eval/exec/compile/pd.eval/query/eval on non-literals | FR-FE-02, NFR-SEC-01 | P07 | P17 (re-verify) | A | G-07 |
| T-SEC-02 | Injection column names escaped in HTML/MD; plan schema still enforced | §24.8 | P13 | P11 (plan part), P17 | B | — |
| T-SEC-03 | Formula injection: leaderboard CSV cells prefixed | §24.9 | P14 | — | B | — |
| T-SEC-04 | Malicious filenames: no path traversal; display sanitized | FR-ING-01 | P04 | P16 (upload), P17 | A | — |
| T-SEC-05 | Secrets absent from logs, bundle, run dir | NFR-SEC-03 | P17 | P10, P14, P15 | D | G-18 |
| T-SEC-06 | Cleanup on success/failure/cancel per §24.7 | FR-RUN-02, NFR-PRIV-02 | P15 | P16 (Delete now), P17 | B | G-15 |
| T-SEC-07 | Sweeper removes old run dirs | FR-RUN-02, NFR-PRIV-02 | P15 | P17 | B | G-15 |
| T-SEC-08 | Hosted build excludes `tabpilot.sandbox` (import fails) | FR-FE-03 | P17 | P01 (contract rule 4) | D | G-07 |
| T-RES-01 | Memory estimator | FR-ING-05 | P04 | P18 (calibration) | A | — |
| T-RES-02 | Simplifications + E-RESOURCE-INFEASIBLE rejection | FR-ING-05 | P04 | P18 (calibration) | A | — |
| T-RES-03 | RSS guard aborts memory-hog candidate; run continues | §29.6 | P09 | — | A | G-08 |
| T-PERF-01 | SM-2: p90 ≤ 600 s on 10K × 30 in hosted container | NFR-PERF-01 | P18 | — | D | G-14 |
| T-PERF-02 | `hosted_limit.csv` completes or simplifies without OOM | NFR-PERF-02 | P18 | P03 (generator) | D | — |
| T-PERF-03 | Estimator predicted peak within ±30% of measured | FR-ING-05, §29.4 | P18 | — | D | — |
| T-UI-01 | Upload → results flow | FR-ING-01, FR-RUN-01 | P16 | — | C | — |
| T-UI-02 | Ambiguous task requires a choice | FR-TGT-02 | P16 | — | C | — |
| T-UI-03 | LLM rationale labeled | FR-RUN-01 | P16 | — | C | — |
| T-UI-04 | Error states render | NFR-UX-01 | P16 | — | C | G-02 |
| T-E2E-01 | Binary full run (stub LLM) → bundle + R1 | FR-ING-06 (via fixtures), FR-ART-01 | P15 | P16 (manual sample runs) | B | G-01 |
| T-E2E-02 | Multiclass full run → bundle + R1 | FR-ING-06 (via fixtures), FR-ART-01 | P15 | P16 (manual sample runs) | B | G-01 |
| T-E2E-03 | Regression full run → bundle + R1 | FR-ING-06 (via fixtures), FR-ART-01 | P15 | P16 (manual sample runs) | B | G-01 |
| T-E2E-04 | Full run with live provider (nightly, optional, `live`) | FR-LLM-01 | P15 | — | B | — |
| T-E2E-05 | Reliability suite: 50 runs, ≥ 95% valid bundles, 100% failures with E-codes | NFR-REL-01 | P15 | — | B | G-17 |
| T-BEN-01 | Harness parity: identical outer-train data and budget | FR-BEN-01 | P19 | — | D | — |
| T-BEN-02 | README table regenerated from `results.csv` | FR-BEN-01 | P19 | P20 | D | G-16 |

---

## 3. Non-functional requirements (PRD §33)

| NFR | Requirement | Class | Verifying prompt | Method / evidence |
| --- | --- | --- | --- | --- |
| NFR-REL-01 | Valid bundle across reliability suite (≥ 95%) | Mandatory | P15 | T-E2E-05 |
| NFR-REL-02 | Baseline path independent of LLM availability | Mandatory | P15 | T-LLM-05 (rule-based run completes) |
| NFR-REL-03 | Single-model failures never fail the run | Mandatory | P08 | T-TRN-03 |
| NFR-REP-01 | R1 prediction parity | Mandatory | P14 | T-ART-02 |
| NFR-REP-02 | R2 retraining parity on fixtures | Desirable | P14 | T-ART-05 |
| NFR-REP-03 | R3 CV parity ±0.5% in trial mode | Desirable | P19 | Benchmark determinism check (trial-count mode, same environment) |
| NFR-PERF-01 | SM-2 runtime | Mandatory | P18 | T-PERF-01 in `--memory=1g --cpus=2` container |
| NFR-PERF-02 | Peak worker RSS ≤ 750 MB on hosted-limit data | Mandatory | P18 | T-PERF-02 |
| NFR-PERF-03 | UI interactions respond ≤ 2 s on hosted-limit data | Desirable | P18 | UI timing in perf suite (P16 provides the interactions) |
| NFR-PERF-04 | Hosted limits raised only with evidence | Mandatory (process) | P18 | Perf report in `docs/perf/`; PR review rule |
| NFR-SEC-01 | No evaluation of LLM-produced strings | Mandatory | P07 | T-SEC-01 (re-verified in P17) |
| NFR-SEC-02 | No known high/critical vulnerable dependency with available fix | Mandatory | P17 | `pip-audit` CI job |
| NFR-SEC-03 | Secrets never persisted or logged | Mandatory | P17 | T-SEC-05, T-PRIV-04 |
| NFR-PRIV-01 | No raw rows to LLM by default | Mandatory | P17 | T-PRIV-01, T-PRIV-02, T-PRIV-03 (audit consolidates; owners P06, P11, P15) |
| NFR-PRIV-02 | Cleanup lifecycle | Mandatory | P15 | T-SEC-06, T-SEC-07 |
| NFR-MNT-01 | Type-checked (mypy strict on `schemas`, `evaluation`, `splitting`, `llm`) | Mandatory | P01 | mypy config in `make check` / CI; every later prompt keeps it green |
| NFR-MNT-02 | Coverage ≥ 80% overall; ≥ 95% `splitting`, `evaluation`, `features.dsl`, `planning.validator` | Mandatory | P01 | Coverage config + CI threshold; per-module thresholds checked by owning prompts (P05, P07, P09, P11) |
| NFR-MNT-03 | Import-linter contracts pass | Mandatory | P01 | `lint-imports` in `make check` |
| NFR-ACC-01 | Labels, keyboard access, non-color encodings, chart text alternatives | Desirable | P16 | UI review checklist + AppTest label assertions |
| NFR-POR-01 | Docker on x86-64 and Streamlit Cloud from the same commit | Mandatory | P18 | CI Docker build; deploy (P20 owner steps) |
| NFR-POR-02 | Runs on macOS arm64 locally | Desirable | P20 | Manual check per release (owner) |
| NFR-RES-01 | Image ≤ 1.5 GB; install ≤ 5 min | Desirable | P18 | Measured image size and install time |
| NFR-EXP-01 | Every run has importance output or an explicit reason | Mandatory | P13 | T-EXP-02 |
| NFR-EXP-02 | Narrative fact-check 100% or template | Mandatory | P13 | T-RPT-02 |
| NFR-COST-01 | No paid or card-required service in the core demo path | Mandatory | P20 | Deployment review in `docs/RELEASE_CHECKLIST.md` |
| NFR-UX-01 | Every failure shows an E-code message with an action | Mandatory | P16 | T-UI-04 |
| NFR-DEF-01 | > 1 concurrent run on hosted | Deferred | P20 (records deferral only) | Not built in MVP |

---

## 4. Release gates (PRD §38)

| Gate | Criterion | Verification (PRD) | Evidence-producing prompts | Milestone where first green |
| --- | --- | --- | --- | --- |
| G-01 | Binary, multiclass, regression workflows work | T-E2E-01..03; manual sample runs | P15 (T-E2E-01, T-E2E-02, T-E2E-03), P16 (sample runs), P20 (record) | B |
| G-02 | Invalid datasets rejected with actionable errors | T-ING-*, T-TGT-*, T-UI-04 | P04 (T-ING-01..13, T-TGT-01..06), P05 (T-TGT-07..11 inputs), P16 (T-UI-04) | C |
| G-03 | Planner returns validated plans or falls back safely | T-PLN-*, T-LLM-05 | P09 (T-PLN-09), P11 (T-PLN-01..08), P15 (T-LLM-05) | B |
| G-04 | Raw rows not sent to the LLM by default | T-PRIV-01..03 | P06 (T-PRIV-01), P11 (T-PRIV-02), P15 (T-PRIV-03), P12/P13 (payload checks, PI-16) | B |
| G-05 | Preprocessing fitted within training folds | T-LEAK-07, T-LEAK-08 | P07, P08 | A |
| G-06 | Holdout sealed and used once; comparable selection | T-LEAK-01..04, T-EVL-02..05 | P05 (T-LEAK-01, T-LEAK-04), P09 (T-LEAK-02, T-LEAK-03, T-EVL-02..05) | A |
| G-07 | Features execute only within the DSL boundary | T-SEC-01, T-SEC-08, T-FE-* | P07 (T-SEC-01, T-FE-01..07), P08 (T-FE-08), P17 (T-SEC-08) | D |
| G-08 | Time, memory, iteration budgets enforced | T-TUN-01, T-RFL-02, T-RES-03, T-RUN-03, T-LLM-07 | P09 (T-TUN-01, T-RES-03), P10 (T-LLM-07), P15 (T-RFL-02, T-RUN-03) | B |
| G-09 | Leaderboard matches stored experiment results | T-EVL-01 | P09 | A |
| G-10 | Best valid pipeline exports and reloads | T-ART-01, T-ART-03 | P14 (clean-venv job in CI: P18) | B |
| G-11 | Exported pipeline reproduces expected predictions | T-ART-02 | P14 | B |
| G-12 | Provider failures do not prevent a baseline run | T-LLM-05, US-F2 | P11, P15 | B |
| G-13 | Report separates validation and independent test | T-EVL-06, T-RPT-01 | P13 | B |
| G-14 | Demo finishes within budget on reference environment | T-PERF-01 + live smoke test | P18 (container measurement), P20 (owner's live smoke test via `docs/DEPLOY.md`) | D |
| G-15 | Temporary data deleted per lifecycle | T-SEC-06, T-SEC-07 | P15, P17 (audit) | B |
| G-16 | Benchmark claims backed by reproducible files | T-BEN-02; claims review | P19, P20 (claims review) | D |
| G-17 | Reliability | T-E2E-05 | P15 | B |
| G-18 | Security hygiene | pip-audit, gitleaks, T-SEC-05 | P17 (CI jobs, T-SEC-05), P18 (release workflow runs them) | D |

All gates are re-verified and recorded with evidence in `docs/RELEASE_CHECKLIST.md` by P20.

---

## Appendix A — E-code coverage (supplementary)

US-F1 and G-02 require every E-code to be triggered by a fixture. This table assigns each E-code in §8.9 to the prompt that raises it and the test that triggers it. Message templates live in `tabpilot.schemas.errors` (P02).

| E-code | Raised by (prompt) | Triggering test |
| --- | --- | --- |
| E-FILE-TOO-LARGE | P04 | T-ING-01 |
| E-FILE-TYPE | P04 | P04 test, no dedicated T-ID (PI-38) |
| E-FILE-EMPTY | P04 | P04 test on header-only and empty fixtures, no dedicated T-ID (PI-38) |
| E-FILE-CORRUPT | P04 | P04 test, no dedicated T-ID (PI-38) |
| E-ARCHIVE-UNSUPPORTED | P04 | T-ING-02 |
| E-ENCODING | P04 | T-ING-03 |
| E-DELIMITER-AMBIGUOUS | P04 | T-ING-04 |
| E-PARSE-MALFORMED | P04 | T-ING-06 |
| E-PARQUET-NESTED | P04 | P04 test on nested-Parquet malicious fixture, no dedicated T-ID (PI-38) |
| E-TOO-MANY-COLUMNS | P04 | P04 limit test, no dedicated T-ID (PI-38) |
| E-TOO-MANY-ROWS | P04 | P04 limit test, no dedicated T-ID (PI-38) |
| E-TARGET-CONSTANT, -ALL-MISSING, -TOO-FEW-ROWS, -CLASS-TOO-SMALL, -TOO-MANY-CLASSES, -UNSUPPORTED-TYPE | P04 | T-TGT-01..06 (templates missing in §8.9: PI-09) |
| E-RESOURCE-INFEASIBLE | P04 | T-RES-02 |
| E-NO-VALID-MODEL | P09 (selection), P15 (run) | T-EVL-04 (no eligible entry), T-RUN-03 (timeout with no valid candidate) |
| E-EXPORT-PARITY | P14 | P14 parity-failure test (injected mismatch → rebuild once → E-code), no dedicated T-ID (PI-38) |
| E-WORKER-LOST | P15 | T-RUN-04 |

UI rendering of every code: P16 (T-UI-04).

## Appendix B — Warning codes found in the PRD (for the P02 `WarningCode` enum)

W-SMALL-DATA, W-IMBALANCE, W-HIGH-MISSING, W-LEAK-STRONG, W-LEAK-NEAR-DETERMINISTIC, W-GROUP-SUSPECTED, W-TEMPORAL, W-DUPLICATES, W-CONFLICTING-DUPLICATES, W-WIDE (§9.3); W-ENCODING-FALLBACK (§8.2); W-COERCED (§8.5); W-INF (§8.6); W-TARGET-MISSING (§8.7); W-NO-HOLDOUT (§11.1). Fifteen codes in total. Only the first ten are catalogued in §9.3 (PI-10).
