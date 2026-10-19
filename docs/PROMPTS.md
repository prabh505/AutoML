# TabPilot — Build Prompt Library

> Ordered prompts for building TabPilot end to end with Claude Code (Claude Opus).
> Each prompt is one session, one coherent unit of work, owned by one domain expert, ending in a validated, committed, documented state.

---

## How to use this file

### Repository layout before the first session

```
CLAUDE.md          ← operating rules (Claude Code loads this automatically)
CONTEXT.md         ← project memory (read first, updated every session)
docs/PRD.md        ← the specification (rename TabPilot_PRD.md to this)
docs/PROMPTS.md    ← this file
```

### The loop for every prompt

1. **Build.** Start a Claude Code session on the repository and send the **Kickoff message** below. Claude executes the next pending prompt, tests it, updates `CONTEXT.md`, commits, and reports.
2. **Validate.** Read the report. Then start a **fresh session** and send **U1 Independent audit** for that prompt. A fresh session matters: the auditor has not seen the builder's reasoning, so it checks the code, not the story.
3. **Correct if needed.** If the audit verdict is not PASS, send **U2 Course correction** (same or new session).
4. **Merge** the branch or pull request into `main` before starting the next prompt, so every session starts from the latest `CONTEXT.md`.

At each **Gate** (after P09, P15, P16, P20), also do the human check described in that prompt before continuing.

### Kickoff message (send this to start each prompt)

```
Read CLAUDE.md and CONTEXT.md completely.
Then open docs/PROMPTS.md and execute the next prompt whose status in the
CONTEXT.md Prompt Ledger is "pending" — that one prompt only.

Follow the SOP in CLAUDE.md exactly:
- read every PRD section the prompt lists, in full, before writing code;
- implement only the prompt's scope;
- write and run the tests it requires, plus the full suite;
- run its Validation commands and Self-review checklist;
- update CONTEXT.md, commit, and finish with the Report Format.

Do not start the following prompt. If anything is ambiguous or blocked,
follow the Deviation Protocol and report instead of guessing.
```

### Optional: run several prompts in one session (after you trust the process)

```
Read CLAUDE.md and CONTEXT.md completely. Execute the pending prompts in
docs/PROMPTS.md in order, one after another, following the full SOP for
each (including the CONTEXT.md update, commit, and report after each one).
Stop immediately when you reach the end of a prompt marked GATE, when any
prompt ends with status other than COMPLETE, or when tests are not green.
```

Recommendation: use one prompt per session at least until Gate A. The audit loop is your early-warning system.

### When to use the utility prompts

| Prompt | Use when |
| --- | --- |
| **U1 Independent audit** | After every build prompt, in a fresh session |
| **U2 Course correction** | The audit found problems, or you noticed the build going off course |
| **U3 Session resume / AI handoff** | A session died mid-prompt, or you move the project to another AI or person |
| **U4 Change request** | You want to change a requirement in the PRD |

### Prompt anatomy

Every build prompt has the same sections: **Role** (the expert you are hiring), **Mission**, **Read first**, **Depends on**, **Scope** (in and out), **Deliverables**, **Requirements and critical correctness points** (where experienced engineers know mistakes happen), **Tests**, **Validation**, **Self-review checklist**, and **Done when**. The SOP in `CLAUDE.md` applies to all of them and is not repeated.

---
---

# MILESTONE 0 — Foundations

---

## P00 — Project charter and context bootstrap

**Role.** You are a Principal Technical Program Manager and Chief Architect who has led the delivery of ML platforms at Fortune-50 scale. You are known for turning large specifications into traceable plans where nothing is lost between requirement, code, and test.

**Mission.** Before any code exists, prove the plan covers the whole PRD, surface every inconsistency in the PRD, and make `CONTEXT.md` a complete, accurate project memory.

**Read first.** `CLAUDE.md`; `CONTEXT.md`; **all of `docs/PRD.md` (§0–§40), in full**; `docs/PROMPTS.md` (all prompts, to learn the plan).

**Depends on.** Nothing.

**Scope.**
- In: documentation only — traceability, ADRs, context.
- Out: any source code, configuration, or dependency files.

**Deliverables.**
1. `docs/TRACEABILITY.md` with four tables:
   - **Functional requirements**: every `FR-*` in PRD §4 → PRD sections → owning prompt (P01–P20) → test IDs → gate.
   - **Tests**: every test ID in PRD §31.3 (expand ranges like `T-ING-03..07` into individual IDs) → owning prompt.
   - **Non-functional requirements**: every `NFR-*` in §33 → verifying prompt and method.
   - **Release gates**: every `G-*` in §38 → prompts that produce the evidence.
2. `docs/adr/0000-template.md` (Context, Decision, Alternatives, Consequences, Status), `docs/adr/0001-baseline-decisions.md` summarizing C-01…C-17 and the D-01…D-16 defaults, and `docs/adr/0002-runtime-package.md` for amendment A-001.
3. `CONTEXT.md` updates: §11 **PRD issues found** (see below), §12 open questions refined, ledger P00 complete, change log, handoff note.

**Requirements and critical correctness points.**
- Read the PRD as an adversarial reviewer. Look for: requirements with no test; tests with no requirement; numbers that differ between sections (limits, thresholds, budgets, timeouts); terms used inconsistently; references to non-existent sections; behavior that is specified in two places differently; requirements impossible on a ~1 GB host; and anything that conflicts with amendment A-001. Record each in CONTEXT §11 with a proposed resolution. Known candidates to check at minimum: time-budget shares in §29.3 vs §19.4; the reflection minimum budget in §13.5 vs §19.4; T-LEAK-08 wording; whether every E-code used anywhere appears in §8.9; and whether the parity fixture in §22.5 is consistent with the privacy rules in §24.4.
- Every FR, test ID, NFR, and gate must appear exactly once as "owned" by a prompt. If a requirement has no owning prompt, assign it to the most appropriate one and note it in the Change Log.
- Do not modify `docs/PRD.md`. Do not resolve issues that change product behavior on your own; mark them `proposed` and list them as open questions.

**Tests.** None (documentation prompt). Instead, run a script (scratch, not committed) that extracts every `FR-`, `T-`, `NFR-`, `G-` ID from the PRD and confirms each appears in `docs/TRACEABILITY.md`. Report the counts.

**Validation.** Paste the ID-coverage counts (found in PRD vs mapped) for FR, T, NFR, G. All must be 100%.

**Self-review checklist.**
- Did I read every PRD section, not just headings?
- Is every test ID individually mapped (ranges expanded)?
- Is every PRD issue concrete (section, quote, conflict, proposed fix)?
- Does CONTEXT.md now let a new engineer start P01 with no other information?

**Done when.** Traceability is 100%, PRD issues are recorded, ADRs exist, CONTEXT.md is updated and committed.

---

## P01 — Repository scaffold, toolchain, CI skeleton

**Role.** You are a Staff DevOps and Python Platform Engineer who designs repositories that stay clean for years: reproducible environments, fast feedback, and enforced architecture boundaries.

**Mission.** Create the repository skeleton, toolchain, and CI so every later prompt lands in a strict, consistent, automatically checked environment.

**Read first.** PRD §25 (all), §34.1–§34.5, §33 (NFR-MNT-*), §2.2; `CONTEXT.md` A-001; `docs/TRACEABILITY.md`.

**Depends on.** P00.

**Scope.**
- In: directory structure, packaging, dependency management, lint/type/test/contract configuration, Makefile, CI workflow `ci.yml`, config YAML skeletons with PRD values, entrypoint shell.
- Out: business logic of any module. Dockerfile, perf/benchmark/release workflows (P18).

**Deliverables.**
1. Repository tree exactly per PRD §25.1, **plus** `src/tabpilot_runtime/` (A-001). Each package has `__init__.py` with a module docstring stating its responsibility and forbidden responsibilities (copied from §25.2). No placeholder functions.
2. `pyproject.toml`: project metadata, Python `>=3.11,<3.12`, core runtime dependencies from §34.2, extras `[xgboost]`, `[catboost]`, `[mlflow]`, `[bench]`, `[dev]`. Choose current stable versions; require `scikit-learn` with `TargetEncoder`, `OneHotEncoder(min_frequency/max_categories)`, `set_output(transform="pandas")` and `StratifiedGroupKFold` support. Dev tools: pytest, pytest-cov, pytest-timeout, pytest-socket (block network in tests), hypothesis, ruff, mypy, import-linter, pip-audit, pre-commit.
3. `uv.lock`; `requirements.txt` exported for the **hosted** subset (no xgboost, catboost, mlflow, flaml) with a script `scripts/export_requirements.sh` and a CI check that it matches the lock.
4. Tool configuration in `pyproject.toml`: ruff (lint + format), mypy (strict for `schemas`, `evaluation`, `splitting`, `llm`; standard elsewhere), pytest markers `unit integration e2e security perf ui slow live`, default `-m "not perf and not live and not slow"`, network disabled by default, coverage config.
5. Import-linter contracts encoding PRD §25.4 rules 1–4, plus: `tabpilot_runtime` must not import `tabpilot`.
6. `Makefile`: `setup`, `lint`, `format`, `type`, `contracts`, `test`, `test-all`, `check` (lint + format-check + type + contracts + test), `schemas`, `requirements`.
7. `config/profiles.yaml`, `config/models.yaml`, `config/providers.yaml` populated with the exact values from PRD §29.1, §16.1, §23.3 (env-var placeholders where the PRD uses them).
8. `.github/workflows/ci.yml` running `make check` on Python 3.11 with uv caching; `.pre-commit-config.yaml`; `.gitignore` (data, runs, `.env`, mlruns, caches); `.env.example` with variable names only.
9. `streamlit_app.py`: a thin entrypoint that imports the UI package's `main()`; the UI package exposes `main()` which renders the app title and a one-line status ("TabPilot is being built — see README"). This is the only allowed shell, and P16 replaces it.
10. `tests/unit/test_smoke.py`: imports every package; verifies config files load as YAML and contain the PRD values for the `hosted` profile.

**Requirements and critical correctness points.**
- Pin exactly via `uv.lock`; record chosen versions of all core packages in CONTEXT §7 with a one-line reason each.
- Confirm LightGBM, SHAP, and Optuna install and import on Python 3.11 in this environment. If installation fails because of network restrictions, record it in Unverified Items and make CI the verifier.
- Do not add any dependency outside PRD §34.2 and the dev tools above without a Decision Log entry.

**Tests.** Smoke tests above.

**Validation.** `uv sync --all-extras`; `make check` (paste output); `uv run lint-imports`; `scripts/export_requirements.sh && git diff --exit-code requirements.txt`.

**Self-review checklist.**
- Does the tree match §25.1 plus `tabpilot_runtime`?
- Do import contracts fail if I add a forbidden import (try one, then revert)?
- Is network blocked in tests by default?
- Are config values identical to the PRD tables?

**Done when.** `make check` is green locally and the CI workflow is valid; CONTEXT §7 lists pinned versions.

---

## P02 — Data contracts, configuration, logging

**Role.** You are a Principal Python Backend Engineer and API-contract architect. You design typed, versioned contracts that let a dozen engineers work in parallel without integration surprises.

**Mission.** Implement every data contract, the configuration system, the error/warning catalog, and privacy-safe structured logging, so all later modules communicate through validated types.

**Read first.** PRD §26 (all), §8.9, §9.3 (W-codes), §10.2 (metric names), §12.2–§12.3, §13.3, §14.3 (DSL catalog and param bounds), §18.1, §19.2–§19.3, §22.3, §23.1, §23.7, §28.2 (`RunConfig`), §29.1, §30.2–§30.3.

**Depends on.** P01.

**Scope.**
- In: `tabpilot.schemas` (all contracts), `tabpilot.config`, `tabpilot.logging`, `scripts/export_schemas.py`, `docs/schemas/`.
- Out: any logic that produces these objects (other prompts).

**Deliverables.**
1. Pydantic v2 models for: `DatasetMetadata`, `ColumnInfo`, `ParseReport`, `TargetReport`, `TaskDetection`, `FeasibilityReport`, `SplitSpec`, `ProfileSummary`, `CompactProfile`, `PlannerInput`, `PlannerOutput` (`plan/1.0`), `PlanValidation`, `FeatureSpec` (discriminated union on `op`, one param model per op with the exact bounds of §14.3), `FeatureProvenance`, `ModelConfig`, `CandidateSpec`, `ExperimentConfig`, `TrainingResult`, `LeaderboardEntry`, `EvaluationResult`, `SelectionRecord`, `ReflectionInput`, `ReflectionProposal` (change types of §19.3 as a discriminated union), `ReflectionRecord`, `EnsembleResult`, `FinalReport`, `ArtifactManifest`, `RunEvent`, `RunConfig`, `RunResult`, `RunState`, `BudgetLedger`, `LLMBudget`, `LLMRequest`, `LLMResponse`, `Warning`, `StageError`, `MetricSpec`.
2. `schemas/errors.py`: `ErrorCode` enum with every E-code in §8.9 (plus any added via the Decision Log), message templates, and a `TabPilotError` exception carrying code + params; `WarningCode` enum for every W-code used in the PRD.
3. `tabpilot.config`: loads the three YAML files, interpolates `${ENV}` values, validates with Pydantic, selects the profile from `TABPILOT_PROFILE` (default `local`), and exposes a frozen `Settings` object. Missing optional provider env vars are allowed; malformed config fails fast with a clear message.
4. `tabpilot.logging`: JSON structured logging (`structlog` or stdlib JSON formatter — record the choice) with a scrubber that redacts API-key patterns (§30.3) and refuses to log values tagged as data. Provide `get_logger(stage)` and a `bind_run(run_id)` helper.
5. `scripts/export_schemas.py` writing JSON Schema for every contract to `docs/schemas/`; Makefile target `schemas`; CI step failing if schemas changed without a `schema_version` bump.

**Requirements and critical correctness points.**
- `ConfigDict(extra="forbid")` on every model, including LLM-output models (`PlannerOutput`, `ReflectionProposal`) — unknown keys must be rejected so the repair loop sees them. `frozen=True` for all except `RunState`.
- Every model has a `schema_version` literal matching §26.
- Encode bounds from the PRD as field constraints (e.g. features ≤ 12, rationale ≤ 800 chars, reason ≤ 200 chars, models 1–5, `cv_folds ∈ {3,5}`, eps range, quantile ranges, `n_bins` 2–20, input arity per op). This makes later validators thin and consistent.
- Make every JSON example in PRD §26 concrete (replace `…` with valid values) and store them in `tests/fixtures/contracts/`; each must validate.
- Logging must never call `repr()` on DataFrames or Series. Provide a guard that raises in tests if a pandas object is passed to the logger.

**Tests.** Round-trip (dump → load → equal) for every model; every PRD example validates; extra keys rejected; bounds enforced (parametrized edge values); `FeatureSpec` discriminator selects the right param model; config loads `hosted` and `local` with PRD values; scrubber redacts `sk-…`, `gsk_…`, `AIza…`, `sk-or-…`; DataFrame passed to logger raises in tests.

**Validation.** `make check`; `make schemas && git diff --stat docs/schemas`.

**Self-review checklist.**
- Is every contract named in §26, §13.3, §23.1, §28.2 implemented?
- Would a malformed LLM plan with one extra key fail validation?
- Can any secret reach the log output?

**Done when.** All contracts, config, and logging are implemented and tested; `docs/schemas/` generated; CONTEXT updated.

---

## P03 — Deterministic fixtures and test harness

**Role.** You are the QA and ML Evaluation Lead. You have built test suites for safety-critical ML where "it passed" must mean something. You design fixtures whose ground truth is known in advance, so tests check correctness, not just "no crash".

**Mission.** Build every fixture dataset and every shared test utility the rest of the project needs, with documented ground truth.

**Read first.** PRD §31 (all), §8 (to know what parsing/quality fixtures must trigger), §11 (leakage tests), §12.6 (invalid plans), §23.5 (provider failure modes), §24 (malicious inputs).

**Depends on.** P02.

**Scope.**
- In: `fixtures/make_fixtures.py`, committed small fixtures, `fixtures/MANIFEST.yaml`, `tests/conftest.py` and `tests/helpers/`.
- Out: tests for features not yet built (later prompts write them using this harness).

**Deliverables.**
1. `fixtures/make_fixtures.py`: generates every dataset in PRD §31.2 deterministically (fixed seeds, no wall-clock dependence). Each synthetic dataset has a **known generating process** (which features are informative, by how much, which are noise), documented in the manifest.
2. Committed fixtures under `fixtures/datasets/`, `fixtures/parsing/`, `fixtures/malicious/` (all small). `hosted_limit.csv` (100K × 30) is **not committed**; provide `generate_hosted_limit(path)` used by perf tests.
3. `fixtures/plans/*.json`: a catalogue of planner outputs — valid, invalid JSON, unknown column, excluded column, target column, prohibited metric, unknown model, > 12 features, unknown DSL op, out-of-range params, code-like strings in params, contradictory column decisions, > 50% unsafe features, prompt-injection text in rationale — each listed in the manifest with its **expected validator outcome** (valid / repaired + issues / invalid).
4. `fixtures/llm_responses/`: scripted response sequences per provider adapter (success, malformed JSON, fenced JSON, 429 with retry-after, 500, timeout, 401), in a format the P10 stub provider can replay.
5. `fixtures/MANIFEST.yaml`: for every fixture — purpose, shape, expected task, expected semantic types, expected warnings/E-codes, expected leakage findings, and the test IDs that use it.
6. `tests/conftest.py` and `tests/helpers/`:
   - `FitSpy` mixin/wrapper that records the row indices each transformer/estimator was fitted on (for T-LEAK-07).
   - `GuardedFrame` that raises when holdout row indices are accessed (for T-LEAK-02).
   - `ScriptedLLMProvider` stub implementing the `LLMProvider` protocol (§23.1) from fixture scripts.
   - Network-blocking fixture (default on), `allow_network` opt-in for `live` tests.
   - Fake clock / sleeper for retry and budget tests.
   - `tmp_run_dir` fixture; helpers to assert an E-code is raised with correct params.
7. A fixture regeneration check: `make fixtures-check` regenerates into a temp dir and verifies byte-identical output (determinism).

**Requirements and critical correctness points.**
- Ground truth must make tests meaningful: e.g. `leaky_binary.csv` has one column equal to the target with 0.5% flips (near-deterministic) and one strong but legitimate predictor (AUC between 0.95 and 0.99); `grouped.csv` has group-level signal so group leakage would inflate CV detectably; `regression_skewed.csv` is lognormal so `log1p` matters.
- `quality_issues.csv` must trigger every check in §8.6 and every case in US-04.
- Malicious fixtures: ZIP renamed `.csv`, column names containing prompt-injection text and HTML/script tags, cells starting with `=`/`+`/`-`/`@`, a 300-character filename, nested Parquet, a header-only file, an empty file.
- No fixture contains real personal data.

**Tests.** Tests of the harness itself: `FitSpy` records indices correctly; `GuardedFrame` raises; `ScriptedLLMProvider` replays sequences; network is blocked; fixtures regenerate identically; manifest lists every file and every file is listed.

**Validation.** `make check`; `make fixtures-check`.

**Self-review checklist.**
- Does every fixture in §31.2 exist (or have a generator, for the large one)?
- Is every expected outcome written down before the code that produces it exists?
- Would a leakage bug in a later module be caught by these fixtures?

**Done when.** Fixtures and harness are complete, deterministic, documented, and tested.

---
---

# MILESTONE 1 — Deterministic AutoML core

---

## P04 — Ingestion, validation, feasibility

**Role.** You are a Principal Data Engineering Architect who has built ingestion layers that process millions of messy customer files. You never trust input, never drop data silently, and always tell users exactly what to fix.

**Mission.** Turn an uploaded file into a typed, normalized, validated dataset with complete metadata and an honest resource-feasibility verdict.

**Read first.** PRD §8 (all), §29.1, §29.4, §29.5, §24.2, §4.1 (FR-ING-01…05), FR-TGT-01, §31.3 rows T-ING-*, T-TGT-01…06, T-RES-01…02, T-SEC-04.

**Depends on.** P02, P03.

**Scope.**
- In: `tabpilot.ingestion` (readers, encoding/delimiter detection, normalization, type inference, column checks, target validation), `tabpilot.resources.estimator` (memory estimator v0 and simplification planner), E-code raising.
- Out: task detection (P05), profiling (P06), UI (P16).

**Deliverables.**
1. `load_dataset(path, display_name, settings) -> LoadedDataset` (DataFrame + `DatasetMetadata` + `ParseReport`).
2. `validate_target(loaded, target) -> TargetReport` and the row-removal rules of §8.7.
3. `estimate_feasibility(metadata, run_config, settings) -> FeasibilityReport` implementing §29.4 with `SAFETY = 1.5` until calibrated (P18), and the simplification order of §29.5.
4. Column-name normalization (§8.4) with `column_map`, idempotent.
5. Type inference (§8.5) and column checks (§8.6) with complete exclusion reasons and coercion counts.

**Requirements and critical correctness points.**
- **Size check before reading content** (the parser must not be invoked for oversized files — T-ING-01 uses a spy).
- Magic-byte verification: ZIP (`PK\x03\x04`), Excel/OLE signatures, Parquet `PAR1` at both ends. Archives are rejected with `E-ARCHIVE-UNSUPPORTED`.
- **Malformed-line handling trap:** pandas only supports a callable `on_bad_lines` with the slow Python engine. Do not switch engines silently. Recommended: pre-scan the text with the `csv` module (respects quoting and embedded newlines) to count fields per record and collect malformed record line numbers, then parse with the C engine, skipping exactly those records. Record the approach in the Decision Log as an implementation detail.
- **String-first parsing** (`dtype=str`, `keep_default_na=False` then explicit NA tokens) so type inference is yours, not pandas' guess.
- Decimal-comma detection only applies per column as specified; record it in `ParseReport`.
- ID-like exclusion requires the **value rule**; a name match alone never excludes a column.
- Datetime: only when ≥ 95% parse with one inferred format; never let pandas guess per element.
- Float downcasting must not lose precision for values where float32 is insufficient (§8.5).
- **No-silent-discard rule:** every dropped row, coerced value, and excluded column appears in `DatasetMetadata`.
- Filenames: sanitized for display (§24.2); storage paths never contain user input.
- Errors are `TabPilotError` with the exact E-codes and message parameters of §8.9.

**Tests.** T-ING-01…13, T-TGT-01…06, T-RES-01, T-RES-02, filename part of T-SEC-04, plus property-based tests (hypothesis) for name normalization (idempotence, uniqueness, allowed characters).

**Validation.** `make check`; coverage of `tabpilot.ingestion` ≥ 90%.

**Self-review checklist.**
- Is there any code path where a row or column disappears without a metadata record?
- Is every E-code in §8.9 relevant to ingestion raised by at least one test?
- Could a crafted filename reach a filesystem path?

**Done when.** All listed tests pass on every relevant fixture; metadata is complete.

---

## P05 — Task detection, metrics, splitting, holdout guard

**Role.** You are a Principal Machine Learning Engineer specializing in evaluation methodology and leakage prevention. You have audited dozens of production models whose reported accuracy collapsed in production, and you build splits that make that impossible.

**Mission.** Implement task detection, the metric system, and the split protocol — including the sealed holdout — exactly as specified, because every later score depends on them.

**Read first.** PRD §10 (all), §11.1–§11.6, §11.10 (benchmark split note), FR-TGT-02, FR-TGT-03, FR-SPL-01, §31.3 rows T-TGT-07…11, T-LEAK-01, T-LEAK-03, T-LEAK-04.

**Depends on.** P04.

**Scope.**
- In: `tabpilot.ingestion.task` (or `tabpilot.evaluation.task`; record the choice), `tabpilot.evaluation.metrics`, `tabpilot.splitting` (strategies, seeds, `split_id`, duplicate grouping, `HoldoutGuard`).
- Out: CV training loop (P08), selection (P09).

**Deliverables.**
1. `detect_task(target_series) -> TaskDetection` per §10.1, including positive-label rule.
2. Metric registry: name, direction, required prediction type, `compute(y_true, y_pred, labels)`, `selection_score`. Primary and secondary metrics per §10.2; F1-optimal threshold computed on OOF predictions; probability clipping; class alignment.
3. `build_split(y, task, settings, group=None, seed_root) -> SplitSpec` per §11.3 (all strategies, K reduction, repeated CV for < 2,000 rows, holdout 20% stratified/quantile-binned/group-aware, duplicate grouping §11.6), seed derivation, deterministic `split_id` hash.
4. `HoldoutGuard` per PRD L-2: `open(caller)` raises `HoldoutSealed` unless the frozen-state callable returns `True`; logs every access; second open is allowed only for the reference baseline evaluation and is logged (define precisely and document; T-LEAK-03 expects one access event per run — record your interpretation in the Decision Log).

**Requirements and critical correctness points.**
- Regression holdout: stratify on 10 target-quantile bins (handle duplicate bin edges).
- Group holdout: `GroupShuffleSplit` does not stratify; for classification with groups, verify both partitions contain every class or fall back with a recorded warning.
- Multiclass log loss must pass the full `labels` list so folds with missing classes compute correctly.
- AP/ROC-AUC use the positive class column; never assume column 1 without the label encoder mapping.
- Metrics are always on the original target scale (§10.4).
- Determinism: same inputs + seed → identical `split_id`, folds, and holdout, across runs and processes.

**Tests.** T-TGT-07…11; T-LEAK-01, T-LEAK-03 (guard level), T-LEAK-04; plus: every strategy branch of §11.3; K reduction; repeated-CV OOF averaging contract; metric values against hand-computed references; selection-score direction; duplicate grouping keeps duplicates together across folds and DEV/HOLD.

**Validation.** `make check`; coverage of `splitting` and `evaluation.metrics` ≥ 95%.

**Self-review checklist.**
- Can any code obtain holdout indices without the guard?
- Is the split identical when the same run is repeated in a fresh process?
- Are all metric directions correct (write them out in the report)?

**Done when.** All tests pass; coverage thresholds met.

---

## P06 — Profiling, leakage scan, compact profile

**Role.** You are a Senior Data Scientist specializing in exploratory analysis, data quality, and leakage detection. Your profiles catch the problem before the model hides it — and you know exactly which statistics are safe to share outside the data boundary.

**Mission.** Build the DEV-only profiling engine, the leakage and structure scans, and the bounded, privacy-safe compact profile that the LLM will see.

**Read first.** PRD §9 (all), §11.7, §24.4, FR-PRF-01…03, §26.2, §31.3 rows T-PRF-*, T-LEAK-05, T-LEAK-06, T-PRIV-01.

**Depends on.** P04, P05.

**Scope.**
- In: `tabpilot.profiling` (statistics, warnings, leakage scan, group/temporal detection, compact profile builder incl. minimal mode).
- Out: planner input assembly (P11), UI display (P16).

**Deliverables.**
1. `profile(dev_frame, metadata, task, settings, seed) -> ProfileSummary` with every statistic in §9.2 and per-group time caps (§9.5) recording `truncated_stats`.
2. Leakage scan (§11.7): single-feature depth-3 tree, 3-fold CV on ≤ 20K DEV rows, thresholds for strong and near-deterministic levels, heuristics as warnings. Near-deterministic columns become `excluded` with reason `leakage_near_deterministic`, re-includable through `RunConfig.column_overrides`.
3. Warnings (§9.3) with codes from `WarningCode`.
4. `build_compact_profile(profile, mode="standard"|"minimal", max_chars=14000) -> CompactProfile` using the truncation algorithm of §9.4, with deterministic ordering.

**Requirements and critical correctness points.**
- Profiling receives **only DEV rows**. Make the function signature impossible to misuse (accept a DEV view type or the split + full frame and select internally); document which.
- The compact profile contains **no cell values**: no category labels, no example rows, no exact values from identifier-like columns. Categorical columns expose shares and entropy only. Numeric quantiles are allowed in standard mode and removed in minimal mode.
- Frequency-encode categoricals inside each leakage-scan fold, not globally.
- Mutual information on high-cardinality columns uses top 50 categories + "other".
- Deterministic: same DEV + seed → byte-identical compact profile JSON (canonical serialization).

**Tests.** T-PRF-01…04, T-LEAK-05, T-LEAK-06, T-PRIV-01 (implement as a property-based test: for random fixtures, no string cell value and no high-precision numeric value from identifier/categorical columns appears in the serialized profile), size bound on `wide.csv`, minimal-mode content, determinism.

**Validation.** `make check`; coverage of `profiling` ≥ 90%.

**Self-review checklist.**
- Could any categorical label leak into the compact profile through any field (including warnings text or column flags)?
- Does profiling ever read HOLD rows?
- Is the profile size bounded for 200 columns?

**Done when.** Profiles are correct, bounded, deterministic, and privacy-safe.

---

## P07 — Feature engine: baseline transformations, Feature DSL, runtime package

**Role.** You are a Staff ML Engineer specializing in feature engineering, with a secure-by-design mindset. You build transformations that are fold-safe, serializable, reproducible, and impossible to turn into a code-execution channel.

**Mission.** Implement the deterministic baseline features and the Feature DSL that turns LLM proposals into safe, fold-local scikit-learn transformers, inside the `tabpilot_runtime` package.

**Read first.** PRD §14.1–§14.6, §15.3 (rare categories), §11.2 L-4, §24.6, FR-FE-01, FR-FE-02; `CONTEXT.md` A-001; §31.3 rows T-FE-*, T-LEAK-07, T-SEC-01.

**Depends on.** P02, P03, P05.

**Scope.**
- In: `tabpilot_runtime` transformers (`BaselineFeatureTransformer`, `DSLFeatureTransformer`, op implementations, rare-category grouper, frequency encoder); `tabpilot.features` (catalog exposure for the planner, static validator, provenance builder); the static code-safety scan test.
- Out: pipeline assembly (P08), plan validation (P11 reuses your validator).

**Deliverables.**
1. Op implementations for every op in the §14.3 catalog with the NaN semantics of §14.5.
2. `DSLFeatureTransformer(specs)`: scikit-learn compliant (`get_params`/`set_params`, `clone`-safe, all state set in `fit`, `transform` never refits), output dtypes and order per §14.5, `llm__` name prefix and collision suffixing, composition depth ≤ 2, degenerate-feature detection reported via a `failed_features_` attribute.
3. `validate_spec(spec, schema) -> ValidationResult` and `validate_specs(specs, schema)` implementing every check of §14.4, including canonical-form deduplication. The planner (P11) and reflector (P12) must reuse this — not reimplement it.
4. `catalog_for_planner()` producing the `dsl_catalog` field of `PlannerInput`.
5. `FeatureProvenance` builder.
6. `tests/security/test_T_SEC_01_no_dynamic_eval.py`: an AST scan of `src/` that fails on `eval`, `exec`, `compile`, `pd.eval`, `DataFrame.query`, `DataFrame.eval`, `__import__`, and `importlib.import_module` with non-literal arguments.

**Requirements and critical correctness points.**
- **Zero code execution:** spec fields are data. Op dispatch is a fixed dictionary from op name to a function you wrote. No string formatting into expressions, no `getattr` on user-controlled names.
- Stateful ops (`clip`, `bin_quantile`, `group_*`, `days_since_min`, rare grouping, frequency encoding) learn state **only** in `fit`. Unseen keys → NaN.
- `tabpilot_runtime` imports nothing from `tabpilot` (import contract).
- Per-feature runtime cap (§14.5): first-party vectorized ops cannot hang, so measure elapsed time per op during `transform` and mark the feature failed if it exceeds the cap; document that this is a soft cap and why it is sufficient for first-party code.
- Everything must pickle and unpickle with only `tabpilot_runtime` + third-party packages importable.

**Tests.** T-FE-01…08; T-LEAK-07 for every stateful op (fit indices ⊆ training fold via `FitSpy`); T-SEC-01; pickle round-trip in a subprocess with `tabpilot` hidden from `sys.path`; sklearn `clone` equivalence; NaN semantics table tests; canonical dedup of commutative ops.

**Validation.** `make check`; coverage of `features` and `tabpilot_runtime` ≥ 95%.

**Self-review checklist.**
- Is there any path from a spec string to executable code?
- Does any stateful op compute statistics in `transform`?
- Can the fitted transformer load in a process where `tabpilot` is not installed?

**Done when.** DSL and baseline features are complete, safe, fold-local, and picklable.

---

## P08 — Preprocessing pipelines, model zoo, CV runner

**Role.** You are a Principal Machine Learning Engineer who has shipped scikit-learn pipelines that serve production traffic. You know every sharp edge of `ColumnTransformer`, categorical handling in boosting libraries, and early stopping inside cross-validation.

**Mission.** Build the pipeline factory, the model registry, and the cross-validation runner that produces fold scores and OOF predictions without leakage.

**Read first.** PRD §15 (all), §16 (all), §11.4, §11.8, §10.3, FR-PRE-01, FR-TRN-01, §29.6 (RAM guard hooks), §31.3 rows T-PRE-*, T-TRN-*, T-LEAK-07, T-LEAK-08.

**Depends on.** P05, P07.

**Scope.**
- In: `tabpilot_runtime` (`ColumnNameMapper`, `TypeCaster`, `ColumnSelector`, early-stopping estimator wrapper), `tabpilot.preprocessing` (pipeline factory per model family), `tabpilot.models` (registry, availability detection, `hist_gb` fallback, class-weight mapping, default params), `tabpilot.training` (CV runner, OOF store, fold failure and fallback-parameter logic, feature-failure restart).
- Out: Optuna (P09), selection (P09), budgets beyond simple hooks (P09).

**Deliverables.**
1. `build_pipeline(candidate, schema, task) -> Pipeline` (or `TransformedTargetRegressor`) with the exact layout of §15.1 and branches of §15.2, per model family.
2. Model registry reading `config/models.yaml`, detecting importable libraries at startup, exposing `available(profile)` and `make_estimator(name, params, task, seed, n_jobs)`.
3. `run_cv(candidate, dev_data, split_spec, cancel, budget_hook) -> TrainingResult` with OOF (float32) and fold scores, per §11.4, including repeated-CV averaging.
4. Early-stopping wrapper estimator: splits the already-preprocessed training fold 90/10 (stratified, fixed seed) internally for early stopping, records best iteration; the outer validation fold is never used.

**Requirements and critical correctness points.**
- **Target encoding:** scikit-learn's `TargetEncoder` cross-fits only inside `fit_transform`. Verify with a test that `Pipeline.fit` calls `fit_transform` on it (so training rows are encoded out-of-fold) and that `transform` on validation rows uses the full-fold fit. T-LEAK-08 (canary) must hold.
- **LightGBM categoricals:** choose one consistent mechanism (ordinal-encoded integer columns declared as categorical, or pandas `category` dtype) and use it identically in CV, refit, and export. Unknown categories → the reserved code.
- Use `set_output(transform="pandas")` so feature names survive for SHAP and reporting.
- Class order: always align `predict_proba` columns to the label encoder's classes; never assume.
- `TransformedTargetRegressor` for `log1p`; metrics on the original scale.
- Fold failure semantics per §16.3: fold 1 failure → retry once with defaults → else `failed` with sanitized reason. A DSL feature failure in any fold → drop it for the candidate and restart from fold 1 (§14.6); never mix feature sets across folds.
- Fitted fold pipelines are discarded after scoring (memory, §29.6).

**Tests.** T-PRE-01…05, T-TRN-01…05, T-LEAK-07 (every stateful step: imputers, encoders, target encoder, scaler, DSL ops), T-LEAK-08 canary on pure-noise data with high-cardinality categorical and target encoding (AUC ≤ 0.55 over 5 seeds), every model × task on fixtures, injected model failure isolation, row-vs-batch transform consistency.

**Validation.** `make check`; coverage of `training` and `preprocessing` ≥ 90%.

**Self-review checklist.**
- Does any learned statistic touch validation-fold rows?
- Are probability columns aligned to classes in every model family?
- Does every pickled object come from `tabpilot_runtime` or third-party libraries?

**Done when.** Every model trains on every task fixture with correct OOF and fold scores, and the leakage tests pass.

---

## P09 — Tuning, evaluation, selection, holdout, budgets, rule-based CLI  ·  **GATE A**

**Role.** You are a Principal AutoML Architect who has built hyperparameter search and model-selection systems used by thousands of data scientists. You care about honest comparisons, strict budgets, and reproducible winners.

**Mission.** Complete the deterministic AutoML core: budgeted Optuna tuning, leaderboard, selection rules, final refit, one-time holdout evaluation, resource monitoring, run records, the rule-based planner, and a headless CLI. This is the LLM-free product (benchmark ablation A2).

**Read first.** PRD §17 (all), §18 (all), §11.10, §12.8, §29.3, §29.6, §30.1, FR-TUN-01, FR-EVL-01, FR-EVL-02, FR-PLN-03, §31.3 rows T-TUN-*, T-EVL-*, T-LEAK-02, T-LEAK-03, T-RES-03, T-PLN-09; `CONTEXT.md` DEC-004.

**Depends on.** P04–P08.

**Scope.**
- In: `tabpilot.tuning`, `tabpilot.evaluation` (leaderboard, comparability, selection, uncertainty, holdout evaluation with bootstrap CI), final refit logic (§11.10), `tabpilot.resources` (budget ledger §29.3, RSS monitor §29.6), `tabpilot.tracking` (run records §30.1), `tabpilot.planning.fallback` (rule-based plan §12.8 producing a valid `PlannerOutput`), CLI `tabpilot run <file> --target <col> [--task] [--budget-s] [--seed] [--profile] --out <dir>`.
- Out: LLM, reflection, ensembling, explainability, export, LangGraph, worker process (later prompts). The CLI here calls stages directly in-process; P15 will route it through orchestration.

**Deliverables.**
1. `tune(model_name, candidate, dev_data, split_spec, allocation_s, settings, cancel) -> TuningResult` per §17.4 (TPE seeded, median pruner on per-fold running mean, trial 0 = defaults, per-trial timeout, failure accounting, 5-consecutive-failure stop, within-fold subsampling §17.6, `tuning.mode = "trials" | "time"`).
2. Leaderboard builder and `select()` per §18.2 with tie-breaks, comparability enforcement, failed-entry handling, runner-up paired differences.
3. Finalize: refit selected config on all DEV (boosting `n_estimators` = mean best iteration × 1.1), freeze selection, open the holdout through `HoldoutGuard`, score final model and reference baseline, bootstrap 95% CI (1,000 resamples, seeded).
4. Budget ledger enforcing §29.3 shares (with roll-forward and the reserved 15% slice) and an RSS monitor (psutil, 1 s sampling, soft/hard limits, `gc.collect`, candidate abort).
5. Run records writer (`events.jsonl`, `stage_timings.json`, `trials.jsonl`, `resources.jsonl`, `experiment.json`).
6. Rule-based planner (§12.8) — deterministic, returns a validated `PlannerOutput`.
7. CLI producing the run directory with records, leaderboard, selection, and holdout JSON.

**Requirements and critical correctness points.**
- Tuning never reads HOLD (T-LEAK-02 with `GuardedFrame`).
- The leaderboard's mean/std must be recomputable from stored fold scores (T-EVL-01).
- In `time` mode results depend on hardware — document; tests use `trials` mode.
- The ledger must stop work at its deadline (T-TUN-01: elapsed ≤ allocation × 1.1 + 15 s).
- Holdout scores never feed back into any decision (enforce: selection frozen flag set before guard opens; assert no selection call after freeze).

**Tests.** T-TUN-01…04, T-EVL-01…06, T-LEAK-02, T-LEAK-03 (run level), T-RES-03, T-PLN-09; CLI end-to-end on `binary_basic`, `multiclass_basic`, `regression_basic`, `binary_small` (no-holdout path) in `trials` mode.

**Validation.** `make check`; run the CLI on the three task fixtures and paste the leaderboard summary and holdout lines from the real output.

**Self-review checklist.**
- Is every number in `experiment.json` traceable to computed values?
- Does any code path select a model after the holdout opened?
- Does the rule-based plan pass the same schema as an LLM plan?

**GATE A — human check before continuing.**
1. Run the CLI yourself on one fixture (command in the report) and open `experiment.json`.
2. Confirm the four leakage invariants are tested: T-LEAK-01, 02, 07, 08 green.
3. Run **U1 Independent audit** with scope "Milestone 1 (P04–P09)" in a fresh session.
4. Mark Gate A in CONTEXT §5 only after the audit verdict is PASS.

**Done when.** The rule-based AutoML runs end to end from the CLI with correct, honest, reproducible results.

---
---

# MILESTONE 2 — Intelligence, artifacts, and the agent

---

## P10 — LLM provider router

**Role.** You are an LLM Systems Architect who has run multi-provider LLM gateways in production. You treat every provider as unreliable and every response as untrusted, and you never let a retry loop exceed its budget.

**Mission.** Build one provider-independent interface to Groq, Gemini, OpenRouter, and Ollama with retries, backoff, fallback, budgets, caching, response normalization, privacy notices, and zero leakage of secrets or prompt bodies into logs.

**Read first.** PRD §23 (all), §6.4 (LLM cache), §24.4–§24.5, §30.3, FR-LLM-01, FR-LLM-02, §31.3 rows T-LLM-01…04, T-LLM-06, T-LLM-07, T-PRIV-04; `config/providers.yaml`.

**Depends on.** P02, P03.

**Scope.**
- In: `tabpilot.llm` (`LLMRouter`, adapters, error classification, backoff, budget enforcement, cache, normalization, consent model, privacy notice texts).
- Out: prompts and planner logic (P11), narrative (P13).

**Deliverables.**
1. Adapters implementing `LLMProvider` (§23.1) for Groq, OpenRouter, and Ollama through the `openai` client with `base_url`; Gemini through its OpenAI-compatible endpoint by default with the adapter isolated so a switch to the `google-genai` SDK (D-11) touches one file.
2. Error classification mapping client exceptions and HTTP statuses to `RateLimited(retry_after)`, `Timeout`, `ServerError`, `AuthError`, `BadRequest`, `ConnectionError`.
3. `LLMRouter.complete(request, budget)` implementing the §23.5 algorithm exactly, with injectable clock/sleeper.
4. `LLMBudget` enforcement (§23.7): calls (all attempts), input tokens, output tokens, LLM wall time. Exhaustion raises a typed exception that callers convert into the deterministic fallback.
5. Response normalization (§23.6) and token estimation with an `estimated` flag.
6. Cache (§6.4): canonical-JSON SHA-256 key; per-run in-memory on hosted, on-disk with 7-day TTL locally; disabled by a flag (benchmark uses it).
7. Consent and notices: `ProviderConsent` model and `privacy_notice()` text per provider (wording based on §23.2, marked "verify current terms before deployment").
8. `llm_calls.jsonl` record writer (metadata and SHA-256 of request/response only).
9. Optional `live` tests that run only when the relevant API key env var is present.

**Requirements and critical correctness points.**
- **Never** log or persist prompt or response bodies on hosted; never log keys (I-10, I-11).
- Fallback reuses the **identical** request payload (T-LLM-06: same payload hash across providers).
- The budget counts every attempt, including retries and repairs.
- `tabpilot.llm` must not import pandas (import contract).
- Model IDs come only from configuration; no hard-coded model names or quotas.
- Temperature 0 by default; JSON mode only where the adapter declares support.

**Tests.** T-LLM-01…04, T-LLM-06, T-LLM-07; partial T-PRIV-04 (no bodies or keys in logs); cache hit counts as zero calls; budget exhaustion mid-retry; `retry_after` > 20 s moves to next provider; 401 marks provider unavailable for the run.

**Validation.** `make check`; coverage of `llm` ≥ 95%.

**Self-review checklist.**
- Can any retry path exceed the call budget?
- Is there any code path that writes a prompt body to disk or logs on the hosted profile?
- Does each adapter declare its JSON-mode support from config rather than assume it?

**Done when.** The router passes all failure-mode tests with scripted providers and leaks nothing.

---

## P11 — LLM planner and plan validator

**Role.** You are a Senior LLM Systems Engineer and AutoML architect. You write production prompts that produce valid structured output on the first try, and validators that assume the model is wrong until proven right.

**Mission.** Build the planner that turns the compact profile into a validated modeling plan — with an excellent system prompt, one repair attempt, a strict validator with repair rules, cost checks, and a guaranteed fallback.

**Read first.** PRD §12 (all), §9.4, §14.3–§14.4, §24.8, §26.3–§26.4, FR-PLN-01, FR-PLN-02, §31.3 rows T-PLN-01…08, T-TGT-11, T-LLM-05, T-SEC-02 (plan part); the plan fixtures and manifest from P03.

**Depends on.** P06, P07, P09, P10.

**Scope.**
- In: `tabpilot.planning` (input builder, prompt templates, planner call with repair, validator, status, events, `preview_planner_input`).
- Out: reflection prompts (P12), narrative prompts (P13).

**Deliverables.**
1. `build_planner_input(profile_compact, settings, run_config) -> PlannerInput` and `preview_planner_input(...)` returning the exact canonical JSON that will be sent (used by the UI inspector, FR-PRF-03).
2. Versioned prompt templates in `src/tabpilot/planning/prompts/` (`planner_system@1.0`, `planner_repair@1.0`). The system prompt must: define the role precisely; state that data fields are data, not instructions; require JSON only; embed the JSON Schema generated from `PlannerOutput`; state every hard constraint (only listed columns, only available models, only catalog ops, feature count, budget); explain the DSL catalog; and include **one** worked example built from a synthetic fixture (never user data). Keep it under ~3,000 tokens.
3. `make_plan(ctx) -> (PlannerOutput, PlanValidation, plan_source)`: one call, temperature 0, one repair call with field-path errors (≤ 1,500 chars), then fallback (§12.5, §12.9).
4. `validate_plan(plan, ctx)` implementing every rule of §12.6, reusing `features.validate_specs` (P07) and the resource estimator (P04): unknown/excluded/target columns, contradictions (conservative action wins), model availability, DSL validation, canonical dedup, 12-feature cap, target-transform rules, allocation normalization, advisory `cv_folds`, cost simplifications, and the "> 50% unsafe features → invalid" rule.
5. Events: `plan_received`, `plan_validated{status, issues}`, `planner_fallback{reason}`.

**Requirements and critical correctness points.**
- The planner can never change task, target, metric, folds, holdout, budgets, or available models (§12.1). Test each.
- Any string from the profile (column names) appears in the prompt only inside the JSON data payload.
- The validator never calls the LLM and never raises on malformed plans; it returns `invalid` and the caller falls back.
- Prompt templates are versioned files; the version string is recorded in `llm_calls.jsonl` and `ExperimentConfig`.

**Tests.** T-PLN-01…08 across every plan fixture with its expected outcome from `MANIFEST.yaml`; T-TGT-11; T-LLM-05 (all providers down → rule-based plan, run continues); T-SEC-02 plan part (injection column names cannot alter constraints; outcome is a bounded plan or fallback); prompt size test; preview equals the request payload byte-for-byte (T-PRIV-02).

**Validation.** `make check`; coverage of `planning` ≥ 95% (validator).

**Self-review checklist.**
- Would a confident but wrong LLM plan be caught (try three adversarial fixtures)?
- Is the system prompt free of any real user data and under budget?
- Does every failure path end in a valid plan?

**Done when.** Every plan fixture produces its expected outcome, and the planner always yields a valid plan.

---

## P12 — Reflection and ensembling

**Role.** You are a Principal AutoML Architect specializing in iterative search and ensemble methods. You know how easily repeated selection on the same folds produces false improvements, and you design acceptance rules that resist it.

**Mission.** Implement the bounded reflection loop's components and the cross-fitted ensemble, with acceptance rules that only keep real improvements.

**Read first.** PRD §19 (all), §20 (all), §18.2, §26.10, FR-RFL-01, FR-ENS-01, §31.3 rows T-RFL-*, T-ENS-*.

**Depends on.** P07, P09, P10, P11.

**Scope.**
- In: `tabpilot.reflection` (input builder, prompt templates `reflector_system@1.0`/`reflector_repair@1.0`, proposal call with one repair, proposal validation reusing P07/P11 validators, change application producing child candidates with lineage, round budget, paired-fold acceptance, stopping rules); `tabpilot.ensembling` (eligibility, cross-fitted hill climbing, keep/discard, cost limits); `VotingEnsemble` in `tabpilot_runtime`.
- Out: graph wiring (P15).

**Deliverables.**
1. `build_reflection_input(state, leaderboard, importances, provenance, budget) -> ReflectionInput` (gain importance for tree models; no SHAP needed here; no raw rows, predictions, or holdout data).
2. `propose(…) -> ReflectionProposal | None` with validation of permitted change types and bounds (§19.3).
3. `apply_changes(proposal, candidates) -> list[CandidateSpec]` with `parent_entry_id` and `change_set_id`.
4. `accept(parent_result, child_result, settings) -> AcceptanceDecision` — the paired-fold rule of §19.5 as a pure function.
5. `reflection_budget(...)` per §19.4 and `should_stop(...)` per §19.6.
6. `cross_fitted_ensemble(...)` per §20.3, eligibility §20.2, keep/discard §20.4 using the same acceptance function, cost limits §20.5.
7. `VotingEnsemble` estimator (fixed weights, class-order alignment, probability renormalization; sklearn-compatible; picklable from `tabpilot_runtime`).

**Requirements and critical correctness points.**
- Use selection-score deltas so "better" is always positive regardless of metric direction; compute the relative threshold on the metric magnitude as §19.5 states.
- The acceptance function is the single source of truth for both reflection and ensembling.
- Ensemble weights for the reported CV score are learned on other folds only (cross-fitted); final weights use all OOF.
- Deterministic tie-breaking by `entry_id` order.
- The reflector cannot propose anything that changes task, metric, folds, seeds, budgets, or holdout access.

**Tests.** T-RFL-01, T-RFL-03…05, T-ENS-01…04; adversarial proposals (prohibited changes, oversize, invalid DSL); acceptance edge cases (exactly at threshold, majority boundary for K=5 and 15 repeated folds, loss vs score metrics); ensemble with fewer than 2 eligible candidates; `VotingEnsemble` pickle round-trip without `tabpilot` importable.

**Validation.** `make check`; coverage of `reflection` and `ensembling` ≥ 95%.

**Self-review checklist.**
- Can a noisy improvement (better on 2 of 5 folds) be accepted? It must not.
- Is the ensemble's CV score comparable to single models (same folds, cross-fitted)?
- Are all proposal paths bounded by the round budget?

**Done when.** Components pass all tests and are ready to be wired into the graph.

---

## P13 — Explainability and reporting

**Role.** You are a Senior Data Scientist specializing in model interpretability and scientific communication. Your reports are precise, honest about uncertainty, never imply causation, and never contain a number you cannot trace.

**Mission.** Build the explainers, error analysis, figures, facts table, fact-checked narrative, template narrative, and the HTML/Markdown report.

**Read first.** PRD §21 (all), §18.3, §18.5, §24.8 (escaping), FR-EXP-01, FR-RPT-01, §31.3 rows T-EXP-*, T-RPT-*, T-EVL-06, T-SEC-02 (report part).

**Depends on.** P08, P09, P10, P12.

**Scope.**
- In: `tabpilot.explain` (SHAP Tree/Linear explainers, permutation fallback, gain importance, one-hot aggregation to source columns, multiclass aggregation, error analysis); `tabpilot.reporting` (facts table, narrator with prompt templates `narrator_system@1.0`/`narrator_repair@1.0`, fact-checker, template narrative, Jinja2 HTML with autoescape, Markdown, matplotlib figures with the Agg backend).
- Out: report display in UI (P16), bundle packaging (P14).

**Deliverables.**
1. `explain(final_model, dev_sample, task, settings) -> ExplanationResult` with method selection per §21.3 and graceful fallback.
2. `error_analysis(oof, y, dev_frame, task) -> ErrorAnalysis` per §21.4.
3. `build_facts(state) -> FactsTable` covering every number and name the report or narrative may mention.
4. `narrate(facts, router, budget) -> Narrative` and `fact_check(text, facts, event_log) -> FactCheckResult` per §21.7 (number matching with declared rounding, citation of fact IDs, names must exist, banned causal/guarantee phrases, action keywords must match events).
5. `template_narrative(facts) -> Narrative` (deterministic, always valid).
6. `render_report(state, explanations, figures) -> (html, md)` with every section of §21.1, the fixed caveats of §21.6, and CV vs holdout in separate labeled sections.
7. Figures of §21.2 as PNG.

**Requirements and critical correctness points.**
- **Time caps:** SHAP cannot be interrupted safely mid-call. Bound runtime by sample sizes (§21.3) and a pre-estimate from a small timed batch; if the estimate exceeds the cap, use permutation importance instead. Record this design in the Decision Log. The worker-level hard timeout (P15) is the final backstop.
- No number reaches the narrative without a fact ID. The fact-checker is strict; it is fine to fall back to the template often.
- All user-derived strings (column names, display filename) are escaped in HTML and Markdown. Never use `|safe` on them.
- Holdout numbers appear only in the "Independent test" section and summary; never in the CV leaderboard table (T-EVL-06).

**Tests.** T-EXP-01…03 (each model family, multiclass, fallback, timeout path); T-RPT-01…04 (all sections present; invented number caught; causal phrase rejected; uncited number rejected; template fallback); T-EVL-06; T-SEC-02 report part (injection/HTML column names rendered inert); figure files produced; deterministic template output.

**Validation.** `make check`; generate a report from a P09 CLI run directory (write a small script) and confirm it opens; paste section headings from the real output.

**Self-review checklist.**
- Can the narrative contain any number not in the facts table?
- Is there any causal wording in fixed text?
- Is every user-derived string escaped?

**Done when.** Reports are complete, honest, escaped, and fact-checked.

---

## P14 — Artifact export and reproducibility

**Role.** You are a Senior MLOps Engineer specializing in reproducibility and artifact integrity. You ship model bundles that load years later on another machine, and you prove it with tests rather than promises.

**Mission.** Produce the downloadable bundle — serialized model, generated standalone `pipeline.py`, pinned requirements, manifest with checksums, metadata, report, synthetic parity fixture — and verify prediction parity before offering it.

**Read first.** PRD §22 (all), §24.9, §26.12, FR-ART-01, FR-ART-02, §31.3 rows T-ART-*, T-SEC-03; `CONTEXT.md` A-001 (critical).

**Depends on.** P07, P08, P09, P12, P13.

**Scope.**
- In: `tabpilot.export` (bundle assembly, manifest, SHA-256, pinned requirements from installed metadata, synthetic parity fixture, R1 check with rebuild-once, size limits, formula-injection-safe CSV writer, `REPRODUCIBILITY.md`), Jinja2 templates for `pipeline.py`.
- Out: download UI (P16).

**Deliverables.**
1. `export_bundle(state, final_model, report, settings) -> BundleResult` producing the exact layout of §22.2, **plus** a copy of `tabpilot_runtime/` at the bundle root (A-001).
2. `pipeline.py` template: `train`, `predict`, `verify` subcommands (§22.4); imports only `tabpilot_runtime` (bundled) and pinned third-party packages; contains the column map, type rules, feature specs, preprocessing construction, model constructor with final hyperparameters, ensemble wrapper if used, and seeds; includes the joblib security warning in its docstring.
3. `requirements.txt` in the bundle with exact versions read from `importlib.metadata` for every package the model and `pipeline.py` need.
4. Synthetic parity fixture (§22.5) and R1 verification (exact classes; max |Δ| ≤ 1e-9 for probabilities/values).
5. `REPRODUCIBILITY.md` stating R1/R2/R3 levels honestly, commands, environment, and the joblib warning.

**Requirements and critical correctness points.**
- **The decisive test:** in a subprocess run with `python -I`, `PYTHONPATH` set to the unzipped bundle only, and `tabpilot` not importable, `pipeline.py verify` must pass. This proves the bundle is standalone.
- The bundle never contains raw dataset rows (T-ART-06: hash every DEV/HOLD row and scan bundle files).
- `pipeline.py train` must rebuild the same pipeline structure the product used; test R2 on fixtures (probabilities within 1e-6) with deterministic settings.
- `pipeline.py` is generated by templates, never by the LLM; it must pass `ruff`.
- Manifest checksums are computed after all files are final; the UI will display the bundle SHA-256.

**Tests.** T-ART-01…06, T-SEC-03; bundle size limit behavior; parity failure path (inject a mismatch) → rebuild once → `E-EXPORT-PARITY`; ensemble bundle; regression with `log1p`.

**Validation.** `make check`; export a bundle from a P09 CLI run, unzip it into a clean temporary directory, run the isolated `verify` command, and paste the real output. If creating a clean virtual environment needs network access you do not have, record T-ART-03 (clean venv) in Unverified Items and make CI run it.

**Self-review checklist.**
- Does the isolated subprocess test truly hide `tabpilot` (print `sys.path` in the test to prove it)?
- Are there any raw rows anywhere in the bundle?
- Is every reproducibility claim in `REPRODUCIBILITY.md` backed by a test?

**Done when.** A bundle exported from a real run verifies in isolation.

---

## P15 — Orchestration and run runtime  ·  **GATE B**

**Role.** You are an ML Platform Engineer who builds reliable execution runtimes: process isolation, cancellation, timeouts, heartbeats, and cleanup that works even when things crash. You use LangGraph where stateful routing earns its place and nowhere else.

**Mission.** Wire every component into the LangGraph run graph inside a per-run worker process, with the Run Manager, events, cancellation, global timeout, partial completion, recovery, concurrency guard, and data cleanup.

**Read first.** PRD §13 (all), §6 (all), §7 (all), §28 (all), §24.7, §27.4, §29.6, FR-RUN-01…03, §26.13, §31.3 rows T-RUN-*, T-RFL-02, T-E2E-01…03, T-E2E-05, T-LLM-05, T-SEC-06, T-SEC-07, T-PRIV-03.

**Depends on.** P04–P14.

**Scope.**
- In: `tabpilot.orchestration` (state, `RunContext` registry, nodes, conditional edges, recover/fail paths, `MemorySaver`, recursion limit, budget checks before nodes); `tabpilot.runtime` (`RunManager`, worker entrypoint with `spawn` context, event queue, heartbeats, cancel escalation, global and hard timeouts, file-lock slot, idle TTL, sweeper, cleanup lifecycle); the CLI rewired through `RunManager`.
- Out: Streamlit pages (P16).

**Deliverables.**
1. Graph per §13.2 with node contracts of §13.4; pre-graph stages (ingest, split, profile) as plain calls in the worker emitting events.
2. `RunState` kept small; large objects in `RunContext` keyed by id (§13.3).
3. `RunManager` API exactly per §28.2, including idempotency and cancellation semantics.
4. Worker protocol per §28.3 (events, heartbeats every 2 s, result; missing heartbeats for 20 s → `E-WORKER-LOST`).
5. Cancellation (§13.7): cooperative checks between folds, trials, and nodes; SIGTERM after 10 s; SIGKILL after 5 s more; no orphan processes.
6. Global timeout with partial completion (`completed_partial`) when a valid candidate exists; hard kill at 130%.
7. Concurrency guard (FR-RUN-03) with an OS file lock.
8. Cleanup lifecycle (§24.7) and sweeper.
9. CLI: `tabpilot run …` now runs through `RunManager` and the graph; flags `--llm {off,stub,providers}`.

**Requirements and critical correctness points.**
- Events are emitted only after the action they describe has completed (§27.4); the LLM never creates events.
- Reflection loop bounded by `max_rounds` (cap 2) and recursion limit 40 (T-RFL-02 at graph level with an adversarial scripted reflector).
- The UI process never receives raw data after validation; the worker reads from the run directory.
- **Spike first (D-02):** before building everything on processes, verify that a `spawn` child process starts, communicates, and is killable in this environment. If it is not viable, implement the documented thread-mode fallback behind the same `RunManager` interface and record the decision.
- `--llm off` must make zero provider calls (T-PRIV-03 with a network spy).

**Tests.** T-RUN-01…05, T-RFL-02, T-E2E-01…03 (stub LLM and `--llm off`), T-E2E-05 reliability suite (50 runs: fixtures × seeds × stub failure modes; ≥ 95% valid bundles; 100% of failures with E-codes), T-LLM-05, T-SEC-06, T-SEC-07, T-PRIV-03, T-PRIV-04 end-to-end log scan; T-E2E-04 (marked `live`, runs only when a provider key is set; otherwise record in Unverified Items).

**Validation.** `make check`; `make test-all` excluding `perf` and `live`; run the CLI with `--llm stub` and `--llm off` on all three task fixtures and paste the final status lines and bundle paths from the real output.

**Self-review checklist.**
- After cancel, is any process left (check with psutil in the test)?
- After failure, is the run directory gone?
- Does every node respect the budget and the reserved 15% slice?

**GATE B — human check before continuing.**
1. Run `tabpilot run fixtures/datasets/binary_basic.csv --target <target> --llm off` and then with `--llm stub`; open the report HTML; run `pipeline.py verify` in the bundle.
2. Optional: set one provider key locally and run with `--llm providers` to see a real plan (not required by tests).
3. Run **U1** with scope "Milestone 2 (P10–P15)". Mark Gate B after PASS.

**Done when.** The complete agent runs headless end to end with every failure mode handled.

---
---

# MILESTONE 3 — Product, hardening, evidence, release

---

## P16 — Streamlit user interface  ·  **GATE C**

**Role.** You are a Frontend and UX Architect for data products, expert in Streamlit's execution model, accessibility, and designing interfaces that make complex ML workflows feel simple and trustworthy.

**Mission.** Build the five-page application of PRD §27 on top of `RunManager`, implementing the full user journey of §3.2 with honest progress, clear errors, consent, and data deletion.

**Read first.** PRD §27 (all), §3 (all), §5 (all user stories), §8.9, §24.5, §2.4, FR-ING-06, D-14, §31.3 rows T-UI-*.

**Depends on.** P15.

**Scope.**
- In: `tabpilot.ui` pages and components, `streamlit_app.py`, `.streamlit/config.toml` (upload limit, error details off), sample datasets (with license notes — D-05; if a license is unverified, use an OpenML or synthetic alternative and record it), precomputed sample results script (D-14).
- Out: business logic (call `RunManager` and schemas only — import contract).

**Deliverables.**
1. Pages: Upload, Configure, Run, Results (tabs: Summary, Data profile, Plan, Leaderboard, Explainability, Report, Downloads), About & Privacy.
2. Components per §27.2: column table with overrides, target selector, task confirmation (ambiguous requires a click), metric selector, group-column selector, budget slider with live feasibility, provider settings, consent dialog, planner-input inspector (byte-identical JSON), event log (system vs `llm_rationale` styling), leaderboard with fold details, score chart, explainability images, downloads with SHA-256 and joblib warning, "Delete my data now".
3. All states of §27.3, including waiting-for-slot with "View sample result".
4. `scripts/precompute_samples.py` generating labeled precomputed results for the bundled samples.
5. Streamlit `AppTest` tests.

**Requirements and critical correctness points.**
- `st.session_state` stores only `run_id` and UI choices — never DataFrames beyond the 20-row preview.
- Live updates via `st.fragment(run_every=…)` if supported by the pinned Streamlit version; otherwise a controlled rerun loop. Stop polling when the run ends.
- Never use `unsafe_allow_html=True` with any user-derived string.
- Every error displays the E-code message (what, where, what to do) — no stack traces.
- Usable at 360 px width; every input labeled; color never the only signal.

**Tests.** T-UI-01…04 with `AppTest` (upload → configure → run with stub LLM → results; ambiguous task requires choice; LLM rationale labeled; error states render); session state contains no DataFrame after start; inspector equals request payload.

**Validation.** `make check`; start the app locally (`uv run streamlit run streamlit_app.py`) in headless mode and confirm it serves the health endpoint; paste the real test summary.

**Self-review checklist.**
- Can a user finish the §3.2 journey without reading documentation?
- Can any user string render as HTML?
- Does "Delete my data now" really delete the run directory (tested)?

**GATE C — human check before continuing.**
1. Run the app locally and complete the journey with a sample dataset and with one of your own CSVs. Try cancel, a bad file, and "No LLM".
2. Note anything confusing in CONTEXT Known Issues for P20.
3. Run **U1** with scope "P16". Mark Gate C after PASS.

**Done when.** The app delivers the full journey locally with all tests green.

---

## P17 — Security and privacy hardening audit

**Role.** You are a Principal Application Security Engineer and red-teamer. You assume an attacker controls the uploaded file, every column name, and the LLM's output. You find what others missed and fix it with tests.

**Mission.** Attack the system end to end against PRD §24, fix every finding, and leave a written security review with residual risks.

**Read first.** PRD §24 (all), §30.3, §14.7, §22.7, §23.4, §31.3 rows T-SEC-*, T-PRIV-*; all source code.

**Depends on.** P16.

**Scope.**
- In: security and privacy tests, fixes to any module (minimal, each with a regression test), CI security tooling (`pip-audit`, `gitleaks`), `docs/security/REVIEW.md`.
- Out: new features.

**Deliverables.**
1. Threat-model walkthrough of §24.1 against the actual code; add attack fixtures where missing (prompt injection in column names and rationale, HTML/script in names, formula cells, path traversal names, oversized and fake-extension files, malformed Parquet, extremely long strings).
2. Verify and complete T-SEC-01…08 and T-PRIV-01…04; add tests for every gap found.
3. Verify the hosted build excludes `tabpilot.sandbox` (T-SEC-08; FR-FE-03 stays unbuilt in v1 per D-09) and that `tabpilot_runtime` contains no dangerous calls.
4. Verify cleanup on success, failure, cancel, TTL, and crash (sweeper).
5. Add `pip-audit` and `gitleaks` jobs to CI (fail on high/critical with fix available; on any secret).
6. `docs/security/REVIEW.md`: findings table (ID, severity, description, fix, test), residual risks (as listed in §24.11 plus any new ones), and what the hosted demo must tell users.

**Requirements and critical correctness points.**
- Every finding is fixed with a regression test, or recorded as an accepted residual risk with justification. Nothing is silently left.
- Do not weaken any invariant to make an attack test pass.

**Tests.** All T-SEC-*, T-PRIV-*, plus new attack tests.

**Validation.** `make check`; `uv run pip-audit`; `gitleaks detect` (if unavailable in session, record and rely on CI); paste results.

**Self-review checklist.**
- Did I try to make the LLM output reach code execution? Report how.
- Did I try to exfiltrate data through column names, reports, logs, and the bundle?
- Is every residual risk communicated to users?

**Done when.** No open high/critical findings; review document committed.

---

## P18 — Containerization, CI/CD, performance validation

**Role.** You are a Staff DevOps and Performance Engineer. You make deployments boring and measure performance on the real constraint — here, a ~1 GB free host — instead of trusting estimates.

**Mission.** Containerize the app, complete CI/CD, run performance tests under hosted-like limits, calibrate the memory estimator, and prove the runtime target.

**Read first.** PRD §34 (all), §29 (all), §33 NFR-PERF-*, NFR-RES-01, NFR-POR-*, §31.3 rows T-PERF-*; CONTEXT Q-01 / D-01.

**Depends on.** P17.

**Scope.**
- In: `Dockerfile` (§34.3), `.dockerignore`, `.streamlit/config.toml` final, workflows `perf.yml`, `benchmark.yml` (smoke), `release.yml`, perf tests, estimator calibration, `requirements.txt` hosted subset final.
- Out: benchmark harness logic (P19).

**Deliverables.**
1. Docker image per §34.3 (non-root user, `libgomp1`, health check, hosted profile, upload limit).
2. Perf tests T-PERF-01…03 run inside the container with `--memory=1g --cpus=2`: 10K × 30 runtime p90 over 10 runs (or a documented smaller count with justification), `hosted_limit.csv` completes or simplifies without OOM, estimator predictions within ±30% of measured peak RSS on the fixture set.
3. Calibrated estimator coefficients written to `config/profiles.yaml` with the measurement table in `docs/perf/CALIBRATION.md`; `SAFETY` reduced from 1.5 only if the data supports it.
4. Workflows: `perf.yml` (nightly + pre-release), `benchmark.yml` (manual smoke on the development suite), `release.yml` (tag → full CI + perf gate + image build + GitHub release).
5. Image size and install-time measurements (NFR-RES-01).

**Requirements and critical correctness points.**
- If Docker is not available in this session, write and lint everything, run the perf tests without the container where possible (clearly labeled "not hosted-equivalent"), and record T-PERF-* as Unverified with the CI workflow as the verifier. **Never** report container numbers you did not measure.
- Performance numbers go in `docs/perf/` with hardware details and date.
- If the 10-minute target fails, do not hide it: record the measured result, identify the bottleneck from stage timings, and propose fixes as a Known Issue for the owner.

**Tests.** T-PERF-01…03; Docker build test; health-check test.

**Validation.** `docker build`; perf tests inside the container (or recorded as unverified); paste real numbers.

**Self-review checklist.**
- Are all numbers in `docs/perf/` measured, dated, and attributed to hardware?
- Does the image run as non-root with the hosted profile?

**Done when.** The container builds, CI/CD is complete, and performance evidence (or an honest gap) is recorded.

---

## P19 — Benchmark harness

**Role.** You are the QA and ML Evaluation Lead responsible for benchmark methodology. Your benchmarks are fair, reproducible, and immune to "tuning on the test set". You would rather publish a disappointing honest result than an impressive unreproducible one.

**Mission.** Build the benchmark harness comparing TabPilot with FLAML, default LightGBM, and the ablations, with frozen dataset suites, a strict protocol, result files, and reporting — without fabricating any result.

**Read first.** PRD §32 (all), §11.10, §1.6, §1.10, FR-BEN-01, §31.3 rows T-BEN-*; CONTEXT Q-03 / D-05.

**Depends on.** P15 (P18 recommended).

**Scope.**
- In: `benchmark/` (suite definitions, harness, runners B0, B1, A1–A4, A5 optional, aggregation, README table generator), `[bench]` extra usage.
- Out: product code changes (record any needed fix as a Known Issue, fix only if trivial and tested).

**Deliverables.**
1. `benchmark/suite.yaml`: development suite (5) and held-out claim suite (10: 4 binary, 3 multiclass, 3 regression, 1K–100K rows) with dataset source IDs and license fields. Verify IDs and licenses from OpenML metadata if network is available; otherwise list candidates marked `verify` and record in Unverified Items. Freeze the held-out suite with a commit hash field.
2. Harness implementing §32.3: outer split per dataset × seed, identical outer-train data and wall-clock budget for every system, identifier exclusion, failure accounting, hardware capture, LLM caching disabled, temperature 0.
3. Runners: B0 default LightGBM; B1 FLAML with the same budget, metric, CPU count; A1 TabPilot full; A2 rule-based (`--llm off`); A3 LLM plan without LLM features; A4 no reflection.
4. `results.csv` schema (one row per dataset × seed × system) and `benchmark/aggregate.py` producing §32.4 statistics; README table generator reading only `results.csv`.
5. A smoke run on 1–2 development datasets with a tiny budget to prove the pipeline works (results labeled `smoke`, never used as claims).

**Requirements and critical correctness points.**
- **Do not run the held-out suite** in this prompt unless the owner explicitly asks; it is run once, deliberately, by the owner (instructions in `benchmark/README.md`).
- Every number in any README comes from `results.csv` (T-BEN-02).
- TabPilot's own holdout is inside the outer-train split; claims use outer-test scores only.

**Tests.** T-BEN-01 (all systems receive identical data and budget — verify by hashing inputs per system), T-BEN-02 (table regenerated from file), aggregation math on a synthetic results file.

**Validation.** `make check`; smoke run output (real) pasted.

**Self-review checklist.**
- Could any system see the outer test data?
- Is any number in the docs not traceable to `results.csv`?

**Done when.** The harness is ready for the owner to run the held-out suite.

---

## P20 — Release readiness  ·  **GATE D**

**Role.** You are a Principal Engineer acting as Release Manager, together with the Technical Product Manager. You ship only what you can prove, and you write documentation a stranger can follow.

**Mission.** Verify every release gate with evidence, finish documentation, prepare deployment instructions for the owner, and leave the project in a clean, handed-off state.

**Read first.** PRD §38, §40, §35, §36, §37, §33, §2.4–§2.5; `CONTEXT.md` (everything, especially Known Issues and Unverified Items); `docs/TRACEABILITY.md`.

**Depends on.** P01–P19.

**Scope.**
- In: verification, documentation, small fixes needed to pass gates (each with a test), release notes.
- Out: new features.

**Deliverables.**
1. `docs/RELEASE_CHECKLIST.md`: every gate G-01…G-18 with verification method, evidence (test names, command outputs, file links), result (PASS / FAIL / WAIVED-by-owner), and date.
2. `docs/TRACEABILITY.md` updated: every FR, T, NFR, and G marked with implementation status and evidence.
3. `README.md`: what TabPilot is, quickstart (local, Docker, Ollama), architecture diagram (Mermaid), privacy summary (what is sent where), limitations, reproducibility levels, benchmark section that renders only from `results.csv` (or states that the held-out benchmark has not been run yet).
4. `docs/DEPLOY.md` — human steps: create the Streamlit Community Cloud app from the repo; set Python 3.11; add secrets; choose and verify provider model IDs (D-03); run the D-01 probe and record resources; verify dataset licenses (D-05); post-deploy smoke test (each sample); rollback.
5. `CHANGELOG.md` and version `0.1.0` prepared (tagging is the owner's decision).
6. Final `CONTEXT.md` handoff: ledger, gates, all open items with owners.

**Requirements and critical correctness points.**
- A gate without evidence is FAIL, not PASS. Unverified items stay listed until verified.
- No claims (performance, benchmark, provider behavior) without a linked measurement.
- Do not tag or deploy — prepare everything for the owner.

**Tests.** `make test-all` (excluding `live`, and `perf` if no Docker — note it); every gate's tests.

**Validation.** Paste the full suite summary and the gate table.

**GATE D — owner actions.**
1. Review `docs/RELEASE_CHECKLIST.md`; decide on any waivers.
2. Follow `docs/DEPLOY.md` to deploy; run the smoke test.
3. Run the held-out benchmark (`benchmark/README.md`) when ready; regenerate the README table.
4. Run **U1** with scope "Full release" before tagging `v0.1.0`.

**Done when.** Every gate is PASS or explicitly waived by the owner, and documentation lets a stranger deploy and use TabPilot.

---
---

# UTILITY PROMPTS

---

## U1 — Independent audit  *(run in a fresh session after each build prompt)*

```
You are a Principal Software Engineer and ML auditor performing an independent
review. You did not write this code. Trust nothing in reports or commit
messages: verify everything by reading code and running commands.

Scope of this audit: <P-ID, milestone, or "Full release">

1. Read CLAUDE.md, CONTEXT.md, the prompt(s) in docs/PROMPTS.md for this scope,
   and every PRD section those prompts list.
2. Run `make check` (and `make test-all` without perf/live for milestone or
   release scope). Record the real output.
3. Build a conformance table: for every requirement, FR, and test ID the scope
   owns → verdict (MET / PARTIAL / MISSING / WRONG) → evidence (file:line,
   test name). Read the test bodies: a test that cannot fail, or that mocks the
   unit under test, does not count.
4. Check the invariants in CLAUDE.md §3 that the scope touches. Attempt at
   least one concrete violation per invariant (e.g. access holdout early, send
   a category label to the LLM, make a spec string execute) and confirm the
   code or tests stop it.
5. Grep for TODO/FIXME/NotImplementedError/placeholder logic, skipped or
   xfailed tests without a Known Issue ID, hard-coded secrets, model IDs, or
   quotas, and dependencies added without a Decision Log entry.
6. Check scope discipline: changes outside the prompt's scope must be minimal,
   tested, and logged.
7. Check that CONTEXT.md is accurate: ledger, module status, decisions, known
   issues, unverified items, change log, handoff note.

Do not fix anything except typos in documentation. Write findings to
CONTEXT.md §16 Audit log (and §13 Known Issues for each finding, with
severity: critical / major / minor) and commit as "U1: audit <scope>".

Finish with:
## Audit — <scope>
Verdict: PASS | PASS_WITH_FIXES (minor only) | FAIL
Conformance: <n met / n total>, table attached
Findings: <ID, severity, description, evidence, required fix>
Invariant checks: <what you tried, result>
```

---

## U2 — Course correction

```
You are the original domain expert for <P-ID> (see its Role in
docs/PROMPTS.md), now fixing problems found in review.

Read CLAUDE.md and CONTEXT.md. Read the findings to fix:
<paste audit findings, or describe what you observed>.

For each finding:
1. Reproduce it with a failing test first (or explain why a test is
   impossible and how you verified instead).
2. Apply the smallest correct fix within <P-ID>'s scope. If the right fix
   requires changing the PRD's behavior, stop and follow the Deviation Protocol.
3. Make the test pass; run the full suite.

Do not change unrelated code. Do not weaken tests. Mark each finding
resolved in CONTEXT.md §13 with the commit, update the ledger status, add a
Change Log entry, and commit as "U2: fix <finding IDs>".
Finish with the standard Report Format, listing each finding → fix → test.
```

---

## U3 — Session resume / AI handoff

```
You are taking over the TabPilot project mid-flight. Your first job is to
understand the true state before changing anything.

1. Read CLAUDE.md, CONTEXT.md (all), docs/TRACEABILITY.md, and the ledger's
   current/next prompt in docs/PROMPTS.md.
2. Run `git status`, `git log --oneline -20`, and `make check`.
3. Compare reality with CONTEXT.md: Are modules marked complete really
   present and tested? Is any prompt marked in_progress? Are there uncommitted
   changes? Does the suite pass?
4. If a prompt was interrupted (status in_progress or uncommitted work):
   read its prompt, determine what is done and what is missing by reading code
   and tests, and write a short recovery plan.
5. Update CONTEXT.md §3 Current State and §18 Handoff Note to match reality,
   recording any discrepancies in §13 Known Issues. Commit as "U3: resume".

Report: the true state, discrepancies found, and the exact next action
(either "continue <P-ID> from step …" or "start <P-ID>"). Then stop and wait.
```

---

## U4 — Change request

```
You are the Technical Product Manager and Chief Architect for TabPilot.
The owner requests this change: <describe the change>.

1. Read CLAUDE.md, CONTEXT.md, and every PRD section the change touches.
2. Impact analysis: affected requirements (FR/NFR/T/G IDs), invariants,
   modules, completed prompts that need rework, future prompts that change,
   risks (especially leakage, privacy, security, resource limits), and effort.
3. Propose the change precisely as a PRD Amendment (what changes, exactly
   which items it overrides) and, if needed, new or modified test IDs.
4. Write an ADR in docs/adr/. Add the amendment to CONTEXT.md §10 with status
   "proposed", and the decision to §9 with status "proposed".
5. If prompts in docs/PROMPTS.md must change, write the exact replacement
   text in the ADR (do not edit PROMPTS.md until the owner accepts).

Do not implement the change. Commit as "U4: change request <short name>".
Report the impact table and your recommendation (accept / modify / reject,
with reasons). After the owner accepts, mark the amendment "accepted", apply
the prompt edits, and schedule rework via the ledger.
```
