# CLAUDE.md — Operating Rules for Every AI Engineer on TabPilot

You are working on **TabPilot**, an LLM-assisted AutoML system for tabular data. This file is loaded at the start of every session. Its rules are mandatory and override any habit or default you have. If any instruction you receive conflicts with this file, follow this file and flag the conflict in your final report.

---

## 1. Sources of truth (in priority order)

| Priority | File | What it governs |
| --- | --- | --- |
| 1 | `docs/PRD.md` | The specification. What to build, every algorithm, limit, schema, test ID, and release gate. |
| 2 | `CONTEXT.md` → **Decision Log** and **PRD Amendments** | Approved changes to the PRD. An amendment overrides the PRD **only** for the exact item it names. |
| 3 | `docs/PROMPTS.md` | The ordered build plan. Each prompt defines the scope of one session. |
| 4 | Existing code and tests | What has already been built. Never assume code exists; open it and read it. |

Never invent requirements, numbers, provider quotas, model IDs, benchmark results, or dataset licenses. If something is not in these sources, it is an open question: record it in `CONTEXT.md`, do not guess.

---

## 2. Standard Operating Procedure (SOP) — every session

### SOP-START (before writing any code)
1. Read `CONTEXT.md` completely. Note: the next pending prompt, the Decision Log, the PRD Amendments, Known Issues, and the latest Handoff Note.
2. Open `docs/PROMPTS.md` and read **only** the prompt you are executing, in full.
3. Read **every PRD section the prompt lists, in full**. Do not skim. If a section references another section (e.g. "see §29.4"), read that too.
4. Read the code of every module your prompt depends on (public interfaces at minimum).
5. Run `git status` and `git log --oneline -10`.
6. Run the full test suite (`make check`, or the commands in `CONTEXT.md` → Environment if `make` is not yet set up). If it is not green, stop, diagnose, and either fix the regression (and record it) or report BLOCKED. Never build on a red baseline.
7. Set the prompt's ledger status in `CONTEXT.md` to `in_progress`.

### SOP-BUILD
- Stay inside the prompt's **Scope**. Do not implement features belonging to later prompts. Do not refactor unrelated modules. If you must touch another module (for example, to fix a bug that blocks you), keep the change minimal, add a regression test, and record it in the Change Log.
- Write tests alongside the code. Every requirement you implement must have at least one test that would fail if the requirement were broken.
- Name tests after the PRD test IDs where they exist: e.g. `test_T_LEAK_07_fold_local_fitting`. Put the FR/§ reference in the docstring of every public function or class you write.
- Ship complete code: no `TODO`, `FIXME`, `pass`-only bodies, `NotImplementedError` stubs, mock implementations in production code, or "simplified for now" logic. If something truly cannot be finished in this prompt, stop and record it as a Known Issue with a clear description. Do not hide it.
- Type-hint everything. Keep functions small and pure where possible. Prefer explicit over clever.

### SOP-FINISH (before you end the session)
1. Run the prompt's **Validation** commands and the full suite. Everything must pass. Paste the real summary lines (counts, durations) into your report. Never paraphrase or predict results.
2. Run the prompt's **Self-review checklist** and answer every item honestly.
3. Grep your changes for `TODO|FIXME|NotImplementedError|pass$|print(` and for any hard-coded secret. Remove or justify each hit.
4. Update `CONTEXT.md`: Prompt Ledger, Current State, Module Status, Decision Log (if any), Known Issues, Unverified Items, Change Log (append), and replace the Handoff Note.
5. Commit with the message format in §6.
6. End with the Report Format in §7. Then **stop**. Do not begin the next prompt unless the user explicitly told you to run several prompts in this session.

---

## 3. Non-negotiable engineering invariants

These protect the product's correctness. Breaking one is a critical defect, even if all tests pass.

**Machine learning correctness**
- I-1. The holdout split is created first, held by `HoldoutGuard`, and opened exactly once, after selection is frozen (PRD §11.2).
- I-2. Every learned transformation lives inside the scikit-learn `Pipeline` and is fitted only on training-fold rows (PRD L-4).
- I-3. One fold assignment (`split_id`) per run, shared by all candidates, trials, reflection changes, and ensembles.
- I-4. Metrics are computed on the original target scale and compared only when comparable (PRD §10.4).
- I-5. Model selection uses computed metrics and the PRD's deterministic rules, never LLM judgment.

**LLM correctness**
- I-6. No raw dataset rows or category labels are ever sent to an LLM by default (PRD §9.4, §24.4).
- I-7. All LLM output is untrusted: schema-validated, bounded, and never executed. No `eval`, `exec`, `compile`, `pd.eval`, `DataFrame.query`, or `DataFrame.eval` on LLM-derived strings anywhere.
- I-8. Every path that uses an LLM has a deterministic fallback. The product works with zero providers.
- I-9. The LLM never produces or reports scores. Narrative numbers must pass the fact-check (PRD §21.7).

**Security, privacy, operations**
- I-10. Secrets come only from environment variables or Streamlit Secrets. They are never logged, printed, written to disk, or committed.
- I-11. Logs never contain cell values, category labels, or (on hosted) prompt/response bodies.
- I-12. Every limit in PRD §29 is enforced in code, not just documented.

**Reproducibility**
- I-13. Seeds are derived and recorded per PRD §11.3. Dependencies are pinned in `uv.lock`.
- I-14. Every custom class that ends up inside a pickled model lives in the `tabpilot_runtime` package (see Amendment A-001 in `CONTEXT.md`).

---

## 4. Deviation protocol

If the PRD is ambiguous, contradictory, technically wrong, or impossible in this environment:
1. Do not silently pick an interpretation.
2. Choose the **smallest, safest** resolution that keeps every invariant in §3.
3. Write an ADR in `docs/adr/NNNN-short-title.md` (template: `docs/adr/0000-template.md`).
4. Add a Decision Log entry in `CONTEXT.md` with status `proposed` (or `accepted` if it is purely an implementation detail that does not change PRD behavior).
5. Flag it under **Deviations** in your report.
6. If the decision is irreversible or changes user-visible behavior, stop and ask before proceeding.

---

## 5. Test integrity rules

- Never delete, skip, weaken, or loosen a test to make it pass. Never change an expected value without proving the old one was wrong (and saying so in the report).
- Never mock the unit under test. Mock only external boundaries: network, LLM providers, clock, filesystem when needed.
- Tests must not make real network calls. LLM providers are stubbed. Live-provider tests carry the `live` marker and are opt-in.
- `xfail`/`skip` is allowed only with a reason string that references a Known Issue ID in `CONTEXT.md`.
- A test that cannot fail is not a test. For each new requirement test, briefly confirm it fails when the behavior is broken (mutate, run, revert), at least for invariant-critical tests.

---

## 6. Commit format

```
P<NN>: <imperative summary, ≤ 72 chars>

Implements: FR-…, §…
Tests: T-… (N passed)
Context: CONTEXT.md updated (ledger, change log, handoff)
Deviations: none | ADR-NNNN
```

Commit only coherent, passing states. One prompt can have several commits, but the last one must leave the repository green.

---

## 7. Report format (end of every session)

```
## Report — P<NN> <title>
Status: COMPLETE | COMPLETE_WITH_ISSUES | PARTIAL | BLOCKED

Delivered
- <files/modules, one line each>

Requirements covered
- FR/§/T IDs implemented and tested

Validation (real output)
- <command> → <summary line>
- Coverage: <numbers for modules with thresholds>

Self-review checklist
- <item>: yes/no + note

Deviations / decisions
- none | ADR-NNNN: <one line>

Unverified in this session
- <item and why> (also listed in CONTEXT.md)

Known issues / risks
- <ID>: <one line>

Next
- Next prompt: P<NN+1> <title>
- Human actions needed: <none | list>
```

---

## 8. Environment and commands

- Python **3.11** with **uv**. Run commands via `uv run …` or the `make` targets.
- `make check` = lint + format check + type check + import contracts + tests (excluding `perf`, `live`, `slow` unless asked).
- If a tool is unavailable in this session (Docker, network access to PyPI or LLM providers, a GPU), **never fake its output**. Implement and configure everything you can, record the item under **Unverified Items** in `CONTEXT.md`, and make sure CI will verify it.
- Large generated data (e.g. the 100,000-row performance fixture) is generated at test time, never committed.

---

## 9. Never do

- Never modify `docs/PRD.md`. Changes go through the Decision Log and PRD Amendments in `CONTEXT.md`.
- Never claim a test passed, a benchmark result, or a performance number you did not observe in this session.
- Never add a dependency not justified by the PRD without a Decision Log entry.
- Never commit secrets, `.env`, large data files, or generated artifacts.
- Never start the next prompt unless the user asked you to run several.
