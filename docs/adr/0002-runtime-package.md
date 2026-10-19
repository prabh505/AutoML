# ADR-0002: `tabpilot_runtime` package for every class pickled into `model.joblib`

| Field | Value |
| --- | --- |
| Status | accepted |
| Date | 2026-10-19 |
| Prompt | P00 (recording the kit decision) |
| Decision Log | DEC-003 |
| PRD refs | §15.1, §20.5, §22.2, §22.4, §22.6, §25.1, §25.4, FR-ART-01, FR-ART-02, T-ART-02, T-ART-03, T-SEC-01 |
| Amendment | A-001 |

## Context

Pickle (and so joblib) stores the module path of every object's class. The exported `model/model.joblib` holds the whole fitted pipeline (§22.2). That includes custom classes: `ColumnNameMapper`, `TypeCaster`, `BaselineFeatureTransformer`, `DSLFeatureTransformer` (§15.1), the early-stopping wrapper (§16.3) and `VotingEnsemble` (§20.5).

If those classes live under `tabpilot.*`, loading the model needs the whole `tabpilot` package and its dependencies (Streamlit, LangGraph, the LLM client, and so on). That contradicts:

- FR-ART-02: "Depends only on pinned third-party packages, not on the `tabpilot` package."
- §22.6 R1: prediction parity with the pinned `requirements.txt`.

The PRD's own workaround (§22.4: "DSL op implementations … copied from the library's reference implementation") covers `pipeline.py` source code. It does not cover unpickling `model.joblib`, which needs the classes at their original module paths.

## Decision

Add the package `src/tabpilot_runtime/` (amendment A-001):

1. It contains every custom class that can appear inside a fitted pipeline: `ColumnNameMapper`, `TypeCaster`, `BaselineFeatureTransformer`, `DSLFeatureTransformer` and the DSL op implementations, `ColumnSelector`, frequency and rare-category encoders, the early-stopping estimator wrapper, and `VotingEnsemble`.
2. It imports only numpy, pandas, scikit-learn and the model libraries. It never imports `tabpilot`. This is enforced by an import-linter contract (P01).
3. `tabpilot` imports these classes from `tabpilot_runtime`. There is one implementation, and no copies.
4. The export bundle ships a copy of `tabpilot_runtime/` at the bundle root next to `pipeline.py` (P14). `pipeline.py` imports from it instead of embedding copied op code.
5. "Standalone `pipeline.py`" therefore means "standalone bundle". The bundle's `model.joblib` loads, and `pipeline.py verify` passes, with only the pinned third-party packages installed and `tabpilot` not importable. This is tested in an isolated subprocess (`python -I`, `PYTHONPATH` = unzipped bundle) in P14.

## Alternatives

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Keep classes in `tabpilot`; require `pip install tabpilot` to load models | Simplest | Breaks FR-ART-02 and R1 as written; bundle pulls UI/LLM deps | Violates the spec |
| Only sklearn built-ins in pipelines (no custom classes) | No custom pickle paths | DSL ops, type casting and the ensemble wrapper can't be expressed with built-ins alone (`FunctionTransformer` with lambdas doesn't pickle reliably) | Not feasible |
| Copy class source into `pipeline.py` and pickle with `__main__` paths | Single file | Model only loads when run from that script; fragile; two implementations to keep equivalent | Fragile and duplicative |
| `skops` serialization | Safer loading | P1 item; still needs the classes importable | Orthogonal; can come later |

## Consequences

- Invariants: adds I-14 (`CLAUDE.md` §3). No other invariant changes.
- The PRD text read as amended (also PI-18 in `CONTEXT.md` §11):
  - §25.1: the tree adds `src/tabpilot_runtime/`.
  - §25.4: the contracts add "`tabpilot_runtime` must not import `tabpilot`".
  - §22.2: the bundle contents add `tabpilot_runtime/`.
  - §22.4: "Imports only packages listed in `requirements.txt`" reads as "… plus the bundled `tabpilot_runtime`". Copied op code is replaced by importing the bundled package, so the "unit-tested for equivalence" requirement is met by construction.
- Tests:
  - T-ART-01's file list includes `tabpilot_runtime/`.
  - T-ART-02 and T-ART-03 run with `tabpilot` hidden.
  - T-SEC-01's AST scan covers `src/tabpilot_runtime/`.
  - P07, P08 and P12 add pickle round-trip tests in a subprocess without `tabpilot`.
- Prompts: P01 (package + contract), P07, P08, P12 (classes live here), and P14 (bundling + isolated verify) already reflect this.
- Risk removed: exported models that silently need the full app to load. Risk added: drift if someone defines a pipeline class under `tabpilot`. The import contract and the isolated-load tests catch that.

## Status

Accepted (project kit, DEC-003 / A-001). Recorded as an ADR by P00.
