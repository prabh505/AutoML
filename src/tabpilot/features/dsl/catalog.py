"""Feature DSL catalog: signatures the validator enforces and the planner sees (PRD §14.3)."""

from __future__ import annotations

from dataclasses import dataclass

from tabpilot.schemas import SemanticType
from tabpilot.schemas.plan import DslCatalogEntry

NUMERIC = frozenset({SemanticType.NUMERIC})
NUMERIC_OR_BOOL = frozenset({SemanticType.NUMERIC, SemanticType.BOOLEAN})
CATEGORICAL = frozenset({SemanticType.CATEGORICAL, SemanticType.BOOLEAN})
DATETIME = frozenset({SemanticType.DATETIME})
ANY = frozenset(
    {SemanticType.NUMERIC, SemanticType.CATEGORICAL, SemanticType.BOOLEAN, SemanticType.DATETIME}
)


@dataclass(frozen=True)
class OpSignature:
    """Input types per position (the last entry repeats for variadic ops) and the output type."""

    inputs: tuple[frozenset[SemanticType], ...]
    output: SemanticType
    variadic: bool = False
    commutative: bool = False
    params: tuple[tuple[str, float, float], ...] = ()


CATALOG: dict[str, OpSignature] = {
    "add": OpSignature((NUMERIC, NUMERIC), SemanticType.NUMERIC, commutative=True),
    "subtract": OpSignature((NUMERIC, NUMERIC), SemanticType.NUMERIC),
    "multiply": OpSignature((NUMERIC, NUMERIC), SemanticType.NUMERIC, commutative=True),
    "ratio": OpSignature((NUMERIC, NUMERIC), SemanticType.NUMERIC, params=(("eps", 1e-9, 1.0),)),
    "log1p": OpSignature((NUMERIC,), SemanticType.NUMERIC),
    "sqrt": OpSignature((NUMERIC,), SemanticType.NUMERIC),
    "square": OpSignature((NUMERIC,), SemanticType.NUMERIC),
    "clip": OpSignature(
        (NUMERIC,), SemanticType.NUMERIC, params=(("lower_q", 0.0, 0.2), ("upper_q", 0.8, 1.0))
    ),
    "bin_quantile": OpSignature((NUMERIC,), SemanticType.CATEGORICAL, params=(("n_bins", 2, 20),)),
    "is_missing": OpSignature((ANY,), SemanticType.BOOLEAN),
    "equals": OpSignature((NUMERIC_OR_BOOL,), SemanticType.BOOLEAN),
    "count_missing": OpSignature((ANY,), SemanticType.NUMERIC, variadic=True, commutative=True),
    "row_sum": OpSignature((NUMERIC,), SemanticType.NUMERIC, variadic=True, commutative=True),
    "row_mean": OpSignature((NUMERIC,), SemanticType.NUMERIC, variadic=True, commutative=True),
    "concat_categories": OpSignature(
        (CATEGORICAL,), SemanticType.CATEGORICAL, variadic=True, commutative=True
    ),
    "group_mean": OpSignature((CATEGORICAL, NUMERIC), SemanticType.NUMERIC),
    "group_std": OpSignature((CATEGORICAL, NUMERIC), SemanticType.NUMERIC),
    "group_count": OpSignature((CATEGORICAL,), SemanticType.NUMERIC),
    "group_deviation": OpSignature((CATEGORICAL, NUMERIC), SemanticType.NUMERIC),
    "date_diff_days": OpSignature((DATETIME, DATETIME), SemanticType.NUMERIC),
}


def _type_names(types: frozenset[SemanticType]) -> str:
    return "|".join(sorted(t.value for t in types))


def catalog_for_planner() -> list[DslCatalogEntry]:
    """The `dsl_catalog` field of `PlannerInput` (§12.2, §26.3)."""
    entries = []
    for op, sig in CATALOG.items():
        inputs = [_type_names(types) for types in sig.inputs]
        if sig.variadic:
            inputs = [f"{inputs[0]} (2-{'3' if op == 'concat_categories' else '20'} inputs)"]
        params = {name: [float(lo), float(hi)] for name, lo, hi in sig.params}
        entries.append(DslCatalogEntry(op=op, inputs=inputs, params=params))
    return entries
