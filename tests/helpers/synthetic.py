"""Synthetic frames covering every pipeline branch (P08 tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabpilot.preprocessing import PipelineSchema
from tabpilot.schemas import SemanticType


def synthetic(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, PipelineSchema]:
    """Numeric with missing, low/mid/high-card categoricals, boolean tokens, datetime."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    x[rng.random(n) < 0.1] = np.nan
    frame = pd.DataFrame({
        "Amount ($)": x,
        "z": rng.normal(5, 2, n),
        "low": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        "mid": pd.Categorical(rng.choice([f"m{i}" for i in range(20)], n)),
        "high": pd.Categorical(rng.choice([f"h{i}" for i in range(60)], n)),
        "flag": pd.Categorical(rng.choice(["Yes", "No"], n)),
        "when": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 400, n), unit="D"),
    })  # fmt: skip
    frame.loc[rng.random(n) < 0.05, "low"] = None
    y = (np.nan_to_num(x) + (frame["low"] == "a") + rng.normal(0, 1, n) > 0.5).astype(np.int64)
    schema = PipelineSchema(
        columns={"amount": SemanticType.NUMERIC, "z": SemanticType.NUMERIC,
                 "low": SemanticType.CATEGORICAL, "mid": SemanticType.CATEGORICAL,
                 "high": SemanticType.CATEGORICAL, "flag": SemanticType.BOOLEAN,
                 "when": SemanticType.DATETIME},
        column_map={"Amount ($)": "amount"},
        distinct={"amount": n, "z": n, "low": 3, "mid": 20, "high": 60, "flag": 2, "when": 300},
        missing_rate={"amount": 0.1}, n_rows=n,
    )  # fmt: skip
    return frame.rename(columns={"Amount ($)": "amount"}), y, schema
