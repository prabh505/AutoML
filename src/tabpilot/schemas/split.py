"""Split specification (`split_spec/1.0`): holdout and fold assignment (PRD §11, FR-SPL-01)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from tabpilot.schemas._base import ColumnName, Contract, NonNegInt, Sha256

SplitStrategy = Literal[
    "stratified_kfold",
    "kfold",
    "stratified_group_kfold",
    "group_kfold",
    "repeated_stratified_kfold",
    "repeated_kfold",
]


class SplitSpec(Contract):
    """Holdout indices plus one fixed fold assignment per repeat (PRD §11.1–§11.6, A-003).

    `fold_assignment[r][i]` is the validation fold of DEV row `dev_indices[i]` in repeat `r`.
    """

    schema_version: Literal["split_spec/1.0"] = "split_spec/1.0"
    split_id: Sha256
    strategy: SplitStrategy
    k: int = Field(ge=2, le=5)
    n_repeats: Literal[1, 3] = 1
    seed: NonNegInt
    n_valid: NonNegInt
    holdout_present: bool
    holdout_fraction: float = Field(ge=0.0, le=0.5)
    holdout_indices: list[NonNegInt] = Field(default_factory=list)
    dev_indices: list[NonNegInt]
    fold_assignment: list[list[NonNegInt]]
    group_column: ColumnName | None = None
    duplicate_grouping: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Partitions are disjoint and complete; folds cover every DEV row (L-1, L-3)."""
        hold, dev = set(self.holdout_indices), set(self.dev_indices)
        if len(hold) != len(self.holdout_indices) or len(dev) != len(self.dev_indices):
            raise ValueError("indices must be unique")
        if hold & dev:
            raise ValueError("holdout and DEV indices overlap")
        if len(hold) + len(dev) != self.n_valid:
            raise ValueError("holdout + DEV must cover all valid rows")
        if self.holdout_present != bool(self.holdout_indices):
            raise ValueError("holdout_present must match holdout_indices")
        if not self.holdout_present and self.holdout_fraction != 0.0:
            raise ValueError("holdout_fraction must be 0 without a holdout")
        if len(self.fold_assignment) != self.n_repeats:
            raise ValueError("fold_assignment needs one row per repeat")
        for row in self.fold_assignment:
            if len(row) != len(self.dev_indices):
                raise ValueError("each fold_assignment row must cover every DEV row")
            if set(row) != set(range(self.k)):
                raise ValueError("fold ids must be exactly 0..k-1")
        if self.strategy.startswith("repeated") != (self.n_repeats == 3):
            raise ValueError("repeated strategies use 3 repeats; others use 1")
        if "group" in self.strategy and not (self.group_column or self.duplicate_grouping):
            raise ValueError("group strategies require a group column or duplicate grouping")
        return self
