"""`GuardedFrame`: a DataFrame that raises when any holdout row is touched (T-LEAK-02).

`guard(df, holdout_index)` returns the full frame (DEV + HOLD) as a `GuardedFrame`.
Any derived frame or column that still contains a holdout row raises `HoldoutAccessError`
when pandas constructs it. This covers row selection, column access, copies, and whole-frame
array conversion. Selecting DEV rows only (e.g. `gf.loc[dev_index]`) works and stays guarded.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


class HoldoutAccessError(AssertionError):
    """A holdout row was accessed before selection was frozen."""


def _check(index: pd.Index, holdout: frozenset[Any]) -> None:
    if holdout and not holdout.isdisjoint(index):
        raise HoldoutAccessError(f"holdout rows accessed: {len(holdout.intersection(index))}")


class GuardedSeries(pd.Series):  # type: ignore[misc]
    _metadata = ["_holdout"]

    @property
    def _constructor(self) -> Any:
        holdout = getattr(self, "_holdout", frozenset())

        def build(*args: Any, **kwargs: Any) -> GuardedSeries:
            s = GuardedSeries(*args, **kwargs)
            s._holdout = holdout
            _check(s.index, holdout)
            return s

        return build

    @property
    def _constructor_expanddim(self) -> Any:
        return GuardedFrame


class GuardedFrame(pd.DataFrame):  # type: ignore[misc]
    _metadata = ["_holdout", "_is_root"]

    @property
    def _constructor(self) -> Any:
        holdout = getattr(self, "_holdout", frozenset())

        def build(*args: Any, **kwargs: Any) -> GuardedFrame:
            f = GuardedFrame(*args, **kwargs)
            f._holdout, f._is_root = holdout, False
            _check(f.index, holdout)
            return f

        return build

    @property
    def _constructor_sliced(self) -> Any:
        holdout = getattr(self, "_holdout", frozenset())

        def build(*args: Any, **kwargs: Any) -> GuardedSeries:
            s = GuardedSeries(*args, **kwargs)
            s._holdout = holdout
            _check(s.index, holdout)
            return s

        return build

    def to_numpy(self, *args: Any, **kwargs: Any) -> np.ndarray:
        _check(self.index, getattr(self, "_holdout", frozenset()))
        return super().to_numpy(*args, **kwargs)

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        _check(self.index, getattr(self, "_holdout", frozenset()))
        return super().__array__(*args, **kwargs)

    @property
    def values(self) -> np.ndarray:
        _check(self.index, getattr(self, "_holdout", frozenset()))
        return super().values


def guard(df: pd.DataFrame, holdout_index: Iterable[Any]) -> GuardedFrame:
    """Wrap `df` (DEV + HOLD) so that touching any holdout row raises."""
    gf = GuardedFrame(df.copy())
    gf._holdout = frozenset(holdout_index)
    gf._is_root = True
    return gf
