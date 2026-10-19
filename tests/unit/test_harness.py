"""P03 harness self-tests: spies, guards, scripted provider, network block, clock, helpers."""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tests.helpers.clock import FakeClock
from tests.helpers.errors import raises_error
from tests.helpers.fit_spy import FitSpy, fitted_rows
from tests.helpers.guarded_frame import HoldoutAccessError, guard
from tests.helpers.llm_stub import (
    ScriptedHTTPError,
    ScriptedLLMProvider,
    ScriptedRateLimited,
    ScriptedTimeout,
    ScriptExhausted,
)

from tabpilot.schemas import ErrorCode, LLMRequest, TabPilotError


def frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=range(100, 100 + n)
    )


# ---------------------------------------------------------------- FitSpy
@pytest.mark.unit
def test_fitspy_records_training_fold_rows_inside_pipeline_and_clone() -> None:
    df = frame()
    y = pd.Series(np.arange(len(df)) % 2, index=df.index)
    pipe = Pipeline(
        [("imp", FitSpy(SimpleImputer(), "imp")), ("sc", FitSpy(StandardScaler(), "sc"))]
    )
    pipe.set_output(transform="pandas")
    folds = list(KFold(4, shuffle=True, random_state=0).split(df))
    for tr, _ in folds:
        clone(pipe).fit(df.iloc[tr], y.iloc[tr])
    for key in ("imp", "sc"):
        recorded = fitted_rows(key)
        assert len(recorded) == 4
        for rows, (tr, va) in zip(recorded, folds, strict=True):
            assert rows == set(df.index[tr])
            assert rows.isdisjoint(set(df.index[va]))


@pytest.mark.unit
def test_fitspy_detects_a_leaky_fit() -> None:
    """A step fitted on all rows is visible as a superset of the training fold."""
    df = frame()
    tr = df.index[:15]
    FitSpy(SimpleImputer(), "leaky").fit(df)  # bug under test: fitted on everything
    assert not fitted_rows("leaky")[0] <= set(tr)


@pytest.mark.unit
def test_fitspy_transparent_outputs_and_requires_pandas() -> None:
    df = frame()
    spy = FitSpy(StandardScaler(), "t").set_output(transform="pandas")
    out = spy.fit_transform(df)
    np.testing.assert_allclose(out.to_numpy(), StandardScaler().fit_transform(df))
    with pytest.raises(TypeError, match="pandas"):
        FitSpy(StandardScaler(), "t").fit(df.to_numpy())


# ---------------------------------------------------------------- GuardedFrame
@pytest.mark.unit
def test_guarded_frame_blocks_every_holdout_access_path() -> None:
    df = frame()
    hold = list(df.index[-4:])
    dev = list(df.index[:-4])
    gf = guard(df, hold)
    for access in (
        lambda: gf.loc[hold],
        lambda: gf.loc[[dev[0], hold[0]]],
        lambda: gf.iloc[-2:],
        lambda: gf["a"],
        lambda: gf.copy(),
        lambda: gf.head(20),
        lambda: gf.to_numpy(),
        lambda: np.asarray(gf),
        lambda: gf.values,
        lambda: gf[gf.index > 0],
    ):
        with pytest.raises(HoldoutAccessError):
            access()


@pytest.mark.unit
def test_guarded_frame_allows_dev_only_work() -> None:
    df = frame()
    hold = list(df.index[-4:])
    dev_view = guard(df, hold).loc[list(df.index[:-4])]
    assert len(dev_view) == 16
    assert dev_view["a"].mean() == pytest.approx(df.loc[df.index[:-4], "a"].mean())
    assert dev_view.to_numpy().shape == (16, 2)
    with pytest.raises(HoldoutAccessError):  # still guarded after selection
        dev_view.reindex(df.index)


# ---------------------------------------------------------------- ScriptedLLMProvider
def request() -> LLMRequest:
    return LLMRequest(purpose="plan", system="planner_system@1.0", user_json={"k": 1},
                      max_output_tokens=2000)  # fmt: skip


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["groq", "gemini", "openrouter", "ollama"])
def test_scripted_provider_replays_success(provider: str) -> None:
    p = ScriptedLLMProvider.from_fixture(provider, "success")
    resp = p.complete(request())
    assert resp.provider == provider
    assert resp.parsed is not None
    assert resp.parsed["schema_version"] == "plan/1.0"
    assert (resp.input_tokens, resp.output_tokens, resp.tokens_estimated) == (3100, 420, False)
    assert p.calls == 1
    assert p.requests[0] == request()


@pytest.mark.unit
def test_scripted_provider_failure_sequences() -> None:
    p = ScriptedLLMProvider.from_fixture("groq", "rate_limit_then_success")
    with pytest.raises(ScriptedRateLimited) as info:
        p.complete(request())
    assert info.value.retry_after == 2
    assert p.complete(request()).parsed is not None
    with pytest.raises(ScriptExhausted):
        p.complete(request())
    with pytest.raises(ScriptedHTTPError) as http:
        ScriptedLLMProvider.from_fixture("gemini", "auth_401").complete(request())
    assert http.value.status == 401
    with pytest.raises(ScriptedTimeout):
        ScriptedLLMProvider.from_fixture("ollama", "timeout_then_success").complete(request())


@pytest.mark.unit
def test_scripted_provider_leaves_fences_and_bad_json_to_the_router() -> None:
    fenced = ScriptedLLMProvider.from_fixture("openrouter", "fenced_json").complete(request())
    assert fenced.parsed is None
    assert "```json" in fenced.content
    bad = ScriptedLLMProvider.from_fixture("groq", "malformed_json").complete(request())
    assert bad.parsed is None
    no_usage = ScriptedLLMProvider.from_fixture("groq", "no_usage").complete(request())
    assert no_usage.tokens_estimated is True


@pytest.mark.unit
def test_scripted_provider_custom_error_mapping() -> None:
    class Classified(Exception):
        pass

    p = ScriptedLLMProvider.from_fixture("groq", "all_failures", raise_with=lambda s: Classified(s))
    with pytest.raises(Classified):
        p.complete(request())


# ---------------------------------------------------------------- network, clock, helpers
@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_network_blocked_by_default() -> None:
    with pytest.raises(Exception, match="socket"):
        socket.create_connection(("example.com", 80), timeout=1)


@pytest.mark.unit
def test_allow_network_refuses_non_live_tests(request: pytest.FixtureRequest) -> None:
    with pytest.raises(pytest.UsageError, match="live"):
        request.getfixturevalue("allow_network")


@pytest.mark.unit
def test_fake_clock(fake_clock: FakeClock) -> None:
    start = fake_clock.monotonic()
    fake_clock.sleep(2.5)
    fake_clock.advance(1)
    assert fake_clock.monotonic() - start == 3.5
    assert fake_clock.sleeps == [2.5]
    with pytest.raises(ValueError, match="negative"):
        fake_clock.sleep(-1)


@pytest.mark.unit
def test_tmp_run_dir(tmp_run_dir: Path) -> None:
    assert tmp_run_dir.parent.name == "runs"
    assert len(tmp_run_dir.name) == 26
    assert stat.S_IMODE(os.stat(tmp_run_dir).st_mode) == 0o700


@pytest.mark.unit
def test_raises_error_helper() -> None:
    with raises_error(ErrorCode.FILE_TYPE, ext=".exe"):
        raise TabPilotError(ErrorCode.FILE_TYPE, ext=".exe")
    with (
        pytest.raises(AssertionError, match="expected E-FILE-EMPTY"),
        raises_error(ErrorCode.FILE_EMPTY),
    ):
        raise TabPilotError(ErrorCode.FILE_TYPE, ext=".exe")
    with pytest.raises(AssertionError, match="path"), raises_error(ErrorCode.FILE_CORRUPT):
        raise TabPilotError(ErrorCode.FILE_CORRUPT, format="/home/user/x.csv")
