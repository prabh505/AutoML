"""P10 router tests: T-LLM-01…04, T-LLM-06, T-LLM-07, partial T-PRIV-04 (PRD §23, §6.4, §24.5)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion
from tests.helpers.clock import FakeClock
from tests.helpers.llm_stub import LLM_FIXTURES, ScriptedLLMProvider, load_script

from tabpilot.config import load_settings
from tabpilot.llm import (
    AuthError,
    BadRequest,
    BudgetTracker,
    ConsentStore,
    GeminiProvider,
    LLMBudgetExhausted,
    LLMCache,
    LLMCallRecorder,
    LLMConnectionError,
    LLMRouter,
    LLMTimeout,
    LLMUnavailable,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderConsent,
    RateLimited,
    ServerError,
    build_providers,
    build_router,
    classify,
    classify_status,
    default_budget,
    estimate_tokens,
    extract_json,
    privacy_notice,
    request_hash,
)
from tabpilot.llm.adapters.gemini import GEMINI_OPENAI_BASE_URL
from tabpilot.schemas import EventType, ExecutionProfileName, LLMBudget, LLMRequest, ProviderName
from tabpilot.schemas.run import BudgetLedger

PROVIDERS = ["groq", "gemini", "openrouter", "ollama"]
SECRET = "gsk_" + "Z" * 40
BODY_MARKER = "PROMPT-BODY-MARKER-7f3a"


def router_error(step: dict[str, Any]) -> Exception:
    """Scripted failure steps → the router's classified errors."""
    if step["kind"] == "timeout":
        return LLMTimeout("timeout")
    if step["kind"] == "connection_error":
        return LLMConnectionError("refused")
    return classify_status(int(step["status"]), step.get("retry_after"))


def scripted(name: str, scenario: str, **kw: Any) -> ScriptedLLMProvider:
    return ScriptedLLMProvider.from_fixture(name, scenario, raise_with=router_error, **kw)


def request(**kw: Any) -> LLMRequest:
    data = {"purpose": "plan", "system": "You are a planner. Respond with JSON only.",
            "user_json": {"task": "binary", "note": BODY_MARKER}, "max_output_tokens": 500}  # fmt: skip
    return LLMRequest(**{**data, **kw})


def tracker(clock: FakeClock | None = None, **kw: Any) -> BudgetTracker:
    budget = LLMBudget(**{"max_wall_s": 1000.0, **kw})
    return BudgetTracker(budget, clock=(clock or FakeClock()).monotonic)


def make_router(
    providers: list[Any], clock: FakeClock | None = None, **kw: Any
) -> tuple[LLMRouter, FakeClock, list[Any]]:
    clock = clock or FakeClock()
    events: list[Any] = []
    router = LLMRouter(providers, consent=ConsentStore.granted_for(list(ProviderName)),
                       sleep=clock.sleep, jitter=lambda: 0.5,
                       events=lambda kind, data: events.append((kind, data)), **kw)  # fmt: skip
    return router, clock, events


# ---------------------------------------------------------------- T-LLM-01 adapters
class FakeClient:
    """`openai.OpenAI` stand-in returning a recorded body or raising an `openai` exception."""

    def __init__(self, outcome: Any, calls: list[dict[str, Any]]) -> None:
        self._outcome, self._calls = outcome, calls
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self._calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return ChatCompletion.model_validate(self._outcome)


def recorded_body(provider: str, scenario: str) -> dict[str, Any]:
    step = next(s for s in load_script(provider, scenario) if s["kind"] == "response")
    return step["body"]


def env_for(settings: Any) -> dict[str, str]:
    return {e.api_key_env: SECRET for e in settings.providers.values() if e.api_key_env}


def adapter(name: str, outcome: Any, calls: list[dict[str, Any]], **entry_kw: Any) -> Any:
    settings = load_settings({"TABPILOT_GROQ_MODEL": "m-groq", "TABPILOT_GEMINI_MODEL": "m-gem",
                              "TABPILOT_OR_MODEL_1": "m-or1", "TABPILOT_OR_MODEL_2": "m-or2",
                              "TABPILOT_OLLAMA_MODEL": "m-oll"})  # fmt: skip
    provider = ProviderName(name)
    entry = settings.providers[provider].model_copy(update=entry_kw)
    factories: list[Any] = []

    def factory(base_url: str | None, key: str, timeout: float) -> FakeClient:
        factories.append((base_url, key, timeout))
        return FakeClient(outcome, calls)

    cls = {"gemini": GeminiProvider, "ollama": OllamaProvider}.get(name, OpenAICompatibleProvider)
    inst = cls(provider, entry, env_for(settings), factory)
    inst.factories = factories
    return inst


@pytest.mark.unit
@pytest.mark.parametrize("name", PROVIDERS)
def test_T_LLM_01_adapters_parse_recorded_responses(name: str) -> None:
    calls: list[dict[str, Any]] = []
    plain = adapter(name, recorded_body(name, "success"), calls).complete(request())
    assert plain.provider == name
    assert plain.parsed is not None
    assert not plain.tokens_estimated
    assert plain.input_tokens == recorded_body(name, "success")["usage"]["prompt_tokens"]
    fenced = adapter(name, recorded_body(name, "fenced_json"), calls).complete(request())
    assert fenced.parsed is not None
    assert fenced.parsed["schema_version"] == "plan/1.0"
    no_usage = adapter(name, recorded_body(name, "no_usage"), calls).complete(request())
    assert no_usage.tokens_estimated
    assert no_usage.output_tokens == estimate_tokens(no_usage.content)
    malformed = adapter(name, recorded_body(name, "malformed_json"), calls).complete(request())
    assert malformed.parsed is None  # the caller repairs (§12.5)
    sent = calls[0]
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 500
    assert sent["messages"][0] == {"role": "system", "content": request().system}
    assert json.loads(sent["messages"][1]["content"]) == request().user_json
    if adapter(name, {}, []).supports_json_mode():  # declared in providers.yaml, not assumed
        assert sent["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in sent


@pytest.mark.unit
def test_adapter_configuration_comes_from_settings() -> None:
    calls: list[dict[str, Any]] = []
    groq = adapter("groq", recorded_body("groq", "success"), calls, json_mode=False)
    groq.complete(request(timeout_s=33.0))
    assert "response_format" not in calls[-1]  # JSON mode only where configured
    assert calls[-1]["model"] == "m-groq"
    assert calls[-1]["timeout"] == 33.0
    assert groq.factories[-1] == ("https://api.groq.com/openai/v1", SECRET, 33.0)
    gem = adapter("gemini", recorded_body("gemini", "success"), calls)
    gem.complete(request())
    assert gem.factories[-1][0] == GEMINI_OPENAI_BASE_URL
    oll = adapter("ollama", recorded_body("ollama", "success"), calls)
    assert oll.available()
    assert oll.factories == []
    oll.complete(request())
    assert oll.factories[-1][1] == "not-needed"
    assert not adapter("groq", {}, [], model_id=None).available()  # model id unset


@pytest.mark.unit
def test_keyless_provider_is_unavailable() -> None:
    settings = load_settings({"TABPILOT_GROQ_MODEL": "m"})
    groq = OpenAICompatibleProvider(ProviderName.GROQ, settings.providers[ProviderName.GROQ], {})
    assert not groq.available()


@pytest.mark.unit
def test_openrouter_rotates_to_next_model_on_404() -> None:
    calls: list[dict[str, Any]] = []
    response = httpx.Response(404, request=httpx.Request("POST", "https://x"))
    gone = openai.NotFoundError("gone", response=response, body=None)
    provider = adapter("openrouter", gone, calls)
    with pytest.raises(BadRequest):
        provider.complete(request())
    assert [c["model"] for c in calls] == ["m-or1", "m-or2"]


# ---------------------------------------------------------------- classification
def status_error(cls: Any, status: int, headers: dict[str, str] | None = None) -> Exception:
    response = httpx.Response(
        status, headers=headers or {}, request=httpx.Request("POST", "https://x")
    )
    return cls("error body must not leak", response=response, body={"secret": BODY_MARKER})


@pytest.mark.unit
@pytest.mark.parametrize(("exc", "kind"), [
    (status_error(openai.RateLimitError, 429, {"retry-after": "7"}), RateLimited),
    (status_error(openai.AuthenticationError, 401), AuthError),
    (status_error(openai.PermissionDeniedError, 403), AuthError),
    (status_error(openai.BadRequestError, 400), BadRequest),
    (status_error(openai.InternalServerError, 503), ServerError),
    (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), LLMTimeout),
    (openai.APIConnectionError(request=httpx.Request("POST", "https://x")), LLMConnectionError),
    (TimeoutError(), LLMTimeout), (ConnectionError(), LLMConnectionError), (RuntimeError(), ServerError),
])  # fmt: skip
def test_client_errors_are_classified_without_bodies(exc: Exception, kind: type) -> None:
    classified = classify(exc)
    assert isinstance(classified, kind)
    assert BODY_MARKER not in str(classified)
    if isinstance(classified, RateLimited):
        assert classified.retry_after == 7.0
    assert classify(classified) is classified
    assert classify_status(408).__class__ is LLMTimeout


@pytest.mark.unit
def test_rate_limit_without_parseable_retry_after() -> None:
    exc = classify(status_error(openai.RateLimitError, 429, {"retry-after": "soon"}))
    assert isinstance(exc, RateLimited)
    assert exc.retry_after is None


# ---------------------------------------------------------------- T-LLM-02..04 router
@pytest.mark.unit
def test_T_LLM_02_short_rate_limit_waits_then_succeeds() -> None:
    router, clock, _events = make_router([scripted("groq", "rate_limit_then_success")])
    resp = router.complete(request(), tracker(clock))
    assert resp.provider == "groq"
    assert resp.attempts == 2
    assert clock.sleeps == [2]


@pytest.mark.unit
def test_T_LLM_02_long_rate_limit_moves_to_next_provider() -> None:
    groq, gemini = scripted("groq", "rate_limit_long"), scripted("gemini", "success")
    router, clock, events = make_router([groq, gemini])
    resp = router.complete(request(), tracker(clock))
    assert resp.provider == "gemini"
    assert clock.sleeps == []
    assert (EventType.LLM_FALLBACK, {"from": "groq", "to": "gemini"}) in events
    assert "groq" not in router.unavailable  # rate limits do not disable the provider


@pytest.mark.unit
@pytest.mark.parametrize("scenario", ["server_error_then_success", "timeout_then_success"])
def test_T_LLM_03_transient_errors_back_off_and_retry(scenario: str) -> None:
    router, clock, _ = make_router([scripted("groq", scenario)])
    resp = router.complete(request(), tracker(clock))
    assert resp.attempts == 2
    assert clock.sleeps == [2.5]


@pytest.mark.unit
def test_T_LLM_03_backoff_is_capped_and_three_attempts_then_next() -> None:
    groq = scripted("groq", "connection_error")
    router, clock, events = make_router([groq, scripted("gemini", "success")])
    resp = router.complete(request(), tracker(clock))
    assert groq.calls == 3
    assert resp.provider == "gemini"
    assert clock.sleeps == [2.5, 4.5]  # no sleep after the last attempt
    failed = [d for k, d in events if k == EventType.LLM_CALL_FAILED]
    assert [d["error_class"] for d in failed] == ["LLMConnectionError"] * 3


@pytest.mark.unit
@pytest.mark.parametrize("scenario", ["auth_401", "bad_request_400"])
def test_T_LLM_04_auth_and_bad_request_skip_provider_for_the_run(scenario: str) -> None:
    groq = scripted("groq", scenario)
    gemini = ScriptedLLMProvider(
        "gemini", load_script("gemini", "success") * 2, raise_with=router_error
    )
    router, clock, _ = make_router([groq, gemini])
    assert router.complete(request(), tracker(clock)).provider == "gemini"
    assert "groq" in router.unavailable
    assert groq.calls == 1
    assert router.complete(request(purpose="reflect"), tracker(clock)).provider == "gemini"
    assert groq.calls == 1  # never called again in this run


@pytest.mark.unit
def test_all_providers_down_raises_llm_unavailable() -> None:
    providers = [scripted(p, "all_failures") for p in ("groq", "gemini")]
    router, clock, events = make_router(providers)
    with pytest.raises(LLMUnavailable) as err:
        router.complete(request(), tracker(clock))
    assert not isinstance(err.value, LLMBudgetExhausted)
    assert (EventType.LLM_UNAVAILABLE, {"purpose": "plan"}) in events


@pytest.mark.unit
def test_unconsented_and_unavailable_providers_are_never_called() -> None:
    groq, gemini = scripted("groq", "success"), scripted("gemini", "success", is_available=False)
    ollama = scripted("ollama", "success")
    router = LLMRouter([groq, gemini, ollama],
                       consent=ConsentStore([ProviderConsent(provider=ProviderName.GROQ, granted=False),
                                             ProviderConsent(provider=ProviderName.OLLAMA, granted=True)]))  # fmt: skip
    assert router.complete(request(), tracker()).provider == "ollama"
    assert groq.calls == 0
    assert gemini.calls == 0
    assert not ConsentStore().allows("groq")
    assert not ConsentStore().allows("nope")


# ---------------------------------------------------------------- T-LLM-06 identical payload
@pytest.mark.unit
def test_T_LLM_06_fallback_sends_identical_payload() -> None:
    providers = [scripted("groq", "auth_401"), scripted("gemini", "rate_limit_long"),
                 scripted("openrouter", "success")]  # fmt: skip
    router, clock, _ = make_router(providers)
    router.complete(request(), tracker(clock))
    sent = [r for p in providers for r in p.requests]
    assert len(sent) == 3
    assert {request_hash(r) for r in sent} == {request_hash(request())}
    assert {(r.purpose, r.system, json.dumps(r.user_json, sort_keys=True)) for r in sent} == {
        (request().purpose, request().system, json.dumps(request().user_json, sort_keys=True))
    }


# ---------------------------------------------------------------- T-LLM-07 budgets
@pytest.mark.unit
def test_T_LLM_07_call_budget_counts_every_attempt_and_stops_mid_retry() -> None:
    groq = scripted("groq", "connection_error")
    router, clock, _ = make_router([groq, scripted("gemini", "success")])
    budget = tracker(clock, max_calls=2)
    with pytest.raises(LLMBudgetExhausted, match="call budget"):
        router.complete(request(), budget)
    assert groq.calls == 2
    assert budget.calls == 2


@pytest.mark.unit
def test_T_LLM_07_token_ceilings() -> None:
    router, clock, _ = make_router([scripted("groq", "success")])
    with pytest.raises(LLMBudgetExhausted, match="per-call input-token cap"):
        router.complete(request(user_json={"blob": "x" * 30_000}), tracker(clock))
    with pytest.raises(LLMBudgetExhausted, match="input-token budget"):
        router.complete(request(), tracker(clock, max_input_tokens=5))
    budget = tracker(clock)
    router.complete(request(), budget)
    budget.output_tokens = budget.budget.max_output_tokens
    with pytest.raises(LLMBudgetExhausted, match="output-token budget"):
        router.complete(request(purpose="reflect"), budget)


@pytest.mark.unit
def test_T_LLM_07_wall_time_budget_includes_backoff() -> None:
    router, clock, _ = make_router([scripted("groq", "server_error_then_success")])
    budget = tracker(clock, max_wall_s=2.0)  # the 2.5 s backoff does not fit
    with pytest.raises(LLMBudgetExhausted, match="no time left"):
        router.complete(request(), budget)
    spent = tracker(clock, max_wall_s=1.0)
    spent.wall_s = 1.0
    with pytest.raises(LLMBudgetExhausted, match="wall-time"):
        router.complete(request(), spent)


@pytest.mark.unit
def test_per_attempt_timeout_respects_budget_caps() -> None:
    groq = scripted("groq", "success")
    router, clock, _ = make_router([groq])
    router.complete(
        request(timeout_s=200.0), tracker(clock, per_call_timeout_s=60.0, max_wall_s=30.0)
    )
    assert groq.requests[0].timeout_s == 30.0  # min(request, per-call cap, wall time left)


@pytest.mark.unit
def test_budget_usage_flows_into_the_run_ledger() -> None:
    router, clock, _ = make_router([scripted("groq", "success")])
    budget = tracker(clock)
    router.complete(request(), budget)
    ledger = budget.add_to(BudgetLedger(time_budget_s=960, trials_max=60))
    assert ledger.llm_calls_used == 1
    assert ledger.llm_input_tokens_used == budget.input_tokens > 0


@pytest.mark.unit
def test_default_budget_follows_A_004() -> None:
    budget = default_budget(960, 12)
    assert (budget.max_calls, budget.max_wall_s, budget.per_call_timeout_s) == (12, 192.0, 90.0)
    assert (budget.max_input_tokens, budget.max_output_tokens) == (40_000, 8_000)


# ---------------------------------------------------------------- cache (§6.4, §23.8)
@pytest.mark.unit
def test_cache_hit_costs_zero_calls_and_is_recorded(tmp_path: Path) -> None:
    groq = ScriptedLLMProvider("groq", load_script("groq", "success"), raise_with=router_error)
    recorder = LLMCallRecorder(tmp_path)
    router, clock, _ = make_router([groq], cache=LLMCache(), recorder=recorder)
    budget = tracker(clock)
    first = router.complete(request(), budget)
    second = router.complete(request(), budget)
    assert groq.calls == 1
    assert budget.calls == 1
    assert second.cached
    assert not first.cached
    assert second.content == first.content
    lines = [json.loads(x) for x in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert [x["outcome"] for x in lines] == ["success", "cache_hit"]
    assert lines[1]["cached"] is True


@pytest.mark.unit
def test_disk_cache_ttl_and_disabled_flag(tmp_path: Path) -> None:
    clock = FakeClock(now=1_000_000.0)
    cache = LLMCache(directory=tmp_path, clock=clock.time)
    resp = scripted("groq", "success").complete(request())
    cache.put("k", resp)
    fresh = LLMCache(directory=tmp_path, clock=clock.time)
    os.utime(tmp_path / "k.json", (clock.now, clock.now))
    assert fresh.get("k") == resp
    clock.advance(8 * 86_400)
    assert LLMCache(directory=tmp_path, clock=clock.time).get("k") is None  # expired, removed
    assert not (tmp_path / "k.json").exists()
    off = LLMCache(enabled=False, directory=tmp_path)
    off.put("k", resp)
    assert off.get("k") is None
    assert not (tmp_path / "k.json").exists()


@pytest.mark.unit
def test_hosted_router_keeps_cache_in_memory_and_writes_no_bodies(tmp_path: Path) -> None:
    """T-PRIV-04 (partial): hosted runs persist no prompt/response bodies, only hashes."""
    settings = load_settings({"TABPILOT_PROFILE": "hosted"})
    router = build_router(settings, {}, consent=ConsentStore.granted_for([ProviderName.GROQ]),
                          records_dir=tmp_path / "records", cache_dir=tmp_path / "cache")  # fmt: skip
    assert router.cache.directory is None
    router.providers = [
        ScriptedLLMProvider("groq", load_script("groq", "success"), raise_with=router_error)
    ]
    router.complete(request(), tracker())
    assert not (tmp_path / "cache").exists()
    text = (tmp_path / "records" / "llm_calls.jsonl").read_text()
    assert BODY_MARKER not in text
    assert "planner" not in text
    record = json.loads(text.splitlines()[0])
    assert record["request_sha256"] == request_hash(request())
    assert len(record["response_sha256"]) == 64
    local = build_router(load_settings({}), {}, consent=ConsentStore(), cache_dir=tmp_path / "c")
    assert local.cache.directory == tmp_path / "c"


@pytest.mark.unit
def test_recorder_refuses_body_fields(tmp_path: Path) -> None:
    recorder = LLMCallRecorder(tmp_path)
    with pytest.raises(ValueError, match="not an llm_calls field"):
        recorder.write(purpose="plan", content="the prompt body")


# ---------------------------------------------------------------- T-PRIV-04 logs
@pytest.mark.unit
def test_T_PRIV_04_logs_hold_no_bodies_or_keys(caplog: pytest.LogCaptureFixture) -> None:
    providers = [scripted("groq", "auth_401"), scripted("gemini", "connection_error"),
                 scripted("openrouter", "fenced_json")]  # fmt: skip
    router, clock, events = make_router(providers)
    with caplog.at_level(logging.DEBUG):
        resp = router.complete(request(), tracker(clock))
    logged = caplog.text + json.dumps([d for _, d in events], default=str)
    assert BODY_MARKER not in logged
    assert SECRET not in logged
    assert resp.content not in logged
    assert request().system not in logged
    assert "llm_call_failed" in caplog.text  # metadata is logged


@pytest.mark.unit
def test_api_keys_never_reach_logs_or_the_adapter_state(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[dict[str, Any]] = []
    groq = adapter("groq", status_error(openai.AuthenticationError, 401), calls)
    router = LLMRouter([groq], consent=ConsentStore.granted_for([ProviderName.GROQ]))
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMUnavailable):
        router.complete(request(), tracker())
    assert SECRET not in caplog.text
    state = {k: v for k, v in vars(groq).items() if k not in ("_env", "factories")}
    assert SECRET not in repr(state)  # the key is read from the environment per call, never kept


# ---------------------------------------------------------------- factory, notices, normalize
@pytest.mark.unit
def test_build_providers_follows_profile_order() -> None:
    hosted = build_providers(load_settings({"TABPILOT_PROFILE": "hosted"}), {})
    assert [p.name for p in hosted] == ["groq", "gemini", "openrouter"]
    local = build_providers(load_settings({}), {})
    assert [p.name for p in local] == ["ollama", "groq", "gemini", "openrouter"]
    assert isinstance(local[0], OllamaProvider)
    assert isinstance(local[2], GeminiProvider)
    assert load_settings({}).profile == ExecutionProfileName.LOCAL


@pytest.mark.unit
@pytest.mark.parametrize("provider", list(ProviderName))
def test_privacy_notices_exist_and_carry_verify_note(provider: ProviderName) -> None:
    text = privacy_notice(provider)
    assert "verify the provider's current terms before deployment" in text
    assert "never sends raw rows" in text


@pytest.mark.unit
@pytest.mark.parametrize(("content", "expected"), [
    ('{"a": 1}', {"a": 1}),
    ('Sure!\n```json\n{"a": {"b": 2}}\n```', {"a": {"b": 2}}),
    ("no json here", None), ("{broken", None), ('["list"]', None),
])  # fmt: skip
def test_extract_json(content: str, expected: Any) -> None:
    assert extract_json(content) == expected


@pytest.mark.unit
def test_llm_fixtures_are_marked_synthetic() -> None:
    for provider in PROVIDERS:
        for path in (LLM_FIXTURES / provider).glob("*.json"):
            assert json.loads(path.read_text())["synthetic"] is True  # UV-002


@pytest.mark.unit
def test_T_LLM_03_backoff_never_exceeds_ten_seconds() -> None:
    clock = FakeClock()
    router = LLMRouter([scripted("groq", "connection_error")], sleep=clock.sleep,
                       jitter=lambda: 9.9, consent=ConsentStore.granted_for([ProviderName.GROQ]))  # fmt: skip
    with pytest.raises(LLMUnavailable):
        router.complete(request(), tracker(clock))
    assert clock.sleeps == [10.0, 10.0]  # min(2^a + jitter, 10)
