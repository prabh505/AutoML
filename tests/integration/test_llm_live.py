"""Live provider calls (P10 deliverable 9). Opt-in: `pytest -m live`, and each case runs only when
its API key and model id are set. Records the real response shape to check UV-002.
"""

from __future__ import annotations

import os

import pytest

from tabpilot.config import load_settings
from tabpilot.llm import BudgetTracker, ConsentStore, LLMRouter, build_providers
from tabpilot.schemas import LLMBudget, LLMRequest, ProviderName

pytestmark = pytest.mark.live


@pytest.mark.parametrize("name", list(ProviderName))
def test_live_provider_returns_parseable_json(name: ProviderName, allow_network: None) -> None:
    settings = load_settings(dict(os.environ))
    provider = next(p for p in build_providers(settings, os.environ) if p.name == name.value)
    if not provider.available():
        pytest.skip("KI-025: no API key / model id for this provider in this environment")
    router = LLMRouter([provider], consent=ConsentStore.granted_for([name]))
    req = LLMRequest(purpose="plan", system='Reply with the JSON object {"ok": true} only.',
                     user_json={"ping": 1}, max_output_tokens=20)  # fmt: skip
    resp = router.complete(req, BudgetTracker(LLMBudget(max_calls=3, max_wall_s=120.0)))
    assert resp.provider == name.value
    assert resp.parsed is not None
