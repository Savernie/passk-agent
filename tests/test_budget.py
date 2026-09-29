import pytest
import litellm
from litellm.exceptions import BudgetExceededError
import providers.deepseek as deepseek_module
from providers.deepseek import LiteLLMProvider
from budget import BudgetTracker


class FakeResponse:
    def __init__(self):
        class Usage:
            prompt_tokens = 10
            completion_tokens = 5
        class Message:
            content = "ok"
            tool_calls = None
        class Choice:
            message = Message()
        self.choices = [Choice()]
        self.usage = Usage()


@pytest.mark.asyncio
async def test_budget_cap_halts_mid_sweep(monkeypatch):
    calls = {"count": 0}

    async def fake_acompletion(**kwargs):
        calls["count"] += 1
        return FakeResponse()

    monkeypatch.setattr(deepseek_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        deepseek_module.litellm, "completion_cost", lambda completion_response: 1.0
    )

    tracker = BudgetTracker(max_budget=1.5)
    provider = LiteLLMProvider("fake-model", budget_tracker=tracker)

    await provider.complete(messages=[{"role": "user", "content": "1"}])
    await provider.complete(messages=[{"role": "user", "content": "2"}])

    assert tracker.current_spend == 2.0

    with pytest.raises(BudgetExceededError):
        await provider.complete(messages=[{"role": "user", "content": "3"}])

    assert calls["count"] == 2, "third call should have been blocked before hitting the network"