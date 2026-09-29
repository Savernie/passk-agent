import pytest
import litellm
import providers.deepseek as deepseek_module
from providers.deepseek import LiteLLMProvider


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
async def test_retries_transient_error_then_succeeds(monkeypatch):
    calls = {"count": 0}

    async def fake_acompletion(**kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise litellm.Timeout(
                message="simulated timeout",
                model="fake-model",
                llm_provider="fake-provider",
            )
        return FakeResponse()

    monkeypatch.setattr(deepseek_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        deepseek_module.litellm, "completion_cost", lambda completion_response: 0.001
    )

    provider = LiteLLMProvider("fake-model")
    result = await provider.complete(messages=[{"role": "user", "content": "hi"}])

    assert calls["count"] == 3, "should have failed twice, succeeded on 3rd try"
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_does_not_retry_non_retryable_error(monkeypatch):
    calls = {"count": 0}

    async def fake_acompletion(**kwargs):
        calls["count"] += 1
        raise litellm.AuthenticationError(
            message="bad key",
            llm_provider="fake-provider",
            model="fake-model",
        )

    monkeypatch.setattr(deepseek_module.litellm, "acompletion", fake_acompletion)

    provider = LiteLLMProvider("fake-model")
    with pytest.raises(litellm.AuthenticationError):
        await provider.complete(messages=[{"role": "user", "content": "hi"}])

    assert calls["count"] == 1, "auth errors should never be retried"