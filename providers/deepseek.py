import litellm
from providers.base import Completion


class LiteLLMProvider:
    """Works for any litellm-supported model string —
    'openrouter/deepseek/deepseek-chat', 'gemini/gemini-2.5-flash', etc.
    One class, swap the model string, not the code."""

    def __init__(self, model: str, **default_kwargs):
        self.model = model
        self.default_kwargs = default_kwargs

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Completion:
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            tools=tools,
            **{**self.default_kwargs, **kwargs},
        )

        choice = response.choices[0].message
        return Completion(
            content=choice.content,
            tool_calls=[tc.model_dump() for tc in (choice.tool_calls or [])],
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "cost_usd": litellm.completion_cost(completion_response=response),
            },
        )