import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from providers.base import Completion

RETRYABLE_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
)


class LiteLLMProvider:
    def __init__(self, model: str, budget_tracker: "BudgetTracker", **default_kwargs):
        self.model = model
        self.budget_tracker = budget_tracker
        self.default_kwargs = default_kwargs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=20),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Completion:
        await self.budget_tracker.check()
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            tools=tools,
            timeout=30.0,
            **{**self.default_kwargs, **kwargs},
        )
        cost = litellm.completion_cost(completion_response=response)
        await self.budget_tracker.record(cost)
        
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