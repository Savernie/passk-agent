import asyncio
from litellm.exceptions import BudgetExceededError


class BudgetTracker:
    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.current_spend = 0.0
        self._lock = asyncio.Lock()

    async def check(self):
        async with self._lock:
            if self.current_spend >= self.max_budget:
                raise BudgetExceededError(
                    current_cost=self.current_spend,
                    max_budget=self.max_budget,
                )

    async def record(self, amount: float):
        async with self._lock:
            self.current_spend += amount