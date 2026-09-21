from dataclasses import dataclass, field
from typing import Protocol, Any


@dataclass
class Completion:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class Provider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Completion:
        ...