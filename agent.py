import asyncio
import json
from typing import Optional

from pydantic import BaseModel
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage, Message, SystemMessage, ToolCall, ToolMessage, UserMessage,
)
from tau2.environment.tool import Tool

from providers.base import Provider


class PasskAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[Message]


class PasskAgent(HalfDuplexAgent[PasskAgentState]):
    """Our own agent: decides tool calls vs. plain replies, using our
    Provider (retries + budget cap already built in) instead of tau2's
    own llm_agent."""

    def __init__(self, tools, domain_policy, provider, env):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.provider = provider
        self.env = env
        self.tool_schemas = [t.openai_schema for t in tools]

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> PasskAgentState:
        system_prompt = f"You are a customer service agent. Follow this policy exactly:\n\n{self.domain_policy}"
        return PasskAgentState(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=message_history or [],
        )

    def _to_litellm_messages(self, state: PasskAgentState) -> list[dict]:
        msgs = [{"role": "system", "content": sm.content} for sm in state.system_messages]
        for m in state.messages:
            if isinstance(m, UserMessage):
                msgs.append({"role": "user", "content": m.content or ""})
            elif isinstance(m, AssistantMessage):
                entry = {"role": "assistant", "content": m.content}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in m.tool_calls
                    ]
                msgs.append(entry)
            elif isinstance(m, ToolMessage):
                msgs.append({"role": "tool", "tool_call_id": m.id, "content": m.content or ""})
            # MultiToolMessage not handled yet -- deferred to the multi-tool-failure session
        return msgs

    def generate_next_message(self, message, state):
        state.messages.append(message)

        while True:
            litellm_messages = self._to_litellm_messages(state)
            completion = asyncio.run(self.provider.complete(messages=litellm_messages, tools=self.tool_schemas))

            tool_calls = [
                ToolCall(id=tc["id"], name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]), requestor="assistant")
                for tc in completion.tool_calls
            ] if completion.tool_calls else None

            assistant_message = AssistantMessage(role="assistant", content=completion.content, tool_calls=tool_calls)
            state.messages.append(assistant_message)

            if not tool_calls:
                return assistant_message, state          # done — plain content, hand back to user

            for tc in tool_calls:                        # execute each tool, feed results back, loop again
                func = getattr(self.env.tools, tc.name)
                try:
                    result = str(func(**tc.arguments))
                except Exception as e:
                    result = f"Error: {e}"
                state.messages.append(ToolMessage(id=tc.id, role="tool", content=result))