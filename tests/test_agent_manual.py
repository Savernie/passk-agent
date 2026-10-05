import sys
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from agent import PasskAgent
from budget import BudgetTracker
from providers.deepseek import LiteLLMProvider
from tau2.data_model.message import UserMessage
from tau2.domains.airline.environment import get_environment
import json


def main():
    env = get_environment()
    tracker = BudgetTracker(max_budget=1.0)
    provider = LiteLLMProvider("openrouter/deepseek/deepseek-chat", budget_tracker=tracker)

    agent = PasskAgent(tools=env.get_tools(), domain_policy=env.get_policy(), provider=provider, env=env)
    state = agent.get_init_state()

    # with open("agent_debug.txt", "w", encoding="utf-8") as f:
    #     f.write("=== SYSTEM PROMPT ===\n")
    #     f.write(state.system_messages[0].content + "\n\n")

    #     f.write("=== DOMAIN POLICY (raw) ===\n")
    #     f.write(str(agent.domain_policy) + "\n\n")

    #     f.write("=== TOOL SCHEMAS ===\n")
    #     f.write(json.dumps(agent.tool_schemas, indent=2) + "\n")

    user_msg = UserMessage(role="user", content="Hi, I need to check the details of reservation Q69X3R.")
    assistant_msg, state = agent.generate_next_message(user_msg, state)

    print("Content:", assistant_msg.content)
    print("Tool calls:", assistant_msg.tool_calls)


if __name__ == "__main__":
    main()