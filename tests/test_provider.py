import asyncio
from providers.deepseek import LiteLLMProvider
from dotenv import load_dotenv;

load_dotenv()

async def main():
    provider = LiteLLMProvider("openrouter/deepseek/deepseek-chat")

    result = await provider.complete(
        messages=[{"role": "user", "content": "Say hello in exactly 3 words."}]
    )

    print("Content:", result.content)
    print("Tool calls:", result.tool_calls)
    print("Usage:", result.usage)


if __name__ == "__main__":
    asyncio.run(main())