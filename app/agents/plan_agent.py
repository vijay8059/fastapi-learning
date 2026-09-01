from anthropic import AsyncAnthropic

from app.core.config import Settings


async def run_plan_agent(message: str, settings: Settings) -> str:
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text
