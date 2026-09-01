from collections.abc import Awaitable, Callable
from typing import Any

from anthropic import AsyncAnthropic

from app.core.config import Settings
from app.graph.state import AgentState
from app.graph.tools import TOOLS


def build_coordinator_node(settings: Settings) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """The reasoning node — calls Claude with the capsule-composed system
    prompt and the ``get_secret_number`` tool bound.

    Runs once per loop iteration (see graph.py's coordinator<->tool_execution
    edge). ``messages`` is always already set by the time this runs —
    grounding_node appends the new user turn on the first entry of a turn;
    later entries (from tool_execution/consult/confirmation_interrupt) have
    already extended it with a tool_result turn themselves. ``stop_reason``
    is read only by the router — it never reaches an LLM call or a caller.
    """

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def coordinator_node(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]

        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=state["system_prompt"],
            messages=messages,
            tools=TOOLS,
        )

        assistant_content = [block.model_dump() for block in response.content]
        updated_messages = [*messages, {"role": "assistant", "content": assistant_content}]

        result: dict[str, Any] = {
            "messages": updated_messages,
            "stop_reason": response.stop_reason,
        }
        if response.stop_reason != "tool_use":
            result["reply"] = "\n".join(
                block["text"] for block in assistant_content if block["type"] == "text"
            )
        return result

    return coordinator_node
