from collections.abc import Awaitable, Callable
from typing import Any

from app.core.config import Settings
from app.graph.state import AgentState
from app.graph.tools import build_tool_implementations


def build_tool_execution_node(settings: Settings) -> Callable[[AgentState], Awaitable[dict[str, list[dict]]]]:
    """Runs plain tool calls only — consult-type calls (sub-agents) go through
    consult_node instead (see build.py's router, which sends state to one or
    the other based on which tool name Claude asked for).
    """

    implementations = build_tool_implementations(settings)

    async def tool_execution_node(state: AgentState) -> dict[str, list[dict]]:
        messages = state["messages"]
        last_assistant_turn = messages[-1]["content"]
        username = state.get("username") or ""

        tool_results: list[dict[str, Any]] = []
        for block in last_assistant_turn:
            if block["type"] != "tool_use" or block["name"] not in implementations:
                continue
            result = await implementations[block["name"]](block.get("input", {}), username)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block["id"], "content": result}
            )

        return {"messages": [*messages, {"role": "user", "content": tool_results}]}

    return tool_execution_node
