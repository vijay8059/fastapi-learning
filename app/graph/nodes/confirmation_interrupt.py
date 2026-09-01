from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.types import interrupt

from app.graph.state import AgentState
from app.graph.tools import WRITE_TOOL_NAMES


def build_confirmation_interrupt_node() -> Callable[[AgentState], Awaitable[dict[str, list[dict]]]]:
    """Pauses the graph before a write-tool call runs, waiting for a human
    decision. Mirrors assist-workflows' confirmation_interrupt node: same
    langgraph.types.interrupt() mechanism, fails closed to "cancel" on any
    unrecognized resume value.

    On first entry, interrupt(...) pauses the whole graph run here — ainvoke
    returns immediately with `__interrupt__` in its result, nothing below
    that line executes yet. On resume (Command(resume=decision), same
    thread_id), this node re-runs from the top, but this time interrupt(...)
    returns the decision instead of pausing again.
    """

    async def confirmation_interrupt_node(state: AgentState) -> dict[str, list[dict]]:
        messages = state["messages"]
        last_assistant_turn = messages[-1]["content"]
        pending_writes = [
            block
            for block in last_assistant_turn
            if block["type"] == "tool_use" and block["name"] in WRITE_TOOL_NAMES
        ]

        decision = interrupt({"kind": "write_confirmation", "pending_writes": pending_writes})
        decision = decision if decision in ("confirm", "cancel") else "cancel"

        tool_results: list[dict[str, Any]] = []
        for block in pending_writes:
            if decision == "confirm":
                content = f"Alert sent: {block['input']['message']}"
            else:
                content = "Cancelled by the user — action was not performed."
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block["id"], "content": content}
            )

        return {"messages": [*messages, {"role": "user", "content": tool_results}]}

    return confirmation_interrupt_node
