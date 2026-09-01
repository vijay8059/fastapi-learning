from collections.abc import Awaitable, Callable
from typing import Any

from app.core.config import Settings
from app.graph.state import AgentState
from app.graph.subgraphs.knowledge_agent import build_knowledge_agent_graph


def build_consult_node(settings: Settings) -> Callable[[AgentState], Awaitable[dict[str, list[dict]]]]:
    """Runs consult-type tool calls only — i.e. sub-agents, not plain tools.

    Separate from tool_execution_node on purpose: this is where a sub-agent's
    compiled subgraph actually gets invoked (mirrors assist-workflows' consult
    node, which fans out to os_mcp_agent/knowledge_agent subgraphs, distinct
    from its own tool_execution node for plain tools).
    """

    knowledge_agent_graph = build_knowledge_agent_graph(settings)
    consult_graphs: dict[str, Any] = {"consult_knowledge_agent": knowledge_agent_graph}

    async def consult_node(state: AgentState) -> dict[str, list[dict]]:
        messages = state["messages"]
        last_assistant_turn = messages[-1]["content"]

        tool_results: list[dict[str, Any]] = []
        for block in last_assistant_turn:
            if block["type"] != "tool_use" or block["name"] not in consult_graphs:
                continue
            graph = consult_graphs[block["name"]]
            result = await graph.ainvoke({"question": block["input"]["question"]})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block["id"], "content": result["answer"]}
            )

        return {"messages": [*messages, {"role": "user", "content": tool_results}]}

    return consult_node
