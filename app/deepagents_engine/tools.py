"""The same tools as `app/graph/tools.py`, re-exposed as LangChain
`StructuredTool`s — the only shape `create_deep_agent` accepts.

Same adaptation problem as `app/agentsdk/tools.py`, same solution: the
harness calls the tool itself, so `username` can't be an argument and rides
a ContextVar instead. The schemas still come from `TOOLS`; nothing is
retyped here.

`consult_knowledge_agent` is absent again — deepagents has its own sub-agent
concept (a `SubAgent` dict reached through the built-in `task` tool), so it
stops being a tool we implement. See KNOWLEDGE_SUBAGENT below.
"""

from contextvars import ContextVar
from typing import Any

from deepagents import SubAgent
from langchain_core.tools import StructuredTool

from app.core.config import Settings
from app.graph.tools import CONSULT_TOOL_NAMES, TOOLS, build_tool_implementations

current_username: ContextVar[str] = ContextVar("current_username", default="")

# The write tool deepagents will pause on — see `interrupt_on` in runner.py.
WRITE_TOOL_NAME = "send_alert"


def build_tools(settings: Settings) -> list[StructuredTool]:
    implementations = build_tool_implementations(settings)

    def _adapt(name: str):
        impl = implementations[name]

        async def coroutine(**kwargs: Any) -> str:
            return await impl(kwargs, current_username.get())

        return coroutine

    async def _send_alert(message: str) -> str:
        # Gated by interrupt_on, so by the time this runs the human approved.
        return f"Alert sent: {message}"

    tools: list[StructuredTool] = []
    for spec in TOOLS:
        name = spec["name"]
        if name in CONSULT_TOOL_NAMES:
            continue
        coroutine = _send_alert if name == WRITE_TOOL_NAME else _adapt(name)
        tools.append(
            StructuredTool(
                name=name,
                description=spec["description"],
                args_schema=spec["input_schema"],
                coroutine=coroutine,
            )
        )
    return tools


# The sub-agent. Structurally this sits exactly between the other two
# engines: it's a dict of configuration like the Agent SDK's
# AgentDefinition, but deepagents compiles it into a real LangGraph subgraph
# underneath, like app/graph/subgraphs/knowledge_agent.py.
KNOWLEDGE_SUBAGENT: SubAgent = {
    "name": "knowledge-agent",
    "description": (
        "Specialist for questions that need careful research or reasoning "
        "beyond a quick answer."
    ),
    "system_prompt": (
        "You are the Knowledge Agent, a research specialist consulted by "
        "another agent (the coordinator) — you are not talking to the end "
        "user directly. Answer thoroughly and precisely. Always end your "
        "answer with the exact line: -- Knowledge Agent v1"
    ),
    "tools": [],
}
