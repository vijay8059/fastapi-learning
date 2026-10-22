"""The same tools as `app/graph/tools.py`, re-exposed the way the Claude
Agent SDK wants them: as an *in-process MCP server*.

Nothing here redefines a tool. The schemas come from `TOOLS` and the bodies
from `build_tool_implementations()` — this module only adapts the calling
convention:

- LangGraph side: we read `tool_use` blocks ourselves and call
  `implementations[name](tool_input, username)` from `tool_execution_node`.
- Agent SDK side: the SDK owns the loop, so it calls the handler for us.
  A handler only ever receives `args` — there is no seam to pass `username`
  through — so the per-turn username rides on a ContextVar instead.

`consult_knowledge_agent` is intentionally absent: a sub-agent is a
first-class concept in the Agent SDK (`AgentDefinition` + the built-in
`Agent` tool), so it stops being a tool we implement. See
KNOWLEDGE_AGENT below.
"""

from contextvars import ContextVar
from typing import Any

from claude_agent_sdk import AgentDefinition, create_sdk_mcp_server, tool

from app.core.config import Settings
from app.graph.tools import CONSULT_TOOL_NAMES, TOOLS, build_tool_implementations

# The name the MCP server is registered under in ClaudeAgentOptions.mcp_servers.
# Claude sees each tool as "mcp__{MCP_SERVER_NAME}__{tool_name}".
MCP_SERVER_NAME = "learning"

# Set by the runner at the start of every turn. The Agent SDK calls tool
# handlers itself, so this is the only way `search_documents` can still scope
# its Qdrant query to the caller (the graph engine passed it as an argument).
current_username: ContextVar[str] = ContextVar("current_username", default="")


def qualified(tool_name: str) -> str:
    """"remember_fact" -> "mcp__learning__remember_fact"."""

    return f"mcp__{MCP_SERVER_NAME}__{tool_name}"


async def _send_alert(args: dict[str, Any]) -> dict[str, Any]:
    """The write tool. In the graph engine its body lived inside
    `confirmation_interrupt_node`, because that node both gated *and*
    performed the action. Here the gate moved out to the runner's
    `can_use_tool` callback, so the tool is just a tool again — by the time
    this runs, the user has already confirmed.
    """

    return {"content": [{"type": "text", "text": f"Alert sent: {args['message']}"}]}


def build_mcp_server(settings: Settings) -> Any:
    """Wraps every plain tool from `app/graph/tools.py` — plus `send_alert` —
    in one in-process MCP server. In-process means no subprocess and no
    socket: the SDK calls these coroutines directly.
    """

    implementations = build_tool_implementations(settings)

    def _adapt(name: str):
        """Bridge (tool_input, username) -> str  to  (args) -> MCP result."""

        impl = implementations[name]

        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            text = await impl(args, current_username.get())
            return {"content": [{"type": "text", "text": text}]}

        return handler

    sdk_tools = []
    for spec in TOOLS:
        name = spec["name"]
        if name in CONSULT_TOOL_NAMES:
            continue  # now an AgentDefinition, not a tool — see KNOWLEDGE_AGENT
        handler = _send_alert if name == "send_alert" else _adapt(name)
        # @tool used as a plain call rather than a decorator, so the schemas
        # stay single-sourced in TOOLS instead of being retyped here.
        sdk_tools.append(tool(name, spec["description"], spec["input_schema"])(handler))

    return create_sdk_mcp_server(name=MCP_SERVER_NAME, version="1.0.0", tools=sdk_tools)


# Every tool except send_alert is pre-approved. send_alert is deliberately
# left off this list: an un-approved tool falls through to the runner's
# can_use_tool callback, which is how the human confirmation gate is built
# here (the graph engine did it with a routing branch + interrupt() instead).
AUTO_APPROVED_TOOLS = [
    qualified(spec["name"])
    for spec in TOOLS
    if spec["name"] not in CONSULT_TOOL_NAMES and spec["name"] != "send_alert"
]

WRITE_TOOL_NAME = qualified("send_alert")

# The sub-agent, expressed as configuration rather than as a second compiled
# graph (app/graph/subgraphs/knowledge_agent.py). The SDK spawns it in its own
# context window when Claude calls the built-in Agent tool; only its final
# message comes back as the tool result.
KNOWLEDGE_AGENT = AgentDefinition(
    description=(
        "Specialist for questions that need careful research or reasoning "
        "beyond a quick answer. Use instead of answering directly when the "
        "question warrants it."
    ),
    prompt=(
        "You are the Knowledge Agent, a research specialist consulted by "
        "another agent (the coordinator) — you are not talking to the end "
        "user directly. Answer thoroughly and precisely. Always end your "
        "answer with the exact line: -- Knowledge Agent v1"
    ),
    tools=[],
)
