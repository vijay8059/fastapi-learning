from collections.abc import Awaitable, Callable

from app.core.config import Settings
from app.graph.mcp_client import mcp_web_search
from app.graph.pg_memory import recall, remember
from app.graph.qdrant_memory import search as qdrant_search

# Shown to Claude as one combined list — it doesn't need to know which of
# these are plain tools vs sub-agent consults. That distinction only matters
# to *our* router, once Claude has already decided to call one of them.
TOOLS: list[dict] = [
    {
        "name": "get_secret_number",
        "description": "Returns today's secret number.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "consult_knowledge_agent",
        "description": (
            "Consult the Knowledge Agent, a specialist sub-agent for questions "
            "that need careful research or reasoning beyond a quick answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Searches the public web via DuckDuckGo and returns matching "
            "results. Use for anything live or time-sensitive — current "
            "events, weather, sports scores, prices — or anything else "
            "outside your own knowledge. Prefer calling this over saying "
            "you lack real-time access."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "remember_fact",
        "description": (
            "Persists a fact under a short key, in Postgres, so it can be "
            "recalled later — including from a completely different "
            "conversation thread. Use for anything the user says to "
            "remember, or that seems worth keeping past this one turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short identifier, e.g. 'favorite_color'."},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall_fact",
        "description": "Retrieves a previously remembered fact by its key.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Searches previously uploaded attachments (any document dropped "
            "into the playground, from any conversation) by meaning, and "
            "returns the most relevant excerpts. Use this whenever the user "
            "asks about something that might be in an uploaded file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "send_alert",
        "description": (
            "Sends an alert message to the team channel. Call this directly "
            "whenever the user asks you to send an alert — a confirmation "
            "step happens automatically before it takes effect, so you do "
            "not need to ask the user yourself first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
]

# Tool names that are sub-agent consults, not plain tools — the router reads
# this to decide between the "consult" and "tool_execution" branches.
CONSULT_TOOL_NAMES = frozenset({"consult_knowledge_agent"})

# Tool names that are write / side-effecting actions — the router sends these
# to the confirmation_interrupt branch instead of running them directly.
WRITE_TOOL_NAMES = frozenset({"send_alert"})


async def _get_secret_number(_input: dict, _username: str) -> str:
    # Hardcoded and unguessable on purpose — proves the tool actually ran,
    # rather than Claude just answering from its own knowledge.
    return "7392"


def build_tool_implementations(settings: Settings) -> dict[str, Callable[[dict, str], Awaitable[str]]]:
    """Plain-tool implementations only — consult_knowledge_agent moved to
    nodes/consult.py now that it has its own routing branch.

    Every implementation takes (tool_input, username) — most ignore the
    latter, but search_documents needs it to scope results to the caller's
    own uploads (see qdrant_memory.search), so the shared call signature
    carries it uniformly rather than special-casing one tool.
    """

    async def _web_search(tool_input: dict, _username: str) -> str:
        return await mcp_web_search(tool_input["query"])

    async def _remember_fact(tool_input: dict, _username: str) -> str:
        return await remember(settings, tool_input["key"], tool_input["value"])

    async def _recall_fact(tool_input: dict, _username: str) -> str:
        return await recall(settings, tool_input["key"])

    async def _search_documents(tool_input: dict, username: str) -> str:
        hits = await qdrant_search(settings, username, tool_input["query"])
        if not hits:
            return "No matching content found in any uploaded document."
        return "\n\n".join(f"[{hit['filename']}] {hit['text']}" for hit in hits)

    return {
        "get_secret_number": _get_secret_number,
        "web_search": _web_search,
        "remember_fact": _remember_fact,
        "recall_fact": _recall_fact,
        "search_documents": _search_documents,
    }
