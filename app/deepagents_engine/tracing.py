"""TraceMiddleware — a breakpoint surface for the deepagents engine.

The problem this solves: deepagents owns the loop, so there is no node of
yours to step through. Every interesting moment happens inside library code
you did not write. This middleware puts one of your own methods at each of
those moments, so the whole turn becomes debuggable line by line.

**How to debug a turn**

1. Run the "FastAPI: fastapi-learning (uvicorn)" config in `.vscode/launch.json`
   (it already sets `justMyCode: false`, so you can also step *into*
   langchain/deepagents from here).
2. Set a breakpoint on any line marked ``# BREAKPOINT`` below.
3. Hit `/deepagents/stream?message=...&thread_id=t1` (or use `/playground`
   with the prefix changed).

Each hook prints a line too, so you can follow a turn without stopping it:

    [trace] turn start | 1 message(s)
    [trace] -> model | 1 message(s) in context
    [trace] <- claude-sonnet-5 | 1.84s | 4987 in / 43 out
    [trace] model chose: get_secret_number
    [trace]     > get_secret_number {}
    [trace]     < get_secret_number | 0.00s | 7392
    [trace] -> model | 3 message(s) in context
    [trace] <- claude-sonnet-5 | 1.29s | 5037 in / 14 out
    [trace] model answered (no tool calls)
    [trace] turn end

**Which hooks become nodes**

``before_*``/``after_*`` become real graph nodes named ``TraceMiddleware.<hook>``
— you will see them in ``agent.get_graph()``. ``wrap_*`` hooks do not; they
wrap execution inside the existing ``model``/``tools`` nodes.

Only the async (``a``-prefixed) variants are implemented, because this app
always drives the graph with ``astream``/``ainvoke``. Add the sync twins if
you ever invoke the agent synchronously.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

logger = logging.getLogger("app.deepagents_engine.trace")

# How much of a tool result to echo into the log line.
_PREVIEW = 120


def _preview(value: Any) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= _PREVIEW else text[:_PREVIEW] + "..."


class TraceMiddleware(AgentMiddleware):
    """Logs every decision point in the loop, and gives each one a breakpoint.

    Every hook returns ``None``, which means "change nothing" — this is
    deliberately an observer. Returning a state dict instead would let you
    rewrite messages mid-turn; see `awrap_tool_call` for where you would
    intercept rather than observe.
    """

    # `AgentMiddleware.name` already defaults to the class name, which is what
    # names the graph nodes ("TraceMiddleware.after_model"). Override it only
    # if you need two instances — the names must be unique or the build fails.

    # --- turn boundaries (each becomes a graph node) ----------------------

    async def abefore_agent(self, state, runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        logger.info("[trace] turn start | %d message(s)", len(messages))
        return None  # BREAKPOINT — state as the turn begins

    async def aafter_agent(self, state, runtime) -> dict[str, Any] | None:
        logger.info("[trace] turn end")
        return None  # BREAKPOINT — final state, including the reply

    # --- around each model call -------------------------------------------

    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        logger.info("[trace] -> model | %d message(s) in context", len(messages))
        return None  # BREAKPOINT — exactly what the model is about to be sent

    async def awrap_model_call(self, request, handler: Callable[[Any], Awaitable[Any]]):
        """Wraps the LLM call itself. `request` carries the messages, system
        prompt and bound tools — inspect it here to see the *real* prompt,
        including everything the harness injected.
        """

        started = time.perf_counter()
        response = await handler(request)  # BREAKPOINT — step into the model call
        elapsed = time.perf_counter() - started

        usage: Any = {}
        model_name = "?"
        message = getattr(response, "result", None) or response
        if isinstance(message, list):
            message = next((m for m in message if isinstance(m, AIMessage)), None)
        if isinstance(message, AIMessage):
            usage = message.usage_metadata or {}
            metadata = message.response_metadata or {}
            # Which model actually answered — useful when sub-agents override it.
            model_name = metadata.get("model_name") or metadata.get("model") or "?"
        logger.info(
            "[trace] <- %s | %.2fs | %s in / %s out",
            model_name,
            elapsed,
            usage.get("input_tokens", "?"),
            usage.get("output_tokens", "?"),
        )
        return response

    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        """The same position `HumanInTheLoopMiddleware.after_model` occupies —
        the model has chosen, nothing has executed yet. This is where you can
        see the routing decision before it happens.
        """

        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            names = [call["name"] for call in last.tool_calls]
            logger.info("[trace] model chose: %s", ", ".join(names))
        else:
            logger.info("[trace] model answered (no tool calls)")
        return None  # BREAKPOINT — inspect last.tool_calls to see the decision

    # --- around each tool call (no node; wraps the tools node) ------------

    async def awrap_tool_call(self, request, handler: Callable[[Any], Awaitable[Any]]):
        """Fires once per tool call, including `task` (sub-agent dispatch) and
        the filesystem tools.

        This is the seam that replaces the hand-built graph's separate
        `consult` / `tool_execution` branches: branch on
        `request.tool_call["name"]` to treat a sub-agent call differently.
        Returning a `ToolMessage` from here *without* calling `handler` skips
        execution entirely — that's how you'd short-circuit or cache.
        """

        call = request.tool_call
        name = call["name"]
        if name == "task":
            logger.info(
                "[trace]     > sub-agent %s", (call.get("args") or {}).get("subagent_type", "?")
            )
        else:
            logger.info("[trace]     > %s %s", name, _preview(call.get("args") or {}))

        started = time.perf_counter()
        result = await handler(request)  # BREAKPOINT — step into the tool
        elapsed = time.perf_counter() - started

        content = getattr(result, "content", result)
        logger.info("[trace]     < %s | %.2fs | %s", name, elapsed, _preview(content))
        return result
