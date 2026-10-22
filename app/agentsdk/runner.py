"""AgentSdkTurnRunner — the same public surface as `GraphTurnRunner`
(app/graph/runner.py), with the Claude Agent SDK driving the turn instead of
a compiled LangGraph `StateGraph`.

It satisfies the identical `GraphRunner` Protocol, takes the identical DTOs,
and writes the identical Postgres transcript, so `app/main.py` can point a
second set of routes at it and nothing downstream notices. What changes is
everything *inside* one turn:

- There is no graph, no nodes, no router. `query()` owns the
  reason -> call tool -> feed result -> reason loop; we never see a
  `tool_use` block or hand-build a `tool_result` turn.
- Conversation state is a Claude Code *session* (a JSONL transcript the SDK
  writes to disk), resumed by `session_id`, not an `AgentState["messages"]`
  list restored by a checkpointer.
- The write-confirmation gate is a `can_use_tool` callback that blocks
  in-loop, not a graph interrupt that unwinds the whole run. That has a real
  consequence, and it's the most important difference to understand: the
  paused turn is a *live coroutine sitting in memory*, so this runner has to
  keep the driving task alive between the /run|/stream request and the
  /resume request (see `_LiveTurn`). LangGraph's `interrupt()` instead
  returns from `ainvoke` entirely and rebuilds the run from the checkpoint
  later.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolUseBlock,
    query,
)
from claude_agent_sdk.types import (
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from app.agentsdk.tools import (
    AUTO_APPROVED_TOOLS,
    KNOWLEDGE_AGENT,
    WRITE_TOOL_NAME,
    build_mcp_server,
    current_username,
)
from app.core.config import Settings
from app.graph.nodes.capsule_composition import _CAPSULES_DIR, _load_capsule_content
from app.graph.pg_memory import record_message, touch_chat
from app.graph.public.contracts import (
    ConfirmationRequest,
    PendingWriteDescriptor,
    StreamEvent,
    TurnRequest,
    TurnResult,
)

# The sub-agent name Claude passes as Agent(subagent_type=...).
KNOWLEDGE_AGENT_NAME = "knowledge-agent"

# Claude invokes a sub-agent through the built-in Agent tool. Older CLI
# builds emitted the same call under the name "Task" — match both.
_AGENT_TOOL_NAMES = frozenset({"Agent", "Task"})

_SENTINEL = object()


def _compose_system_prompt(role: str) -> str:
    """The capsule_composition node, minus the node. Same two YAML files,
    same concatenation — reused rather than reimplemented, since prompt
    composition isn't what changes between the two engines.
    """

    fragments = [
        _load_capsule_content(_CAPSULES_DIR / "core.yaml"),
        _load_capsule_content(_CAPSULES_DIR / "role" / f"{role}.yaml"),
    ]
    return "\n\n".join(fragment for fragment in fragments if fragment)


def _stage_label(block: ToolUseBlock) -> str:
    if block.name in _AGENT_TOOL_NAMES:
        return "Consulting knowledge agent"
    return f"Running tool: {block.name.rsplit('__', 1)[-1]}"


def _human_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
    return f"{tool_name.rsplit('__', 1)[-1]}({args})"


@dataclass
class _LiveTurn:
    """One in-flight turn. Holds the task actually running `query()`, the
    queue its events are published on, and (while paused) the future the
    `can_use_tool` callback is awaiting.

    This object is the Agent SDK equivalent of a LangGraph checkpoint — with
    the crucial difference that a checkpoint is *data* and this is a *running
    coroutine*. It cannot be serialized, shared across processes, or survive
    a restart.
    """

    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    decision: asyncio.Future | None = None


class AgentSdkTurnRunner:
    """Concrete GraphRunner backed by the Claude Agent SDK.

    Like GraphTurnRunner, this is also the single place every turn's
    user-visible transcript gets durably recorded — the routes and the tools
    stay unaware that persistence is happening.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mcp_server = build_mcp_server(settings)
        # thread_id -> Claude Code session id. The SDK's own transcript lives
        # on disk (~/.claude/projects/...), so only this mapping is in-memory;
        # persisting it would survive a restart, which InMemorySaver cannot.
        self._sessions: dict[str, str] = {}
        self._live: dict[str, _LiveTurn] = {}

    # ---------------------------------------------------------------- public

    async def run_turn(self, request: TurnRequest) -> TurnResult:
        return await _collect(self.stream_turn(request))

    async def resume_turn(self, request: ConfirmationRequest) -> TurnResult:
        return await _collect(self.resume_stream_turn(request))

    async def stream_turn(self, request: TurnRequest) -> AsyncIterator[StreamEvent]:
        await record_message(self._settings, request.username, request.thread_id, "user", request.message)
        await touch_chat(self._settings, request.username, request.thread_id, request.message)

        live = _LiveTurn()
        self._live[request.thread_id] = live
        live.task = asyncio.create_task(self._drive(request, live))

        async for event in self._drain(request.thread_id, request.username, live):
            yield event

    async def resume_stream_turn(
        self, request: ConfirmationRequest
    ) -> AsyncIterator[StreamEvent]:
        """Hand the decision to the `can_use_tool` callback that is still
        awaiting it, then keep streaming the *same* turn. Nothing is replayed:
        unlike a graph resume, the coroutine never stopped — it was parked on
        a future this line completes.
        """

        await record_message(
            self._settings,
            request.username,
            request.thread_id,
            "user",
            f"[decision: {request.decision}]",
        )

        live = self._live.get(request.thread_id)
        if live is None or live.decision is None or live.decision.done():
            yield StreamEvent(kind="error", error="No turn is awaiting confirmation on this thread.")
            return

        live.decision.set_result(request.decision)
        live.decision = None

        async for event in self._drain(request.thread_id, request.username, live):
            yield event

    # --------------------------------------------------------------- internal

    async def _drain(
        self, thread_id: str, username: str, live: _LiveTurn
    ) -> AsyncIterator[StreamEvent]:
        """Yield events until the turn either finishes or parks at a
        confirmation. The queue outlives this generator on an interrupt —
        that's what lets resume_stream_turn pick the same turn back up.
        """

        while True:
            event = await live.queue.get()
            if event is _SENTINEL:
                self._live.pop(thread_id, None)
                return
            if event.kind == "final":
                await record_message(self._settings, username, thread_id, "assistant", event.reply)
            yield event
            if event.kind == "interrupt":
                return
            if event.kind in ("final", "error"):
                self._live.pop(thread_id, None)
                return

    def _build_options(self, request: TurnRequest, live: _LiveTurn) -> ClaudeAgentOptions:
        async def can_use_tool(
            tool_name: str, input_data: dict[str, Any], _context: ToolPermissionContext
        ) -> PermissionResultAllow | PermissionResultDeny:
            """The human-in-the-loop gate. Reached only for tools missing
            from `allowed_tools` — i.e. send_alert — which is the Agent SDK
            counterpart of the graph router's write > consult > tool branch
            priority. Fails closed to "cancel", same as the graph node.
            """

            if tool_name != WRITE_TOOL_NAME:
                return PermissionResultAllow(updated_input=input_data)

            live.decision = asyncio.get_running_loop().create_future()
            await live.queue.put(
                StreamEvent(
                    kind="interrupt",
                    label="Awaiting confirmation",
                    node="can_use_tool",
                    pending_writes=(
                        PendingWriteDescriptor(
                            tool_use_id="",  # the SDK doesn't surface it to this callback
                            tool_name=tool_name.rsplit("__", 1)[-1],
                            human_summary=_human_summary(tool_name, input_data),
                        ),
                    ),
                )
            )
            decision = await live.decision
            if decision == "confirm":
                return PermissionResultAllow(updated_input=input_data)
            return PermissionResultDeny(message="Cancelled by the user — action was not performed.")

        # Documented workaround: a PreToolUse hook has to be registered for
        # can_use_tool to be consulted on a streaming-input query.
        async def _keepalive_hook(_input, _tool_use_id, _context):
            return {"continue_": True}

        return ClaudeAgentOptions(
            model=self._settings.anthropic_model,
            system_prompt=_compose_system_prompt("analyst"),
            mcp_servers={"learning": self._mcp_server},
            agents={KNOWLEDGE_AGENT_NAME: KNOWLEDGE_AGENT},
            # Drop every built-in tool except Agent: this is a chat
            # coordinator, not a coding agent — Claude gets no Read/Write/
            # Bash/WebSearch, only our MCP tools and the sub-agent.
            tools=["Agent"],
            allowed_tools=[*AUTO_APPROVED_TOOLS, "Agent"],
            permission_mode="default",
            can_use_tool=can_use_tool,
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_keepalive_hook])]},
            # Don't inherit this repo's CLAUDE.md / .claude settings — they
            # describe the project, not this agent.
            setting_sources=[],
            resume=self._sessions.get(request.thread_id),
            # The key lives in .env, which the spawned CLI doesn't read — hand
            # it over explicitly. Left unset, the CLI falls back to whatever
            # credentials it already has (`claude` login), which is fine too.
            env=(
                {"ANTHROPIC_API_KEY": self._settings.anthropic_api_key}
                if self._settings.anthropic_api_key
                else {}
            ),
            max_turns=12,
        )

    async def _drive(self, request: TurnRequest, live: _LiveTurn) -> None:
        """Run one turn and publish its events. This is the whole replacement
        for build.py's graph: the loop, the tool dispatch and the sub-agent
        fan-out all happen inside `query()`.
        """

        token = current_username.set(request.username)
        try:
            await live.queue.put(StreamEvent(kind="stage", label="Reasoning", node="coordinator"))

            async def prompt_stream():
                yield {
                    "type": "user",
                    "message": {"role": "user", "content": request.message},
                }

            async for message in query(
                prompt=prompt_stream(), options=self._build_options(request, live)
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            await live.queue.put(
                                StreamEvent(
                                    kind="stage", label=_stage_label(block), node=block.name
                                )
                            )
                elif isinstance(message, ResultMessage):
                    self._sessions[request.thread_id] = message.session_id
                    if message.subtype == "success":
                        await live.queue.put(
                            StreamEvent(
                                kind="final",
                                label="Reasoning",
                                node="coordinator",
                                reply=message.result or "",
                            )
                        )
                    else:
                        await live.queue.put(
                            StreamEvent(kind="error", error=f"Turn ended: {message.subtype}")
                        )
        except Exception as exc:  # noqa: BLE001 - keep the stream contract stable
            await live.queue.put(StreamEvent(kind="error", error=str(exc)))
        finally:
            current_username.reset(token)
            await live.queue.put(_SENTINEL)


async def _collect(stream: AsyncIterator[StreamEvent]) -> TurnResult:
    """Non-streaming view of a streamed turn — the run_turn/resume_turn
    counterpart of ainvoke, built on top of the same event sequence.
    """

    async for event in stream:
        if event.kind == "final":
            return TurnResult(status="completed", reply=event.reply)
        if event.kind == "interrupt":
            return TurnResult(status="awaiting_confirmation", pending_writes=event.pending_writes)
        if event.kind == "error":
            return TurnResult(status="completed", reply=f"Error: {event.error}")
    return TurnResult(status="completed", reply="")
