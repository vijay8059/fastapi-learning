"""DeepAgentsTurnRunner — the same public surface as `GraphTurnRunner` and
`AgentSdkTurnRunner`, with a `deepagents` agent driving the turn.

The thing to notice is how much of `app/graph/runner.py` survives here: the
agent `create_deep_agent()` returns *is* a `CompiledStateGraph`, so this
runner still calls `ainvoke`/`astream`, still passes
`{"configurable": {"thread_id": ...}}`, still reads `__interrupt__` out of
the stream, and still resumes with `Command(resume=...)`. What disappears is
everything *inside* the graph — nodes, edges, the router, the interrupt node,
the consult node — replaced by four keyword arguments in `_build_agent`.

So this is the middle position between the other two engines: the Agent
SDK's "configure a harness" ergonomics, but the harness is a LangGraph graph
you keep full checkpointer/interrupt/streaming control over.
"""

from collections.abc import AsyncIterator
from typing import Any

from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from app.core.config import Settings
from app.deepagents_engine.tools import (
    KNOWLEDGE_SUBAGENT,
    WRITE_TOOL_NAME,
    build_tools,
    current_username,
)
from app.deepagents_engine.tracing import TraceMiddleware
from app.graph.nodes.capsule_composition import _CAPSULES_DIR, _load_capsule_content
from app.graph.pg_memory import record_message, touch_chat
from app.graph.public.contracts import (
    ConfirmationRequest,
    PendingWriteDescriptor,
    StreamEvent,
    TurnRequest,
    TurnResult,
)

# deepagents' own node names, mapped to the same user-facing vocabulary the
# hand-built graph uses (app/graph/streaming.py). There are only two nodes
# now — the ReAct pair — because planning, the filesystem and sub-agent
# dispatch are all *tools* inside them rather than nodes beside them.
_STAGE_LABELS = {"model": "Reasoning", "tools": "Running tool"}

# Tools deepagents adds for free, labelled so the stream reads sensibly.
_BUILTIN_LABELS = {
    "write_todos": "Planning",  # only if TodoListMiddleware is added
    "task": "Consulting knowledge agent",
    "ls": "Reading scratch files",
    "read_file": "Reading scratch files",
    "write_file": "Writing scratch file",
    "edit_file": "Editing scratch file",
    "glob": "Searching scratch files",
    "grep": "Searching scratch files",
}


def _compose_system_prompt(role: str) -> str:
    """Identical to the other two engines — same capsules, same order."""

    fragments = [
        _load_capsule_content(_CAPSULES_DIR / "core.yaml"),
        _load_capsule_content(_CAPSULES_DIR / "role" / f"{role}.yaml"),
    ]
    return "\n\n".join(fragment for fragment in fragments if fragment)


def _tool_stage_label(tool_name: str) -> str:
    if tool_name in _BUILTIN_LABELS:
        return _BUILTIN_LABELS[tool_name]
    return f"Running tool: {tool_name}"


def _pending_writes(interrupt_value: Any) -> tuple[PendingWriteDescriptor, ...]:
    """deepagents' interrupt payload is `{"action_requests": [...]}` rather
    than the dict our own confirmation_interrupt node built, so it gets
    translated into the same public DTO here — the boundary doesn't change
    just because the engine did.
    """

    requests = (interrupt_value or {}).get("action_requests", []) if isinstance(interrupt_value, dict) else []
    return tuple(
        PendingWriteDescriptor(
            tool_use_id=str(req.get("id") or ""),
            tool_name=req.get("name", ""),
            human_summary=(
                f"{req.get('name', '')}("
                f"{', '.join(f'{k}={v!r}' for k, v in (req.get('args') or {}).items())})"
            ),
        )
        for req in requests
    )


class DeepAgentsTurnRunner:
    """Concrete GraphRunner backed by a deepagents agent."""

    def __init__(self, settings: Settings, checkpointer: BaseCheckpointSaver) -> None:
        self._settings = settings
        self._agent = self._build_agent(settings, checkpointer)

    @staticmethod
    def _build_agent(settings: Settings, checkpointer: BaseCheckpointSaver) -> Any:
        """The whole graph, as four keyword arguments.

        - `tools` replaces tool_execution_node's dispatch table.
        - `subagents` replaces consult_node + the compiled knowledge subgraph.
        - `interrupt_on` replaces the router's write branch *and* the entire
          confirmation_interrupt node — deepagents inserts a
          HumanInTheLoopMiddleware that raises the same `interrupt()` we
          used to call by hand.
        - `checkpointer` is the same object the hand-built graph compiles with.

        What you get on top, unasked: a virtual filesystem (ls/read_file/
        write_file/edit_file/delete/glob/grep/execute) living in graph state
        for offloading long content, the `task` tool for sub-agent dispatch,
        and Anthropic prompt caching — nine extra tool schemas on top of our
        six. `write_todos` is NOT included: TodoListMiddleware is opt-in in
        0.7.17, so add it to `middleware=[...]` if you want planning.
        """

        return create_deep_agent(
            model=ChatAnthropic(
                model_name=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
                max_tokens=4096,
                timeout=None,
                stop=None,
            ),
            tools=build_tools(settings),
            system_prompt=_compose_system_prompt("analyst"),
            subagents=[KNOWLEDGE_SUBAGENT],
            interrupt_on={WRITE_TOOL_NAME: {"allowed_decisions": ["approve", "reject"]}},
            checkpointer=checkpointer,
            # The fifth argument, and the one that makes the other four
            # debuggable: middleware is the only seam this harness offers for
            # stepping through a turn, since none of the nodes are ours.
            middleware=[TraceMiddleware()] if settings.deepagents_trace else [],
        )

    # ---------------------------------------------------------------- public

    async def run_turn(self, request: TurnRequest) -> TurnResult:
        return await _collect(self.stream_turn(request))

    async def resume_turn(self, request: ConfirmationRequest) -> TurnResult:
        return await _collect(self.resume_stream_turn(request))

    async def stream_turn(self, request: TurnRequest) -> AsyncIterator[StreamEvent]:
        await record_message(self._settings, request.username, request.thread_id, "user", request.message)
        await touch_chat(self._settings, request.username, request.thread_id, request.message)

        payload = {"messages": [{"role": "user", "content": request.message}]}
        async for event in self._run(payload, request.thread_id, request.username):
            yield event

    async def resume_stream_turn(
        self, request: ConfirmationRequest
    ) -> AsyncIterator[StreamEvent]:
        """A real graph resume, exactly like the LangGraph engine's — the
        previous run already returned and this one is rebuilt from the
        checkpoint. Only the payload shape differs: deepagents wants a list
        of decisions, one per pending action, instead of our bare
        "confirm"/"cancel" string. Fails closed to reject.
        """

        await record_message(
            self._settings,
            request.username,
            request.thread_id,
            "user",
            f"[decision: {request.decision}]",
        )

        decision = (
            {"type": "approve"}
            if request.decision == "confirm"
            else {"type": "reject", "message": "Cancelled by the user — action was not performed."}
        )
        command = Command(resume={"decisions": [decision]})
        async for event in self._run(command, request.thread_id, request.username):
            yield event

    # --------------------------------------------------------------- internal

    async def _run(
        self, payload: Any, thread_id: str, username: str
    ) -> AsyncIterator[StreamEvent]:
        """Translate deepagents' update stream into our StreamEvents.

        Note what this costs compared to the hand-built graph: there, one
        update *was* one named stage. Here both real work and planning
        happen inside a single `model`/`tools` pair, so the useful labels
        have to be recovered from the tool calls inside each update.
        """

        token = current_username.set(username)
        config = {"configurable": {"thread_id": thread_id}}
        reply = ""
        try:
            async for chunk in self._agent.astream(payload, config=config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        value = update[0].value if isinstance(update, (list, tuple)) else update
                        yield StreamEvent(
                            kind="interrupt",
                            label="Awaiting confirmation",
                            node="interrupt_on",
                            pending_writes=_pending_writes(value),
                        )
                        return

                    messages = (update or {}).get("messages", []) if isinstance(update, dict) else []
                    for message in messages:
                        if not isinstance(message, AIMessage):
                            continue
                        if message.tool_calls:
                            for call in message.tool_calls:
                                yield StreamEvent(
                                    kind="stage",
                                    label=_tool_stage_label(call["name"]),
                                    node=node,
                                )
                        else:
                            reply = _text_of(message) or reply
                            yield StreamEvent(
                                kind="stage", label=_STAGE_LABELS.get(node) or str(node), node=node
                            )
        except Exception as exc:  # noqa: BLE001 - keep the stream contract stable
            yield StreamEvent(kind="error", error=str(exc))
            return
        finally:
            current_username.reset(token)

        await record_message(self._settings, username, thread_id, "assistant", reply)
        yield StreamEvent(kind="final", label="Reasoning", node="model", reply=reply)


def _text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


async def _collect(stream: AsyncIterator[StreamEvent]) -> TurnResult:
    async for event in stream:
        if event.kind == "final":
            return TurnResult(status="completed", reply=event.reply)
        if event.kind == "interrupt":
            return TurnResult(status="awaiting_confirmation", pending_writes=event.pending_writes)
        if event.kind == "error":
            return TurnResult(status="completed", reply=f"Error: {event.error}")
    return TurnResult(status="completed", reply="")
