"""GraphTurnRunner — the concrete GraphRunner.

Mirrors assist-workflows' RunAssistTurnUseCase (app/modules/assist/use_cases/
run_assist_turn.py): the ONLY thing that actually calls the compiled graph's
ainvoke/astream. main.py's routes call this class through the GraphRunner
Protocol; nothing outside this file touches _graph directly.
"""

from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.core.config import Settings
from app.graph.pg_memory import record_message, touch_chat
from app.graph.public.contracts import (
    ConfirmationRequest,
    PendingWriteDescriptor,
    StreamEvent,
    TurnRequest,
    TurnResult,
)
from app.graph.streaming import STAGE_LABELS


def _pending_writes_from_interrupt_payload(payload: dict[str, Any]) -> tuple[PendingWriteDescriptor, ...]:
    return tuple(
        PendingWriteDescriptor(
            tool_use_id=block["id"],
            tool_name=block["name"],
            human_summary=(
                f"{block['name']}("
                f"{', '.join(f'{k}={v!r}' for k, v in block.get('input', {}).items())})"
            ),
        )
        for block in payload.get("pending_writes", [])
    )


def _turn_result_from_state(result: dict[str, Any]) -> TurnResult:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return TurnResult(
            status="awaiting_confirmation",
            pending_writes=_pending_writes_from_interrupt_payload(payload),
        )
    return TurnResult(status="completed", reply=result["reply"])


def _stream_event_from_chunk(chunk: dict[str, Any]) -> StreamEvent:
    if "__interrupt__" in chunk:
        payload = chunk["__interrupt__"][0].value
        return StreamEvent(
            kind="interrupt",
            label="Awaiting confirmation",
            node="confirmation_interrupt",
            pending_writes=_pending_writes_from_interrupt_payload(payload),
        )

    node_name, node_output = next(iter(chunk.items()))
    label = STAGE_LABELS.get(node_name, node_name)
    if "reply" in node_output:
        return StreamEvent(kind="final", label=label, node=node_name, reply=node_output["reply"])
    return StreamEvent(kind="stage", label=label, node=node_name)


class GraphTurnRunner:
    """Concrete GraphRunner wrapping one compiled graph instance.

    Also the single point where every turn's user-visible transcript is
    durably recorded (see app/graph/pg_memory.py's learning.messages) —
    main.py's routes and the graph nodes never need to know persistence is
    happening.
    """

    def __init__(self, graph: CompiledStateGraph, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    async def run_turn(self, request: TurnRequest) -> TurnResult:
        await record_message(self._settings, request.username, request.thread_id, "user", request.message)
        await touch_chat(self._settings, request.username, request.thread_id, request.message)

        config = {"configurable": {"thread_id": request.thread_id}}
        result = await self._graph.ainvoke(
            {"message": request.message, "username": request.username}, config=config
        )
        turn_result = _turn_result_from_state(result)

        if turn_result.status == "completed":
            await record_message(
                self._settings, request.username, request.thread_id, "assistant", turn_result.reply
            )
        return turn_result

    async def resume_turn(self, request: ConfirmationRequest) -> TurnResult:
        await record_message(
            self._settings,
            request.username,
            request.thread_id,
            "user",
            f"[decision: {request.decision}]",
        )

        config = {"configurable": {"thread_id": request.thread_id}}
        result = await self._graph.ainvoke(Command(resume=request.decision), config=config)
        turn_result = _turn_result_from_state(result)

        if turn_result.status == "completed":
            await record_message(
                self._settings, request.username, request.thread_id, "assistant", turn_result.reply
            )
        return turn_result

    async def stream_turn(self, request: TurnRequest) -> AsyncIterator[StreamEvent]:
        await record_message(self._settings, request.username, request.thread_id, "user", request.message)
        await touch_chat(self._settings, request.username, request.thread_id, request.message)

        config = {"configurable": {"thread_id": request.thread_id}}
        stream = self._graph.astream(
            {"message": request.message, "username": request.username},
            config=config,
            stream_mode="updates",
        )
        async for chunk in stream:
            event = _stream_event_from_chunk(chunk)
            if event.kind == "final":
                await record_message(
                    self._settings, request.username, request.thread_id, "assistant", event.reply
                )
            yield event

    async def resume_stream_turn(self, request: ConfirmationRequest) -> AsyncIterator[StreamEvent]:
        await record_message(
            self._settings,
            request.username,
            request.thread_id,
            "user",
            f"[decision: {request.decision}]",
        )

        config = {"configurable": {"thread_id": request.thread_id}}
        stream = self._graph.astream(
            Command(resume=request.decision), config=config, stream_mode="updates"
        )
        async for chunk in stream:
            event = _stream_event_from_chunk(chunk)
            if event.kind == "final":
                await record_message(
                    self._settings, request.username, request.thread_id, "assistant", event.reply
                )
            yield event
