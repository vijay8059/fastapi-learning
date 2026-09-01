"""Cross-module contracts exposed by the graph package.

Mirrors assist-workflows' app/modules/assist/public/contracts.py: main.py's
route handlers target this Protocol and these DTOs, never the compiled graph
or its ainvoke/astream calls directly. The DTOs are self-contained
(primitive fields only) — they don't leak internal shapes like the raw
Anthropic tool_use block across the boundary.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

TurnStatus = Literal["completed", "awaiting_confirmation"]
ConfirmationDecision = Literal["confirm", "cancel"]


@dataclass(frozen=True)
class PendingWriteDescriptor:
    """Public-facing description of a write-class action awaiting confirmation.

    ``human_summary`` is what the user reads; the raw tool_use ``input``
    dict stays internal so the public surface doesn't leak tool-argument
    shape (same reasoning as assist-workflows' own PendingWriteDescriptor).
    """

    tool_use_id: str
    tool_name: str
    human_summary: str


@dataclass(frozen=True)
class TurnRequest:
    """Public input for GraphRunner.run_turn / stream_turn."""

    thread_id: str
    message: str
    username: str


@dataclass(frozen=True)
class ConfirmationRequest:
    """Public input for GraphRunner.resume_turn / resume_stream_turn."""

    thread_id: str
    decision: ConfirmationDecision
    username: str


@dataclass(frozen=True)
class TurnResult:
    """Public output of one graph turn.

    ``status == "awaiting_confirmation"`` means the graph paused at a
    write-tool interrupt; the caller surfaces ``pending_writes`` to the UI
    and must later call ``resume_turn``.
    """

    status: TurnStatus
    reply: str = ""
    pending_writes: tuple[PendingWriteDescriptor, ...] = ()


StreamEventKind = Literal["stage", "interrupt", "final", "error"]


@dataclass(frozen=True)
class StreamEvent:
    """One user-facing event emitted during a streamed turn.

    Field semantics by ``kind``:
    - ``stage``: ``label`` is the user-facing stage description; ``node``
      is the underlying graph-node name.
    - ``interrupt``: ``label`` summarizes the pause; ``pending_writes``
      mirrors ``TurnResult.pending_writes``.
    - ``final``: ``reply`` carries the completed assistant answer.
    - ``error``: ``error`` is a user-safe error string. Terminal.
    """

    kind: StreamEventKind
    label: str = ""
    node: str = ""
    reply: str = ""
    pending_writes: tuple[PendingWriteDescriptor, ...] = field(default_factory=tuple)
    error: str = ""


@runtime_checkable
class GraphRunner(Protocol):
    """Public boundary for invoking the graph — main.py's routes target
    this, never the compiled graph or its ainvoke/astream calls directly.
    """

    async def run_turn(self, request: TurnRequest) -> TurnResult:
        """Execute one turn end-to-end: completed reply or an
        awaiting-confirmation marker for a write action."""
        ...

    async def resume_turn(self, request: ConfirmationRequest) -> TurnResult:
        """Resume a turn paused at a confirmation interrupt."""
        ...

    def stream_turn(self, request: TurnRequest) -> AsyncIterator[StreamEvent]:
        """Run one turn and stream user-facing events as they happen."""
        ...

    def resume_stream_turn(self, request: ConfirmationRequest) -> AsyncIterator[StreamEvent]:
        """Resume a paused turn and stream its remaining events."""
        ...
