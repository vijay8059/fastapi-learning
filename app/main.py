import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.agents.plan_agent import run_plan_agent
from app.agentsdk.runner import AgentSdkTurnRunner
from app.deepagents_engine.runner import DeepAgentsTurnRunner
from app.core.auth import (
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    create_session,
    delete_session,
    ensure_auth_schema,
    get_current_user,
    resolve_session,
    verify_credentials,
)
from app.core.config import Settings, get_settings
from app.graph.build import build_graph
from app.graph.chunking import chunk_markdown
from app.graph.docling_client import convert_to_markdown
from app.graph.pg_memory import (
    ensure_learning_schema,
    list_attachments,
    list_chats,
    list_messages,
    record_attachment,
)
from app.graph.public.contracts import ConfirmationRequest, StreamEvent, TurnRequest, TurnResult
from app.graph.qdrant_memory import ensure_collection, index_chunks
from app.graph.runner import GraphTurnRunner


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    await ensure_learning_schema(settings)
    await ensure_auth_schema(settings)
    await ensure_collection(settings)
    yield


app = FastAPI(title=get_settings().app_name, lifespan=_lifespan)

_PLAYGROUND_HTML = Path(__file__).parent / "static" / "playground.html"

# Built once at import time, not per-request: the checkpointer holds paused
# threads in memory, so the same graph instance must serve both the initial
# /graph/run call and any later /graph/resume call for that thread_id.
#
# build_graph() is declared `-> object` (see graph/build.py) — deliberately
# vague rather than importing langgraph's compiled-graph type there. cast()
# here is a pure type-checker annotation, exactly like assist-workflows'
# `cast(AsyncGenerator[StreamEvent, None], service.stream_message(...))` in
# chat/api.py: it does NOT check or convert anything at runtime, it just
# tells the type checker "treat this as CompiledStateGraph from here on".
#
# Routes below never touch `_graph` directly — only GraphTurnRunner does
# (app/graph/runner.py), the same way assist-workflows' routes only ever
# call through the AssistRunner Protocol, never the compiled graph itself.
_checkpointer = InMemorySaver()
_graph = cast(CompiledStateGraph, build_graph(get_settings(), _checkpointer))
_runner = GraphTurnRunner(_graph, get_settings())

# The same turn, executed by the Claude Agent SDK instead of LangGraph (see
# app/agentsdk/). It implements the same GraphRunner Protocol and takes the
# same DTOs, so the /agentsdk/* routes below are copies of the /graph/* ones
# with only this object swapped — which is the whole point of the comparison.
_agentsdk_runner = AgentSdkTurnRunner(get_settings())

# And a third: deepagents, which is itself a LangGraph graph, so it shares
# the checkpointer type (its own instance, since thread_ids are per-engine).
_deepagents_runner = DeepAgentsTurnRunner(get_settings(), InMemorySaver())

# SSE keepalive — our slower branches (web_search's subprocess spawn,
# consult's separate LLM call) can take a few seconds with no node
# completing in between; without a keepalive, a proxy/load balancer sitting
# between client and server could decide the connection is idle and close it.
_STREAM_KEEPALIVE_SECONDS = 2.0
_STREAM_KEEPALIVE_FRAME = ": keepalive\n\n"  # SSE comment line — clients ignore it


class PlanRequest(BaseModel):
    message: str


class PlanResponse(BaseModel):
    reply: str


class LoginRequest(BaseModel):
    username: str
    password: str


def _turn_result_response(result: TurnResult) -> dict[str, object]:
    if result.status == "awaiting_confirmation":
        return {
            "status": "waiting_for_confirmation",
            "pending": {
                "kind": "write_confirmation",
                "pending_writes": [
                    {"tool_name": pw.tool_name, "human_summary": pw.human_summary}
                    for pw in result.pending_writes
                ],
            },
        }
    return {"status": "done", "reply": result.reply}


def _sse_frame(event: StreamEvent) -> str:
    """Serialize a StreamEvent as an SSE data: frame. Matches
    assist-workflows' chat/api.py _sse_frame: empty default fields elided
    to keep the payload compact.
    """

    payload: dict[str, object] = {"kind": event.kind}
    if event.label:
        payload["label"] = event.label
    if event.node:
        payload["node"] = event.node
    if event.reply:
        payload["reply"] = event.reply
    if event.pending_writes:
        payload["pending_writes"] = [
            {"tool_name": pw.tool_name, "human_summary": pw.human_summary}
            for pw in event.pending_writes
        ]
    if event.error:
        payload["error"] = event.error
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/playground", response_class=HTMLResponse)
def playground() -> Response:
    # no-store: this file is edited frequently during development and is
    # read fresh off disk on every request anyway — no reason for the
    # browser to ever serve a stale cached copy of it.
    return HTMLResponse(_PLAYGROUND_HTML.read_text(), headers={"Cache-Control": "no-store"})


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello from fastapi-learning"}


@app.get("/config")
def config(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str | bool]:
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "api_prefix": settings.api_prefix,
    }


@app.post("/login")
async def login(
    request: LoginRequest,
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    if not await verify_credentials(settings, request.username, request.password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    session_id = await create_session(settings, request.username)
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.environment != "local",
    )
    return {"username": request.username}


@app.post("/logout")
async def logout(
    request: Request,
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        await delete_session(settings, session_id)
    response.delete_cookie(SESSION_COOKIE)
    return {"status": "ok"}


@app.get("/me")
async def me(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> dict[str, object]:
    session_id = request.cookies.get(SESSION_COOKIE)
    username = await resolve_session(settings, session_id) if session_id else None
    return {"authenticated": username is not None, "username": username}


@app.post("/agent/plan")
async def agent_plan(
    request: PlanRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> PlanResponse:
    reply = await run_plan_agent(request.message, settings)
    return PlanResponse(reply=reply)


@app.post("/attachments")
async def upload_attachment(
    file: UploadFile,
    thread_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    username: Annotated[str, Depends(get_current_user)],
) -> dict[str, object]:
    """Parses an uploaded file via docling-serve, then splits storage the
    same way pg_memory/qdrant_memory split it for facts: the full converted
    document + metadata goes to Postgres (source of truth for "what was
    uploaded"), the chunked/embedded content goes to Qdrant (the searchable
    index `search_documents` queries against).
    """

    content = await file.read()
    markdown = await convert_to_markdown(settings, file.filename or "upload", content)
    chunks = chunk_markdown(markdown)

    attachment_id = await record_attachment(
        settings, username, thread_id, file.filename or "upload", markdown, len(chunks)
    )
    await index_chunks(settings, attachment_id, username, file.filename or "upload", thread_id, chunks)

    return {"attachment_id": attachment_id, "filename": file.filename, "chunk_count": len(chunks)}


@app.get("/attachments")
async def get_attachments(
    settings: Annotated[Settings, Depends(get_settings)],
    username: Annotated[str, Depends(get_current_user)],
) -> list[dict[str, object]]:
    """Lists what the logged-in user has uploaded — filename, chunk_count,
    when — without needing to ask the chat agent to go find something in
    them first. Content search stays search_documents' job; this is just
    the catalog.
    """

    rows = await list_attachments(settings, username)
    return [
        {
            "attachment_id": row["id"],
            "filename": row["filename"],
            "thread_id": row["thread_id"],
            "chunk_count": row["chunk_count"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


@app.get("/chats")
async def get_chats(
    settings: Annotated[Settings, Depends(get_settings)],
    username: Annotated[str, Depends(get_current_user)],
) -> list[dict[str, object]]:
    """Lists the logged-in user's chats, most recently active first — the
    sidebar's data source. A chat row is created lazily by the runner on
    its first message (see GraphTurnRunner.run_turn/stream_turn), not here.
    """

    rows = await list_chats(settings, username)
    return [
        {"thread_id": row["thread_id"], "title": row["title"], "updated_at": row["updated_at"].isoformat()}
        for row in rows
    ]


@app.get("/chats/{thread_id}/messages")
async def get_chat_messages(
    thread_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    username: Annotated[str, Depends(get_current_user)],
) -> list[dict[str, object]]:
    """Replays one chat's transcript for the sidebar's "reopen" flow.
    Scoped to the caller's own username, not just thread_id — a user can't
    fetch another user's chat by guessing its id.
    """

    rows = await list_messages(settings, username, thread_id)
    return [
        {"role": row["role"], "content": row["content"], "created_at": row["created_at"].isoformat()}
        for row in rows
    ]


@app.get("/graph/run")
async def graph_run(
    message: str, thread_id: str, username: Annotated[str, Depends(get_current_user)]
) -> dict[str, object]:
    result = await _runner.run_turn(TurnRequest(thread_id=thread_id, message=message, username=username))
    return _turn_result_response(result)


@app.get("/graph/resume")
async def graph_resume(
    thread_id: str, decision: str, username: Annotated[str, Depends(get_current_user)]
) -> dict[str, object]:
    result = await _runner.resume_turn(
        ConfirmationRequest(thread_id=thread_id, decision=decision, username=username)
    )
    return _turn_result_response(result)


def _stream_graph_events(
    stream: AsyncIterator[StreamEvent], http_request: Request
) -> StreamingResponse:
    """Shared SSE plumbing for /graph/stream and /graph/resume/stream — same
    keepalive/disconnect/error handling regardless of which GraphTurnRunner
    method produced `stream`. Mirrors assist-workflows' chat/api.py _wire():
    that function only ever consumes an AsyncIterator[StreamEvent] too, never
    the graph itself — this is the HTTP layer's job, not the runner's.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        pending_next: asyncio.Task[StreamEvent] | None = None
        try:
            while True:
                if pending_next is None:
                    pending_next = asyncio.create_task(anext(stream))
                done, _ = await asyncio.wait({pending_next}, timeout=_STREAM_KEEPALIVE_SECONDS)
                if not done:
                    if await http_request.is_disconnected():
                        return
                    yield _STREAM_KEEPALIVE_FRAME
                    continue
                try:
                    event = pending_next.result()
                except StopAsyncIteration:
                    pending_next = None
                    break
                except Exception as exc:  # noqa: BLE001 - keep the stream contract stable
                    pending_next = None
                    yield _sse_frame(StreamEvent(kind="error", error=str(exc)))
                    break
                pending_next = None
                yield _sse_frame(event)
            yield "event: done\ndata: {}\n\n"
        finally:
            if pending_next is not None and not pending_next.done():
                pending_next.cancel()
                with suppress(asyncio.CancelledError):
                    await pending_next
            with suppress(Exception):  # noqa: BLE001 - best-effort generator cleanup
                await stream.aclose()

    return StreamingResponse(
        event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.get("/graph/stream")
async def graph_stream(
    message: str,
    thread_id: str,
    http_request: Request,
    username: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    """Streams node-progress events over SSE as the graph runs — which stage
    is active right now, not token-by-token answer text. The final event
    carries the reply once the coordinator's terminal turn produces one.
    """

    stream = _runner.stream_turn(TurnRequest(thread_id=thread_id, message=message, username=username))
    return _stream_graph_events(stream, http_request)


@app.get("/graph/resume/stream")
async def graph_resume_stream(
    thread_id: str,
    decision: str,
    http_request: Request,
    username: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    """Same as /graph/stream, but resumes a paused thread instead of starting
    a fresh one — the confirm/cancel counterpart, so the coordinator's final
    synthesis after a write decision streams stage progress too.
    """

    stream = _runner.resume_stream_turn(
        ConfirmationRequest(thread_id=thread_id, decision=decision, username=username)
    )
    return _stream_graph_events(stream, http_request)


# --------------------------------------------------------------------------
# The same four routes again, served by the Claude Agent SDK engine instead
# of the LangGraph one. Identical request/response shapes — the only change
# is _agentsdk_runner in place of _runner, which is possible because both
# implement the GraphRunner Protocol from app/graph/public/contracts.py.
# --------------------------------------------------------------------------


@app.get("/agentsdk/run")
async def agentsdk_run(
    message: str, thread_id: str, username: Annotated[str, Depends(get_current_user)]
) -> dict[str, object]:
    result = await _agentsdk_runner.run_turn(
        TurnRequest(thread_id=thread_id, message=message, username=username)
    )
    return _turn_result_response(result)


@app.get("/agentsdk/resume")
async def agentsdk_resume(
    thread_id: str, decision: str, username: Annotated[str, Depends(get_current_user)]
) -> dict[str, object]:
    result = await _agentsdk_runner.resume_turn(
        ConfirmationRequest(thread_id=thread_id, decision=decision, username=username)
    )
    return _turn_result_response(result)


@app.get("/agentsdk/stream")
async def agentsdk_stream(
    message: str,
    thread_id: str,
    http_request: Request,
    username: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    stream = _agentsdk_runner.stream_turn(
        TurnRequest(thread_id=thread_id, message=message, username=username)
    )
    return _stream_graph_events(stream, http_request)


@app.get("/agentsdk/resume/stream")
async def agentsdk_resume_stream(
    thread_id: str,
    decision: str,
    http_request: Request,
    username: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    """Note the difference from /graph/resume/stream: here the paused turn is
    a coroutine still parked in memory on this worker, so this call hands it
    a decision and keeps streaming the *same* run. The graph version instead
    starts a fresh run from the checkpointer.
    """

    stream = _agentsdk_runner.resume_stream_turn(
        ConfirmationRequest(thread_id=thread_id, decision=decision, username=username)
    )
    return _stream_graph_events(stream, http_request)


# --------------------------------------------------------------------------
# Engine three: deepagents. Same four routes again, same swap — the only
# runner of the three that is both a configured harness *and* a LangGraph
# graph underneath.
# --------------------------------------------------------------------------


@app.get("/deepagents/run")
async def deepagents_run(
    message: str, thread_id: str, username: Annotated[str, Depends(get_current_user)]
) -> dict[str, object]:
    result = await _deepagents_runner.run_turn(
        TurnRequest(thread_id=thread_id, message=message, username=username)
    )
    return _turn_result_response(result)


@app.get("/deepagents/resume")
async def deepagents_resume(
    thread_id: str, decision: str, username: Annotated[str, Depends(get_current_user)]
) -> dict[str, object]:
    result = await _deepagents_runner.resume_turn(
        ConfirmationRequest(thread_id=thread_id, decision=decision, username=username)
    )
    return _turn_result_response(result)


@app.get("/deepagents/stream")
async def deepagents_stream(
    message: str,
    thread_id: str,
    http_request: Request,
    username: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    stream = _deepagents_runner.stream_turn(
        TurnRequest(thread_id=thread_id, message=message, username=username)
    )
    return _stream_graph_events(stream, http_request)


@app.get("/deepagents/resume/stream")
async def deepagents_resume_stream(
    thread_id: str,
    decision: str,
    http_request: Request,
    username: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    stream = _deepagents_runner.resume_stream_turn(
        ConfirmationRequest(thread_id=thread_id, decision=decision, username=username)
    )
    return _stream_graph_events(stream, http_request)
