# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI + LangGraph learning project. The point of the repo is to build a small but structurally real multi-agent chat system — many docstrings and comments explicitly say a given piece "mirrors assist-workflows" (an internal reference architecture this project is deliberately modeled after at learning scale). When a comment references "assist-workflows," treat it as intentional design provenance, not a stray leftover.

## Commands

Package manager is `uv` (`uv.lock` + `.python-version` present, no `requirements.txt`).

```bash
uv sync                      # install/update dependencies from pyproject.toml
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

(`.vscode/launch.json` runs the same module without `--reload`, as a debugpy config.)

Config comes from `.env` (copy `.env.example` — needs at least `ANTHROPIC_API_KEY`; Postgres/Qdrant/docling defaults assume the local services below).

**Required running services before startup** — `app/main.py`'s `_lifespan` calls `ensure_learning_schema`/`ensure_auth_schema` (Postgres) and `ensure_collection` (Qdrant) at startup and will fail to boot if these aren't reachable:
- **Postgres** on `localhost:5432` — a native local service, not managed by this repo.
- **Qdrant** — `docker start qdrant` (or `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest` if the container doesn't exist).
- **docling-serve** — `docker start docling-serve` (or `docker run -p 5001:5001 quay.io/docling-project/docling-serve-cpu:latest`).

**No test suite exists** — no `tests/` directory, no pytest/ruff/mypy config. Changes so far have been verified with `python -m py_compile <changed files>` plus manually exercising `/playground` in a browser.

## Architecture

### The coordinator loop (the core of the app)

`app/graph/build.py` compiles a LangGraph `StateGraph` shaped like this:

```
START → grounding → capsule_composition → coordinator ⇄ {confirmation_interrupt | consult | tool_execution} → END
```

- **`grounding`** (no LLM) resolves identity from the authenticated session and — this is the load-bearing detail — **appends the new user turn onto `AgentState.messages`**. It runs exactly once per outer `ainvoke`/`astream` call (no edge ever loops back into it), which is what makes it the single correct place to do this rather than `coordinator`. When the checkpointer restores prior `messages` for a continuing thread, grounding appends onto them; when the thread is new, it seeds from empty.
- **`capsule_composition`** builds the system prompt by concatenating YAML fragments from `app/graph/capsules/`: `core.yaml` always loads, plus one `role/<role>.yaml` selected by the grounded role. This is a deliberately minimal stand-in for a larger priority-merged capsule registry.
- **`coordinator`** is the only node that calls Claude (`app/graph/nodes/coordinator.py`). It reads `state["messages"]` directly (trusting grounding already set it), appends the assistant turn, and sets `stop_reason`.
- The router (`_route_after_coordinator` in `build.py`) inspects which tool name(s) Claude's last turn asked for and picks a branch. **Priority: write > consult > plain tool > end.** This routing decision is entirely about *where an already-chosen tool call gets executed* — which tool Claude picks in the first place is governed only by system-prompt text (`core.yaml`) and each tool's own `description` in `app/graph/tools.py`, not by any code.
- **`confirmation_interrupt`** — human-in-the-loop gate for write/side-effecting tools, via `langgraph.types.interrupt()`. Fails closed to "cancel" on any unrecognized resume value.
- **`consult`** — routes to a sub-agent's own fully separate compiled `StateGraph` (currently just `consult_knowledge_agent` → `app/graph/subgraphs/knowledge_agent.py`). To add another sub-agent: write a new subgraph module, register it in `consult_node`'s `consult_graphs` dict, add its tool schema to `TOOLS`, and add its name to `CONSULT_TOOL_NAMES`.
- **`tool_execution`** — runs everything else. Tool implementations in `app/graph/tools.py` share a uniform `(tool_input, username) -> str` call signature even though most ignore `username` — only `search_documents` needs it, but the shared signature lets `tool_execution_node` call any tool generically by name.
- All three branches fold a `tool_result` back into `messages` and loop back to `coordinator`; the turn only ends when `stop_reason != "tool_use"`.

### The public boundary: never call the compiled graph directly

`app/graph/runner.py`'s `GraphTurnRunner` is the **only** thing that calls the compiled graph's `ainvoke`/`astream`. `app/main.py`'s routes only go through it (via the `GraphRunner` Protocol + DTOs in `app/graph/public/contracts.py`). `GraphTurnRunner` is also where every turn's transcript gets durably persisted (see below) and where a new chat's row gets created — so persistence logic lives here, not in route handlers or graph nodes.

### Persistence is split three ways — know which one to touch

1. **LangGraph `InMemorySaver` checkpointer** (`app/main.py`) — the *actual* conversational state (`AgentState.messages`, including tool_use/tool_result blocks) that the model sees, keyed by `thread_id`. Fully in-process: **wiped on server restart.** This is a known, currently-unsolved gap — reopening an old chat still shows the correct transcript (from Postgres) after a restart, but continuing it will start Claude with no real memory of that conversation, because the checkpoint is gone.
2. **Postgres, `learning` schema** (`app/graph/pg_memory.py`, plus `app/core/auth.py` for auth tables) — durable, cross-restart, cross-thread:
   - `memories` — explicit key/value facts (`remember_fact`/`recall_fact`).
   - `attachments` — uploaded-document metadata + the full **docling-converted markdown** (not the original file bytes, which are never stored).
   - `messages` — the durable human-readable transcript (one row per user/assistant turn), attributed to `username`+`thread_id`.
   - `chats` — one row per `thread_id` for the sidebar list (title, `updated_at`), created lazily on a chat's first message.
   - `users`/`sessions` (`app/core/auth.py`) — the single hardcoded account and opaque-token session cookies.
3. **Qdrant** (`app/graph/qdrant_memory.py`) — hybrid dense (`BAAI/bge-small-en-v1.5`) + BM25 vector search over uploaded-document chunks, filtered by `username` in the payload so `search_documents` only ever surfaces the caller's own uploads.

### Document ingestion pipeline

Upload → **docling-serve** (a separate HTTP microservice, `app/graph/docling_client.py`; only `/v1/convert/file` is used, converting to markdown) → **custom chunking** (`app/graph/chunking.py` — greedy paragraph packing, 1200 chars/chunk with 150-char overlap; deliberately *not* docling's own chunker, which needs a Hugging Face tokenizer download that fails behind this network's TLS-inspecting proxy) → embedded and upserted into Qdrant, while the full converted markdown goes to Postgres.

### Auth

Single hardcoded account (`test`/`test`, seeded into `learning.users` on startup — a real DB lookup, not a Python literal comparison). Session is an opaque `secrets.token_urlsafe` token in an `HttpOnly` cookie, resolved against `learning.sessions` server-side (`app/core/auth.py`) — not JWT, not Starlette's `SessionMiddleware`. `get_current_user` is the FastAPI dependency gating most routes. No CSRF protection, no password hashing, no multi-user support — all deliberate, matching the single-account scope.

### Frontend

`app/static/playground.html` is a single static file (no build step, vanilla JS) served at `/playground` — a login gate, a chat sidebar (list/create/switch chats), and a composer that drives `/graph/stream` and `/graph/resume/stream` over SSE. It doubles as the manual test harness for the whole system; there's no other UI.

### Separate, unrelated endpoint

`POST /agent/plan` (`app/agents/plan_agent.py`) is a single raw `client.messages.create` call with no tools, no graph, no persistence — not part of the coordinator loop. Don't assume it shares any of the above machinery.

### Two alternate execution engines (`app/agentsdk/`, `app/deepagents_engine/`)

The same coordinator turn is implemented three times. Both alternates replace
**only the graph execution** — the loop that calls Claude, runs tools and gates
writes — and reuse everything else unchanged: tool schemas/implementations in
`app/graph/tools.py`, the capsules, `pg_memory`, and the `GraphRunner` Protocol
+ DTOs. Each runner is a drop-in for `GraphTurnRunner`, so
`/agentsdk/*` and `/deepagents/*` in `main.py` are copies of the `/graph/*`
routes with only the runner object swapped.

- **`app/deepagents_engine/`** — LangChain's `deepagents`. `create_deep_agent()`
  returns a `CompiledStateGraph`, so this runner still uses `astream`,
  `thread_id`, `__interrupt__` and `Command(resume=...)` exactly like the
  hand-built graph; the router, interrupt node, consult node and tool_execution
  node collapse into four keyword arguments (`tools`, `subagents`,
  `interrupt_on`, `checkpointer`). It also adds nine tool schemas whether you
  want them or not — a state-backed virtual filesystem plus the `task` sub-agent
  tool. (`write_todos` is opt-in in 0.7.17; add `TodoListMiddleware` for it.)
  The compiled graph is only four nodes: `__start__` →
  `PatchToolCallsMiddleware.before_agent` → `model` →
  `HumanInTheLoopMiddleware.after_model` → {`tools` | `model` | `__end__`}. Resume payload differs: `{"decisions": [{"type": "approve"}]}`.
- **`app/agentsdk/`** — the Claude Agent SDK (`query()` + an in-process MCP
  server). Two things to know: the write gate is a `can_use_tool` callback that
  blocks **inside** the live loop (so a paused turn is a coroutine kept alive in
  `_live[thread_id]`, not a serializable checkpoint), and it spawns the Node.js
  `claude` CLI as a subprocess — Node + `claude` on PATH are extra runtime
  requirements.

`app/agentsdk/COMPARISON.md` is the three-way write-up.
