import asyncpg

from app.core.config import Settings

# Cross-thread, cross-restart memory: unlike AgentState.messages (one thread,
# wiped on process exit unless checkpointed), rows in here outlive any single
# conversation. This is the "long-term Store" half of the memory/persistence
# split — the checkpointer answers "resume this paused thread later", this
# answers "recall this fact in a totally different thread".


async def _connect(settings: Settings) -> asyncpg.Connection:
    return await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


async def ensure_learning_schema(settings: Settings) -> None:
    """Idempotent — safe to call on every app startup."""

    conn = await _connect(settings)
    try:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS learning")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning.memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # The relational half of attachment storage: metadata + the full
        # converted document. The chunk *vectors* (and chunk text, for
        # retrieval) live in Qdrant instead — see qdrant_memory.py. Postgres
        # is the source of truth for "what was uploaded"; Qdrant is the
        # searchable index over its content.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning.attachments (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                markdown TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # Guards a DB that already had this table before `username` existed
        # (this table predates the login work) — CREATE TABLE IF NOT EXISTS
        # above is a no-op against it, so the column needs its own migration.
        await conn.execute(
            "ALTER TABLE learning.attachments ADD COLUMN IF NOT EXISTS username TEXT NOT NULL DEFAULT 'test'"
        )
        # The durable transcript — one row per user-visible message.
        # AgentState.messages (raw Anthropic format, including tool_use/
        # tool_result blocks) lives only in the checkpointer, one thread at a
        # time; this is the cross-thread, cross-restart record of who said
        # what, attributed to the logged-in user (see app/core/auth.py).
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning.messages (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS messages_username_created_at_idx
            ON learning.messages (username, created_at)
            """
        )
        # The chat list — one row per thread_id, holding just enough to
        # render a sidebar (title, last-active time) without deriving it
        # from a GROUP BY over learning.messages on every request.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning.chats (
                thread_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chats_username_updated_at_idx
            ON learning.chats (username, updated_at DESC)
            """
        )
    finally:
        await conn.close()


async def record_attachment(
    settings: Settings, username: str, thread_id: str, filename: str, markdown: str, chunk_count: int
) -> int:
    conn = await _connect(settings)
    try:
        attachment_id = await conn.fetchval(
            """
            INSERT INTO learning.attachments (username, thread_id, filename, markdown, chunk_count)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            username,
            thread_id,
            filename,
            markdown,
            chunk_count,
        )
    finally:
        await conn.close()
    return attachment_id


async def list_attachments(settings: Settings, username: str) -> list[dict]:
    conn = await _connect(settings)
    try:
        rows = await conn.fetch(
            """
            SELECT id, filename, thread_id, chunk_count, created_at
            FROM learning.attachments
            WHERE username = $1
            ORDER BY created_at DESC
            """,
            username,
        )
    finally:
        await conn.close()
    return [dict(row) for row in rows]


async def record_message(settings: Settings, username: str, thread_id: str, role: str, content: str) -> None:
    conn = await _connect(settings)
    try:
        await conn.execute(
            """
            INSERT INTO learning.messages (username, thread_id, role, content)
            VALUES ($1, $2, $3, $4)
            """,
            username,
            thread_id,
            role,
            content,
        )
    finally:
        await conn.close()


async def list_messages(settings: Settings, username: str, thread_id: str) -> list[dict]:
    conn = await _connect(settings)
    try:
        rows = await conn.fetch(
            """
            SELECT role, content, created_at
            FROM learning.messages
            WHERE username = $1 AND thread_id = $2
            ORDER BY created_at ASC
            """,
            username,
            thread_id,
        )
    finally:
        await conn.close()
    return [dict(row) for row in rows]


_CHAT_TITLE_MAX_CHARS = 60


async def touch_chat(settings: Settings, username: str, thread_id: str, first_message: str) -> None:
    """Creates the chat row on its first message, or just bumps
    `updated_at` on every later one — the title is set once and never
    overwritten, so a chat keeps the topic it started with.
    """

    title = first_message[:_CHAT_TITLE_MAX_CHARS]
    if len(first_message) > _CHAT_TITLE_MAX_CHARS:
        title += "…"

    conn = await _connect(settings)
    try:
        await conn.execute(
            """
            INSERT INTO learning.chats (thread_id, username, title)
            VALUES ($1, $2, $3)
            ON CONFLICT (thread_id) DO UPDATE SET updated_at = now()
            """,
            thread_id,
            username,
            title,
        )
    finally:
        await conn.close()


async def list_chats(settings: Settings, username: str) -> list[dict]:
    conn = await _connect(settings)
    try:
        rows = await conn.fetch(
            """
            SELECT thread_id, title, updated_at
            FROM learning.chats
            WHERE username = $1
            ORDER BY updated_at DESC
            """,
            username,
        )
    finally:
        await conn.close()
    return [dict(row) for row in rows]


async def remember(settings: Settings, key: str, value: str) -> str:
    conn = await _connect(settings)
    try:
        await conn.execute(
            """
            INSERT INTO learning.memories (key, value, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()
            """,
            key,
            value,
        )
    finally:
        await conn.close()
    return f"Remembered '{key}'."


async def recall(settings: Settings, key: str) -> str:
    conn = await _connect(settings)
    try:
        value = await conn.fetchval(
            "SELECT value FROM learning.memories WHERE key = $1", key
        )
    finally:
        await conn.close()
    return value if value is not None else f"No memory found for '{key}'."
