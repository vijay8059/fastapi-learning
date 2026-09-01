import secrets
from datetime import timedelta
from typing import Annotated

import asyncpg
from fastapi import Depends, HTTPException, Request

from app.core.config import Settings, get_settings

# Server-side sessions: an opaque token in an HttpOnly cookie, resolved
# against a Postgres row on every request. Deliberately not Starlette's
# SessionMiddleware (needs itsdangerous, stores state client-side) — this
# matches pg_memory.py's existing pattern of owning its own short-lived
# asyncpg connections instead.

# The one seed account — inserted into learning.users on startup (see
# ensure_auth_schema) so login is a real DB lookup, not a Python literal
# comparison. Plaintext on purpose: single hardcoded learning-project
# account, no signup flow to protect.
_SEED_USERNAME = "test"
_SEED_PASSWORD = "test"

SESSION_COOKIE = "session_id"
_SESSION_TTL = timedelta(days=7)
SESSION_TTL_SECONDS = int(_SESSION_TTL.total_seconds())


async def _connect(settings: Settings) -> asyncpg.Connection:
    return await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


async def ensure_auth_schema(settings: Settings) -> None:
    """Idempotent — safe to call on every app startup."""

    conn = await _connect(settings)
    try:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS learning")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning.users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO learning.users (username, password)
            VALUES ($1, $2)
            ON CONFLICT (username) DO NOTHING
            """,
            _SEED_USERNAME,
            _SEED_PASSWORD,
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning.sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at TIMESTAMPTZ NOT NULL
            )
            """
        )
    finally:
        await conn.close()


async def verify_credentials(settings: Settings, username: str, password: str) -> bool:
    conn = await _connect(settings)
    try:
        return await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM learning.users WHERE username = $1 AND password = $2)",
            username,
            password,
        )
    finally:
        await conn.close()


async def create_session(settings: Settings, username: str) -> str:
    session_id = secrets.token_urlsafe(32)
    conn = await _connect(settings)
    try:
        await conn.execute(
            """
            INSERT INTO learning.sessions (session_id, username, expires_at)
            VALUES ($1, $2, now() + $3::interval)
            """,
            session_id,
            username,
            _SESSION_TTL,
        )
    finally:
        await conn.close()
    return session_id


async def resolve_session(settings: Settings, session_id: str) -> str | None:
    conn = await _connect(settings)
    try:
        return await conn.fetchval(
            "SELECT username FROM learning.sessions WHERE session_id = $1 AND expires_at > now()",
            session_id,
        )
    finally:
        await conn.close()


async def delete_session(settings: Settings, session_id: str) -> None:
    conn = await _connect(settings)
    try:
        await conn.execute("DELETE FROM learning.sessions WHERE session_id = $1", session_id)
    finally:
        await conn.close()


async def get_current_user(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> str:
    """FastAPI dependency gating the protected routes — 401s if there's no
    valid session cookie."""

    session_id = request.cookies.get(SESSION_COOKIE)
    username = await resolve_session(settings, session_id) if session_id else None
    if username is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return username
