from collections.abc import Awaitable, Callable
from typing import Any

from app.graph.state import AgentState


def build_grounding_node() -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Stage A — resolve who's asking, before anything else runs.

    No LLM call here — deterministic, like assist-workflows' grounding node.
    ``user_id`` now comes from the session-authenticated caller (see
    app/core/auth.py, threaded in via TurnRequest.username); ``role`` stays
    hardcoded since there's still no role system behind the one login.

    Also where a new user turn gets appended onto the conversation. This
    node runs exactly once per outer ainvoke/astream call — never mid-loop,
    since no edge in build.py ever routes back into it — so it's the single
    correct place to do this: on a brand-new thread `state.get("messages")`
    is empty and this just seeds it; on a continued thread, the checkpointer
    has already restored the previous turns under this thread_id, and this
    appends the new message onto them instead of silently dropping it (see
    coordinator_node, which used to special-case the empty case here and no
    longer needs to).
    """

    async def grounding_node(state: AgentState) -> dict[str, Any]:
        incoming_message = state["message"]  # <-- reading data injected via ainvoke()
        prior_messages = state.get("messages") or []
        return {
            "grounding": {
                "user_id": state.get("username") or "learning-user",
                "role": "analyst",
                "echoed_message": incoming_message,
            },
            "messages": [*prior_messages, {"role": "user", "content": incoming_message}],
        }

    return grounding_node
