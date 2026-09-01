"""Node-name -> user-facing stage label. Mirrors assist-workflows'
infra/langgraph/streaming.py's _STAGE_LABELS — the raw-event-to-StreamEvent
translation itself now lives in graph/runner.py (the thing that actually
calls astream), same split as their translate_langgraph_event living
alongside the graph layer while wire serialization (_sse_frame) lives in
the HTTP layer (chat/api.py — here, main.py).
"""

STAGE_LABELS: dict[str, str] = {
    "grounding": "Resolving context",
    "capsule_composition": "Composing capsules",
    "coordinator": "Reasoning",
    "consult": "Consulting knowledge agent",
    "tool_execution": "Running tool",
    "confirmation_interrupt": "Awaiting confirmation",
}
