from typing import TypedDict


class AgentState(TypedDict, total=False):
    message: str
    username: str
    grounding: dict[str, str]
    system_prompt: str
    reply: str
    # The running Anthropic-format conversation. grounding_node appends each
    # new user turn (continuing whatever the checkpointer restored for this
    # thread_id); then one [assistant, user] pair per coordinator<->
    # tool_execution loop iteration (assistant's tool_use + our tool_result).
    messages: list[dict]
    # Set by coordinator from the LLM response, read only by the router —
    # never consumed by any other node.
    stop_reason: str
