from typing import TypedDict

from anthropic import AsyncAnthropic
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings

_SYSTEM_PROMPT = (
    "You are the Knowledge Agent, a research specialist consulted by another "
    "agent (the coordinator) — you are not talking to the end user directly. "
    "Answer thoroughly and precisely. Always end your answer with the exact "
    "line: -- Knowledge Agent v1"
)


class KnowledgeAgentState(TypedDict, total=False):
    question: str
    answer: str


def _build_knowledge_agent_llm_node(settings: Settings):
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def knowledge_agent_llm(state: KnowledgeAgentState) -> dict[str, str]:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["question"]}],
        )
        return {"answer": response.content[0].text}

    return knowledge_agent_llm


def build_knowledge_agent_graph(settings: Settings) -> object:
    """A real sub-agent: its own StateGraph, own state, compiled independently
    of the coordinator's graph. Just one node today (no tools of its own, no
    loop) — but because it's a graph and not a function, it could grow an
    internal tool_execution loop later without changing how the coordinator
    calls it at all. That's the actual point of a sub-agent looking like a
    tool from the outside.
    """

    g = StateGraph(KnowledgeAgentState)
    g.add_node("knowledge_agent_llm", _build_knowledge_agent_llm_node(settings))
    g.add_edge(START, "knowledge_agent_llm")
    g.add_edge("knowledge_agent_llm", END)
    return g.compile()
