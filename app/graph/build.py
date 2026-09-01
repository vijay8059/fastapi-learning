from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.graph.nodes.capsule_composition import build_capsule_composition_node
from app.graph.nodes.confirmation_interrupt import build_confirmation_interrupt_node
from app.graph.nodes.consult import build_consult_node
from app.graph.nodes.coordinator import build_coordinator_node
from app.graph.nodes.grounding import build_grounding_node
from app.graph.nodes.tool_execution import build_tool_execution_node
from app.graph.state import AgentState
from app.graph.tools import CONSULT_TOOL_NAMES, WRITE_TOOL_NAMES

NODE_GROUNDING = "grounding"
NODE_CAPSULE_COMPOSITION = "capsule_composition"
NODE_COORDINATOR = "coordinator"
NODE_INTERRUPT = "confirmation_interrupt"
NODE_CONSULT = "consult"
NODE_TOOL_EXECUTION = "tool_execution"


def _route_after_coordinator(
    state: AgentState,
) -> Literal["interrupt", "consult", "tool_execution", "end"]:
    """The router — reads which tool(s) Claude just asked for and picks a
    branch. Priority mirrors assist-workflows' route_entry_after_llm: a
    pending write always wins over a consult, which wins over a plain tool.
    """

    if state["stop_reason"] != "tool_use":
        return "end"

    tool_names = {
        block["name"] for block in state["messages"][-1]["content"] if block["type"] == "tool_use"
    }
    if tool_names & WRITE_TOOL_NAMES:
        return "interrupt"
    if tool_names & CONSULT_TOOL_NAMES:
        return "consult"
    return "tool_execution"


def build_graph(settings: Settings, checkpointer: BaseCheckpointSaver) -> object:
    g = StateGraph(AgentState)
    g.add_node(NODE_GROUNDING, build_grounding_node())
    g.add_node(NODE_CAPSULE_COMPOSITION, build_capsule_composition_node())
    g.add_node(NODE_COORDINATOR, build_coordinator_node(settings))
    g.add_node(NODE_INTERRUPT, build_confirmation_interrupt_node())
    g.add_node(NODE_CONSULT, build_consult_node(settings))
    g.add_node(NODE_TOOL_EXECUTION, build_tool_execution_node(settings))

    g.add_edge(START, NODE_GROUNDING)
    g.add_edge(NODE_GROUNDING, NODE_CAPSULE_COMPOSITION)
    g.add_edge(NODE_CAPSULE_COMPOSITION, NODE_COORDINATOR)
    g.add_conditional_edges(
        NODE_COORDINATOR,
        _route_after_coordinator,
        {
            "interrupt": NODE_INTERRUPT,
            "consult": NODE_CONSULT,
            "tool_execution": NODE_TOOL_EXECUTION,
            "end": END,
        },
    )
    g.add_edge(NODE_INTERRUPT, NODE_COORDINATOR)
    g.add_edge(NODE_CONSULT, NODE_COORDINATOR)
    g.add_edge(NODE_TOOL_EXECUTION, NODE_COORDINATOR)

    return g.compile(checkpointer=checkpointer)
