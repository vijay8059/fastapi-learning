"""A third execution engine for the same coordinator turn, built on
`deepagents` — LangChain's opinionated harness *on top of* LangGraph.

Same narrow scope as `app/agentsdk/`: this replaces only the loop that calls
Claude, runs tools and gates writes. The tool schemas and implementations
(`app/graph/tools.py`), the capsules, `pg_memory` and the `GraphRunner`
Protocol + DTOs are reused unchanged.

Where it sits between the other two: LangGraph is the raw graph framework
(you draw the nodes), the Claude Agent SDK is a finished harness you
configure, and deepagents is a *pre-built LangGraph graph* — a ReAct loop
with planning, a virtual filesystem and sub-agent middleware already wired
in. You get the SDK's "just configure it" ergonomics while the result is
still a `CompiledStateGraph` you can checkpoint, interrupt and stream
exactly like the hand-built one.
"""
