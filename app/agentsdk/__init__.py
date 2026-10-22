"""A second *execution engine* for the same coordinator turn, built on the
Claude Agent SDK instead of LangGraph.

Scope is deliberately narrow: this package replaces what `app/graph/build.py`
+ `app/graph/nodes/*` + `app/graph/runner.py` do (the loop that calls Claude,
runs tools, gates writes) and nothing else. Everything around it is reused
unchanged —

- tool schemas and implementations: `app/graph/tools.py`
- system prompt capsules: `app/graph/capsules/`
- durable transcript / chats: `app/graph/pg_memory.py`
- the public boundary + DTOs: `app/graph/public/contracts.py`

so `AgentSdkTurnRunner` is a drop-in for `GraphTurnRunner` behind the same
`GraphRunner` Protocol. See COMPARISON.md for what actually differs.
"""
