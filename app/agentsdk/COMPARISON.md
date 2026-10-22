# Three engines, one turn

The same coordinator turn is implemented three times. Tools, capsules, the
Postgres transcript and the `GraphRunner` Protocol are shared and unchanged;
only the engine differs.

| | `app/graph/` — LangGraph | `app/deepagents_engine/` — deepagents | `app/agentsdk/` — Claude Agent SDK |
|---|---|---|---|
| What it is | A graph framework. You draw the nodes | A **pre-built LangGraph graph** you configure | A finished harness (Claude Code as a library) |
| Engine code here | 352 lines, 12 files | 209 lines, 3 files | 249 lines, 3 files |
| Who owns the loop | You (`coordinator` node + edges) | deepagents (a ReAct model/tools pair) | The SDK (`query()`) |
| Routing | Your router: write > consult > tool > end | None — everything is a tool call | None — everything is a tool call |
| Tool shape | `(tool_input, username) -> str`, dispatched by you | LangChain `StructuredTool` | In-process MCP server |
| Per-turn context (`username`) | An argument | ContextVar | ContextVar |
| Sub-agent | A second compiled `StateGraph` | A `SubAgent` dict → compiled subgraph, called via `task` | An `AgentDefinition`, called via `Agent` |
| Conversation state | `AgentState["messages"]` + checkpointer | Same — it's still a LangGraph graph | A Claude Code session (JSONL on disk) |
| Write gate | `interrupt()` in your own node | `interrupt_on={...}` → the same `interrupt()`, inserted for you | `can_use_tool` callback blocking in-loop |
| Resume | `Command(resume="confirm")` | `Command(resume={"decisions":[...]})` | Hand a future to a coroutine still in memory |
| Progress events | One update per node = your stages | Two nodes only; stages recovered from tool calls | No nodes; stages recovered from tool calls |
| Free extras | none | virtual filesystem, `task`, prompt caching (`write_todos` is opt-in) | compaction, retries, sub-agent isolation |
| Extra runtime | none | none | Node.js + the `claude` CLI subprocess |

## The one structural difference that matters

**Two of the three pause the same way; one doesn't.**

LangGraph and deepagents both pause with `interrupt()`: the run unwinds,
`ainvoke` returns, and a later resume is rebuilt from the checkpointer. Nothing
has to stay in memory in between, so a paused turn can move between workers and
survive a restart (given a durable saver).

The Agent SDK's `can_use_tool` is an `await` in the middle of a live loop.
`AgentSdkTurnRunner` therefore parks the whole driving task in
`_live[thread_id]` and feeds it a decision on the next HTTP request — so a
paused turn is pinned to one process and dies with a restart. This is the
single biggest architectural consequence of the choice, and it isn't visible in
any feature list.

## Where deepagents actually sits

It is not "LangGraph but easier" and not "the Agent SDK but LangChain". It is a
**pre-assembled LangGraph graph**: `create_deep_agent()` returns a
`CompiledStateGraph`, so `app/deepagents_engine/runner.py` still calls
`astream`, still passes `thread_id`, still reads `__interrupt__`, still resumes
with `Command`. Everything that vanished — the router, the interrupt node, the
consult node, the tool_execution node — was replaced by four keyword arguments:

```python
create_deep_agent(
    model=ChatAnthropic(...),
    tools=build_tools(settings),                  # was tool_execution_node
    subagents=[KNOWLEDGE_SUBAGENT],               # was consult_node + a subgraph
    interrupt_on={"send_alert": {...}},           # was the router's write branch
    checkpointer=checkpointer,                    # unchanged
)
```

What you also get, whether you asked or not: a virtual filesystem
(`ls`/`read_file`/`write_file`/`edit_file`/`delete`/`glob`/`grep`/`execute`)
living in graph state so the agent can offload long content, a `task` tool for
sub-agent dispatch, and Anthropic prompt caching. Nine extra tool schemas reach
the model on every turn alongside our six. That is the "deep" part — the harness
assumes long-horizon work, and it costs context whether or not this chat needs
it. (`write_todos` is *not* among them: the planning middleware is opt-in in
0.7.17, despite how often it's described as standard.)

## Costs and surprises, per engine

**LangGraph** — most code, and every line of it is a place you can intervene.
Progress labels are exactly your node names. If the control flow *is* the
product, nothing else gives you this. The checkpointer is yours to configure;
ours is `InMemorySaver`, so today it's just as volatile as the SDK — but fixing
that is swapping one object, not rewriting the loop.

**deepagents** — the least code, and it keeps LangGraph's escape hatches. The
cost is opinion: you get a ReAct loop with a fixed middleware stack, so the
write/consult/plain distinction disappears and the extra built-in tools are in
the prompt whether you want them or not. Progress reporting degrades from
"named stages" to "guess from tool calls".

**Claude Agent SDK** — least code you own for the most harness, but it arrives
*full*: Read/Write/Bash/WebSearch, this repo's own `CLAUDE.md`,
`.claude/settings.json`, filesystem sub-agents. Two lines in `_build_options`
(`tools=["Agent"]`, `setting_sources=[]`) exist purely to undo those defaults.
It also spawns a Node.js subprocess per turn.

## Running them

Identical routes, three prefixes:

```
/graph/{run,stream,resume,resume/stream}        LangGraph
/deepagents/{run,stream,resume,resume/stream}   deepagents
/agentsdk/{run,stream,resume,resume/stream}     Claude Agent SDK
```

`/playground` still drives the `/graph/*` ones; change the prefix in
`playground.html` to compare.
