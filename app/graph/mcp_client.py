from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# uvx fetches and runs duckduckgo-mcp-server fresh — no local install needed,
# no API key needed. UV_SYSTEM_CERTS mirrors the --system-certs flag we've
# needed for every `uv add` on this network (corporate TLS-inspecting proxy).
_SERVER_PARAMS = StdioServerParameters(
    command="uvx",
    args=["duckduckgo-mcp-server"],
    env={"UV_SYSTEM_CERTS": "1"},
)


async def mcp_web_search(query: str) -> str:
    """Spawns the real open-source duckduckgo-mcp-server as a subprocess,
    speaks the actual MCP protocol over stdio (initialize handshake, then
    call_tool), and returns its search results as text.

    Mirrors assist-workflows' MCP wiring (app/ai/mcp/client.py) at learning
    scale — a real MCP server process and a real JSON-RPC session, not a
    hand-rolled HTTP wrapper. One caveat: this spawns a fresh subprocess and
    redoes the handshake on every call, which is slow (a few seconds) — fine
    for learning, but a real deployment would keep a session open across calls.
    """

    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search", {"query": query})
            return "\n".join(block.text for block in result.content if hasattr(block, "text"))
