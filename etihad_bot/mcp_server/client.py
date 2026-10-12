"""MCP client: spawns the flight MCP server over stdio and calls its tools.

Synchronous facade (`call_tool`) so LangGraph's sync nodes can use it; each call runs in its own
event loop, which is safe because FastAPI runs sync endpoints in a worker thread.
"""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from etihad_bot.config import get_settings

def _server_params() -> StdioServerParameters:
    s = get_settings()
    # .env is parsed by pydantic, not exported to os.environ, so pass the flight settings explicitly.
    env = {**os.environ, "AVIATIONSTACK_API_KEY": s.aviationstack_api_key,
           "FLIGHT_API_MOCK": str(s.flight_api_mock).lower()}
    return StdioServerParameters(command=sys.executable,
                                 args=["-m", "etihad_bot.mcp_server.server"], env=env)


async def _with_session(fn):
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


async def _list_tools_async() -> list[dict]:
    async def run(s):
        tools = (await s.list_tools()).tools
        return [{"name": t.name, "description": t.description, "schema": t.inputSchema}
                for t in tools]
    return await _with_session(run)


async def _call_async(name: str, args: dict) -> dict:
    async def run(s):
        res = await s.call_tool(name, args)
        text = "".join(c.text for c in res.content if getattr(c, "type", "") == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"ok": False, "error": text or "empty tool response"}
    return await _with_session(run)


def list_tools() -> list[dict]:
    return asyncio.run(_list_tools_async())


def call_tool(name: str, args: dict) -> dict:
    try:
        return asyncio.run(_call_async(name, args))
    except Exception as e:
        return {"ok": False, "error": f"MCP call failed: {type(e).__name__}: {e}"}
