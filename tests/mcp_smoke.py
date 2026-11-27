"""End-to-end check of the MCP server over the real stdio protocol (needs the index; not part of pytest).

    python tests/mcp_smoke.py
"""

import asyncio
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "godot_rag.mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        for name, args in [
            ("search_godot_docs", {"query": "detect if a body is on the floor", "k": 2}),
            ("lookup_godot_class", {"class_name": "KinematicBody2D"}),
            ("lookup_godot_class", {"class_name": "CharacterBody2D", "topic": "is_on_floor", "k": 1}),
            ("check_godot_api", {"code_or_answer": "extends KinematicBody2D\nfunc _ready():\n    var s = scene.instance()\n    move_and_slide()"}),
            ("parse_godot_error", {"log": "Invalid call. Nonexistent function 'jump' in base 'CharacterBody2D'."}),
        ]:
            result = await session.call_tool(name, args)
            text = "\n".join(getattr(c, "text", "") for c in result.content)
            print(f"\n=== {name}({args}) ===\n{text[:700]}")


asyncio.run(main())
