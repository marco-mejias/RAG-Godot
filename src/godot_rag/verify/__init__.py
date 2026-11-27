"""Answer verification in layers.

1. api_check.py  - free and deterministic: every Godot class/method an answer mentions must exist
                   in the Godot 4 class reference (catches invented and Godot 3 APIs).
2. judge.py      - a stronger LLM (Claude by default) checks each claim against the cited excerpts
                   and corrects unsupported ones; can trigger one extra search (corrective RAG).
3. ../mcp_server.py exposes the same checks to Claude Code / Claude Desktop as MCP tools.
"""
