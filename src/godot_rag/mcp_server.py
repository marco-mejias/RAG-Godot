"""Layer 3: the RAG exposed as an MCP server, so Claude Code / Claude Desktop can use it as tools.

Here the roles flip: Claude is the LLM and decides when to search. The server has no
LLM of its own except in `ask_godot_docs`, which returns a draft from the configured
generator (Gemini by default) together with its excerpts and the API check, and asks
Claude to verify it before using it: cheap model drafts, Claude checks.

Run it:   python -m godot_rag.mcp_server        (stdio transport; see README for the client config)
"""

from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache

# The embedding model is already downloaded by `ingest`: don't check Hugging Face on every start.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mcp.server.mcpserver import MCPServer

from godot_rag.config import load_settings
from godot_rag.prompts import format_excerpts

INSTRUCTIONS = """\
Tools over the official Godot 4 documentation (latest stable release), indexed locally.
- For any question about Godot 4 classes, methods, signals or engine behaviour, search the docs \
(search_godot_docs / lookup_godot_class) instead of answering from memory, and cite the URLs you use.
- Before showing the user GDScript code, run check_godot_api on it: it reports classes and methods \
that don't exist in Godot 4 (including Godot 3 names).
- ask_godot_docs returns a draft written by another model plus the excerpts it used: verify every \
claim against those excerpts and fix or drop what they don't support before answering."""

server = MCPServer("godot-docs", instructions=INSTRUCTIONS)


# --- shared resources, loaded on first use (the embedding model takes a few seconds) ------

@lru_cache(maxsize=1)
def _settings():
    return load_settings()


@lru_cache(maxsize=1)
def _retriever():
    from godot_rag.retrieve import Retriever
    with contextlib.redirect_stdout(sys.stderr):   # stdout is the MCP channel: never print to it
        return Retriever(_settings())


@lru_cache(maxsize=1)
def _api():
    from godot_rag.verify.api_check import ApiIndex
    return ApiIndex.from_settings(_settings())


# --- tools -------------------------------------------------------------------------------

@server.tool()
def search_godot_docs(query: str, k: int = 5) -> str:
    """Semantic search over the Godot 4 documentation (manual + class reference).

    Use a short English query, e.g. "connect a signal with extra arguments".
    Returns numbered excerpts, each with its section and exact URL.
    """
    hits = _retriever().search(query, k=max(1, min(k, 10)))
    return format_excerpts(hits) if hits else "No results."


@server.tool()
def lookup_godot_class(class_name: str, topic: str = "", k: int = 4) -> str:
    """Look inside one Godot 4 class reference page, e.g. class_name="CharacterBody2D", topic="move_and_slide".

    Also returns the class's inheritance chain. Use it to confirm a method or property exists.
    """
    info = _api().get(class_name)
    if not info:
        from godot_rag.verify.api_check import _close
        guess = _close(class_name, [c.name for c in _api().classes.values()])
        return f"No class named '{class_name}' in Godot 4." + (f" Did you mean '{guess}'?" if guess else "")
    chain = " < ".join([info.name, *info.parents]) if info.parents else info.name
    hits = _retriever().search_class(info.name, topic, k=max(1, min(k, 10)))
    return f"Inheritance: {chain}\n\n" + (format_excerpts(hits) if hits else "No matching section.")


@server.tool()
def check_godot_api(code_or_answer: str) -> str:
    """Check that every Godot class, method and property used in some GDScript code (or in an
    answer containing code in backticks) exists in Godot 4. Flags invented APIs and Godot 3 names."""
    from godot_rag.verify.api_check import check_text
    text = code_or_answer if "`" in code_or_answer else f"```gdscript\n{code_or_answer}\n```"
    return check_text(text, _api()).summary()


@server.tool()
def ask_godot_docs(question: str) -> str:
    """Get a draft answer to a Godot 4 question, written by the project's generator model from
    retrieved excerpts, plus the excerpts themselves and an automatic API check.
    You must verify the draft against the excerpts before relying on it."""
    from godot_rag.llm import LLMError, get_llm
    from godot_rag.qa import answer_question
    try:
        with contextlib.redirect_stdout(sys.stderr):
            answer = answer_question(question, _retriever(), get_llm(_settings()), api=_api())
    except LLMError as exc:
        hits = _retriever().search(question)
        return f"(Generator unavailable: {exc})\n\nExcerpts:\n\n{format_excerpts(hits)}"
    report = answer.api_report.summary() if answer.api_report else ""
    return (f"DRAFT (unverified, by {_settings().provider}):\n{answer.text}\n\n===\n{report}\n\n===\n"
            f"EXCERPTS (search query: {answer.search_query!r}):\n\n{format_excerpts(answer.context)}")


@server.tool()
def parse_godot_error(log: str) -> str:
    """Extract structured facts from a Godot error log (Output panel, debugger tab or script editor):
    category, script, line, function, classes and names involved. Deterministic, no LLM."""
    from godot_rag.errors.parse_error import parse_godot_errors
    known = {c.name for c in _api().classes.values()}
    errors = parse_godot_errors(log, known)
    return "\n".join(f"- {e.summary()}" for e in errors) or "No Godot error recognised in that text."


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
