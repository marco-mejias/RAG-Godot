"""Command line interface.

    python -m godot_rag ingest                       download the docs and build the index
    python -m godot_rag search "query"               show matching excerpts (no LLM, free)
    python -m godot_rag ask "question"               answer with citations
    python -m godot_rag explain-error error.log      diagnose a Godot error (agent)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

from godot_rag.config import Settings, load_settings


def _print_sources(sources) -> None:
    if not sources:
        return
    print("\nSources:")
    for n, hit in sources:
        print(f"  [{n}] {hit.section}\n      {hit.url}")


def cmd_ingest(settings: Settings, args) -> int:
    from godot_rag.index import Embedder, build_index
    from godot_rag.ingest.chunk import chunk_page
    from godot_rag.ingest.download import download_docs, iter_rst_files
    from godot_rag.ingest.parse import parse_file

    download_docs(settings, force=args.force_download)
    files = iter_rst_files(settings)
    if args.limit:
        files = files[: args.limit]
    print(f"Parsing {len(files)} pages ...")
    chunks = []
    for path in files:
        page = parse_file(path, settings.raw_dir, settings.docs_base_url)
        chunks.extend(chunk_page(page, settings.chunk_chars, settings.chunk_overlap))
    print(f"{len(chunks)} chunks. Loading embedding model {settings.embedding_model} ...")
    start = time.perf_counter()
    total = build_index(settings, chunks, Embedder(settings.embedding_model))
    print(f"Done: {total} chunks indexed in {time.perf_counter() - start:.0f}s -> {settings.index_dir}")
    return 0


def cmd_search(settings: Settings, args) -> int:
    from godot_rag.retrieve import Retriever

    for n, hit in enumerate(Retriever(settings).search(args.query, k=args.k), 1):
        print(f"\n[{n}] {hit.score:.3f}  {hit.section}\n    {hit.url}")
        print(textwrap.indent(textwrap.shorten(hit.body, 300), "    "))
    return 0


def cmd_ask(settings: Settings, args) -> int:
    from godot_rag.llm import get_llm
    from godot_rag.qa import answer_question
    from godot_rag.retrieve import Retriever

    llm = get_llm(settings, args.provider)
    answer = answer_question(args.question, Retriever(settings), llm, k=args.k, rewrite=not args.no_rewrite)
    if args.show_context:
        print(f"Search query: {answer.search_query}")
        print("Context given to the LLM:")
        for n, hit in enumerate(answer.context, 1):
            print(f"  [{n}] {hit.score:.3f} {hit.section}")
        print()
    print(answer.text)
    _print_sources(answer.sources)
    print(f"\n({llm!r})")
    return 0


def _read_input(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    path = Path(value)
    return path.read_text(encoding="utf-8") if path.exists() else value   # a file, or the error text itself


def cmd_explain_error(settings: Settings, args) -> int:
    from godot_rag.errors.parse_error import parse_godot_errors

    log = _read_input(args.log)
    known = {p.stem.removeprefix("class_") for p in (settings.raw_dir / "classes").glob("class_*.rst")} or None

    if args.parse_only:
        errors = parse_godot_errors(log, known)
        for e in errors:
            print("-", e.summary())
        return 0 if errors else 1

    from godot_rag.errors.agent import ErrorAgent
    from godot_rag.llm import get_llm
    from godot_rag.retrieve import Retriever

    code = Path(args.code).read_text(encoding="utf-8") if args.code else None
    llm = get_llm(settings, args.provider)
    agent = ErrorAgent(llm, Retriever(settings), settings.max_agent_steps, settings.results_per_search, known)
    diagnosis = agent.run(log, code)
    if diagnosis.steps:
        print("Searches made by the agent:")
        for step in diagnosis.steps:
            print(f"  - {step}")
        print()
    print(diagnosis.text)
    _print_sources(diagnosis.sources)
    print(f"\n({llm!r})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="godot_rag", description="Godot 4 documentation assistant (RAG).")
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download the Godot docs and build the vector index")
    p.add_argument("--force-download", action="store_true", help="download the docs again")
    p.add_argument("--limit", type=int, help="only index the first N pages (quick test)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="show the excerpts that match a query (no LLM)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="ask a question about Godot 4")
    p.add_argument("question")
    p.add_argument("-k", type=int, help="number of excerpts to retrieve")
    p.add_argument("--provider", help="gemini | claude | ollama (overrides config)")
    p.add_argument("--show-context", action="store_true", help="also print the search query and retrieved excerpts")
    p.add_argument("--no-rewrite", action="store_true", help="search with the question as typed (no LLM rewrite)")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("explain-error", help="explain a Godot error and how to fix it")
    p.add_argument("log", help="log file, '-' for stdin, or the error text itself")
    p.add_argument("--code", help="the .gd script involved (optional, improves the answer)")
    p.add_argument("--provider", help="gemini | claude | ollama (overrides config)")
    p.add_argument("--parse-only", action="store_true", help="only show what was extracted from the log (no LLM)")
    p.set_defaults(func=cmd_explain_error)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
        except AttributeError:
            pass
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)

    from godot_rag.llm import LLMError

    try:
        return args.func(settings, args)
    except (LLMError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
