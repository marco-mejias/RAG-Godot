"""Question answering: rewrite query -> retrieve -> build prompt -> LLM -> answer with sources."""

from __future__ import annotations

import re
from dataclasses import dataclass

from godot_rag.llm import LLM, Message
from godot_rag.prompts import QA_SYSTEM, REWRITE_SYSTEM, qa_prompt
from godot_rag.retrieve import Hit, Retriever

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    sources: list[tuple[int, Hit]]     # (citation number, excerpt) actually cited in the text
    context: list[Hit]                 # everything that was given to the LLM
    search_query: str                  # what was actually searched


def cited_sources(text: str, hits: list[Hit], start: int = 1) -> list[tuple[int, Hit]]:
    numbers = sorted({int(n) for n in CITATION_RE.findall(text)})
    return [(n, hits[n - start]) for n in numbers if 0 <= n - start < len(hits)]


def rewrite_query(question: str, llm: LLM) -> str:
    """The embedding model only understands English, and short queries match better than
    conversational questions, so the LLM first turns the question into an English search query."""
    reply = llm.chat(REWRITE_SYSTEM, [Message("user", question)])
    query = reply.text.strip().splitlines()[0].strip(' "`') if reply.text.strip() else ""
    return query or question


def answer_question(question: str, retriever: Retriever, llm: LLM, k: int | None = None,
                    rewrite: bool = True) -> Answer:
    query = rewrite_query(question, llm) if rewrite else question
    hits = retriever.search(query, k=k)
    reply = llm.chat(QA_SYSTEM, [Message("user", qa_prompt(question, hits))])
    return Answer(reply.text.strip(), cited_sources(reply.text, hits), hits, query)
