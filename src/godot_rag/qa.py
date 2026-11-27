"""Question answering: rewrite query -> retrieve -> draft -> verify (optional) -> answer with sources.

    question ──rewrite──▶ English query ──search──▶ excerpts ──LLM──▶ draft
                                                                      │
                          layer 1: API check (always, free) ◀─────────┤
                          layer 2: judge LLM (--verify) ◀─────────────┘
                              │ unsupported claims → corrected answer
                              │ missing info → one extra search, new draft, judged again
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from godot_rag.llm import LLM, Message
from godot_rag.prompts import QA_SYSTEM, REWRITE_SYSTEM, qa_prompt
from godot_rag.retrieve import Hit, Retriever
from godot_rag.verify.api_check import ApiIndex, ApiReport, check_text
from godot_rag.verify.judge import Verdict, judge

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str                          # final answer (corrected by the judge if needed)
    sources: list[tuple[int, Hit]]     # (citation number, excerpt) actually cited in the text
    context: list[Hit]                 # everything that was given to the LLM
    search_query: str                  # what was actually searched
    draft: str = ""                    # the generator's answer before verification
    api_report: ApiReport | None = None
    verdict: Verdict | None = None


def cited_sources(text: str, hits: list[Hit], start: int = 1) -> list[tuple[int, Hit]]:
    numbers = sorted({int(n) for n in CITATION_RE.findall(text)})
    return [(n, hits[n - start]) for n in numbers if 0 <= n - start < len(hits)]


def rewrite_query(question: str, llm: LLM) -> str:
    """The embedding model only understands English, and short queries match better than
    conversational questions, so the LLM first turns the question into an English search query."""
    reply = llm.chat(REWRITE_SYSTEM, [Message("user", question)])
    query = reply.text.strip().splitlines()[0].strip(' "`') if reply.text.strip() else ""
    return query or question


def _draft(question: str, hits: list[Hit], llm: LLM) -> str:
    return llm.chat(QA_SYSTEM, [Message("user", qa_prompt(question, hits))]).text.strip()


def answer_question(question: str, retriever: Retriever, llm: LLM, k: int | None = None,
                    rewrite: bool = True, api: ApiIndex | None = None, verifier: LLM | None = None) -> Answer:
    query = rewrite_query(question, llm) if rewrite else question
    hits = retriever.search(query, k=k)
    draft = _draft(question, hits, llm)
    report = check_text(draft, api) if api else None
    verdict = None
    final = draft

    if verifier:
        summary = report.summary() if report else "API check: not run."
        verdict = judge(question, hits, draft, summary, verifier)
        if verdict.search_again.strip():
            # Corrective RAG: the excerpts miss something the question needs (even if what the
            # draft says is correct). Fetch it, redraft, judge once more.
            seen = {(h.url, h.body) for h in hits}
            extra = [h for h in retriever.search(verdict.search_again, k=3) if (h.url, h.body) not in seen]
            if extra:
                hits = hits + extra
                draft = _draft(question, hits, llm)
                report = check_text(draft, api) if api else None
                summary = report.summary() if report else "API check: not run."
                first_query = verdict.search_again
                verdict = judge(question, hits, draft, summary, verifier)
                verdict.rounds = 2
                verdict.notes.append(f"Extra search: {first_query!r} added {len(extra)} excerpt(s).")
        final = verdict.corrected_answer.strip() or draft

    return Answer(final, cited_sources(final, hits), hits, query, draft, report, verdict)
