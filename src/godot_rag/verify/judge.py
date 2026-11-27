"""Layer 2: a second, stronger LLM checks the draft claim by claim against the cited excerpts.

Generating a good answer is hard; checking whether a sentence is backed by a given
paragraph is much easier, so a strong model used as a judge is a cheap way to catch
hallucinations from a cheaper generator. Output is structured JSON (JUDGE_SCHEMA),
so the verdict can be shown, counted and tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from godot_rag.llm import LLM, Message
from godot_rag.prompts import JUDGE_SCHEMA, JUDGE_SYSTEM, judge_prompt
from godot_rag.retrieve import Hit


@dataclass
class Claim:
    text: str
    verdict: str            # supported | unsupported | contradicted
    citations: list[int]
    note: str


@dataclass
class Verdict:
    claims: list[Claim]
    corrected_answer: str
    search_again: str
    judge: str              # which model judged, e.g. "claude:claude-opus-5-5"
    rounds: int = 1         # 2 when an extra search was made (corrective RAG)
    notes: list[str] = field(default_factory=list)

    def count(self, verdict: str) -> int:
        return sum(c.verdict == verdict for c in self.claims)

    @property
    def needs_fix(self) -> bool:
        return bool(self.corrected_answer.strip()) or any(c.verdict != "supported" for c in self.claims)

    def summary(self) -> str:
        head = (f"Verification ({self.judge}): {len(self.claims)} claims - "
                f"{self.count('supported')} supported, {self.count('unsupported')} unsupported, "
                f"{self.count('contradicted')} contradicted")
        if self.rounds > 1:
            head += " (after one extra search)"
        lines = [head]
        mark = {"unsupported": "⚠", "contradicted": "✘"}
        for c in self.claims:
            if c.verdict != "supported":
                lines.append(f"  {mark[c.verdict]} {c.text}" + (f" - {c.note}" if c.note else ""))
        lines += [f"  {n}" for n in self.notes]
        return "\n".join(lines)


def judge(question: str, hits: list[Hit], draft: str, api_summary: str, llm: LLM) -> Verdict:
    data = llm.chat_json(JUDGE_SYSTEM, [Message("user", judge_prompt(question, hits, draft, api_summary))], JUDGE_SCHEMA)
    claims = [
        Claim(str(c.get("claim", "")), str(c.get("verdict", "unsupported")),
              [int(n) for n in c.get("citations", []) if str(n).isdigit()], str(c.get("note", "")))
        for c in data.get("claims", [])
    ]
    return Verdict(claims, str(data.get("corrected_answer", "")), str(data.get("search_again", "")), repr(llm))
