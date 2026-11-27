"""The error-explaining agent.

Unlike plain Q&A (one search, one LLM call), here the LLM drives the search:
it gets two tools and decides what to look up, possibly several times,
before answering. The loop below is provider-neutral, so it runs the same
on Gemini, Claude and Ollama.

    error log --parse--> hints --> LLM --tool call--> search docs --results--> LLM --> ... --> answer
"""

from __future__ import annotations

from dataclasses import dataclass, field

from godot_rag.errors.parse_error import GodotError, parse_godot_errors
from godot_rag.llm import LLM, Message, ToolCall, ToolSpec
from godot_rag.prompts import AGENT_SYSTEM, FINAL_ANSWER_NUDGE, format_excerpts
from godot_rag.qa import CITATION_RE
from godot_rag.retrieve import Hit, Retriever
from godot_rag.verify.api_check import ApiIndex, ApiReport, check_text
from godot_rag.verify.judge import Verdict, judge

TOOLS = [
    ToolSpec(
        name="search_docs",
        description=(
            "Semantic search over the whole Godot 4 documentation (manual and class reference). "
            "Use a short descriptive English query, e.g. 'get_node returns null before node is ready'."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for."}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="lookup_class",
        description=(
            "Search inside the class reference page of one Godot class, e.g. class_name='CharacterBody2D', "
            "topic='move_and_slide'. Use it to check that a method/property exists and how it works."
        ),
        parameters={
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Exact Godot class name, e.g. 'Node'."},
                "topic": {"type": "string", "description": "Method, property or topic within the class."},
            },
            "required": ["class_name"],
        },
    ),
]


@dataclass
class Diagnosis:
    text: str
    errors: list[GodotError]
    sources: list[tuple[int, Hit]]          # excerpts cited in the answer
    steps: list[str] = field(default_factory=list)   # tool calls made, for transparency
    draft: str = ""                                  # the agent's answer before verification
    api_report: ApiReport | None = None
    verdict: Verdict | None = None


class ErrorAgent:
    def __init__(self, llm: LLM, retriever: Retriever, max_steps: int = 6, results_per_search: int = 4,
                 known_classes: set[str] | None = None, api: ApiIndex | None = None, verifier: LLM | None = None):
        self.llm = llm
        self.api = api                      # layer 1: API existence check
        self.verifier = verifier            # layer 2: judge LLM
        self._log = ""
        self.retriever = retriever
        self.max_steps = max_steps
        self.k = results_per_search
        self.known_classes = known_classes
        self._numbered: list[Hit] = []      # every excerpt shown to the LLM, numbered from 1
        self._seen: dict[tuple[str, str], int] = {}

    # --- tools -------------------------------------------------------------------------

    def _register(self, hits: list[Hit]) -> str:
        """Give each new excerpt a stable number across the whole conversation."""
        blocks = []
        for hit in hits:
            key = (hit.url, hit.body)
            if key not in self._seen:
                self._numbered.append(hit)
                self._seen[key] = len(self._numbered)
                blocks.append(format_excerpts([hit], start=self._seen[key]))
            else:
                blocks.append(f"[{self._seen[key]}] (already shown above) {hit.section}")
        return "\n\n---\n\n".join(blocks) if blocks else "No results."

    def run_tool(self, call: ToolCall) -> str:
        args = call.arguments
        if call.name == "search_docs":
            hits = self.retriever.search(str(args.get("query", "")), k=self.k)
        elif call.name == "lookup_class":
            hits = self.retriever.search_class(str(args.get("class_name", "")), str(args.get("topic", "")), k=self.k)
            if not hits:
                return f"No class named '{args.get('class_name')}' in the Godot 4 class reference."
        else:
            return f"Unknown tool '{call.name}'."
        return self._register(hits)

    # --- loop --------------------------------------------------------------------------

    @staticmethod
    def build_prompt(log: str, errors: list[GodotError], code: str | None) -> str:
        parts = [f"Godot output:\n```\n{log.strip()}\n```"]
        if errors:
            hints = "\n".join(f"- {e.summary()}" for e in errors[:5])
            parts.append(f"Parsed from the log (automatic, may be incomplete):\n{hints}")
        if code:
            parts.append(f"The script involved:\n```gdscript\n{code.strip()}\n```")
        parts.append("Explain this error and how to fix it.")
        return "\n\n".join(parts)

    def run(self, log: str, code: str | None = None) -> Diagnosis:
        self._log = log
        self._numbered.clear()
        self._seen.clear()
        errors = parse_godot_errors(log, self.known_classes)
        messages = [Message("user", self.build_prompt(log, errors, code))]
        steps: list[str] = []

        for _ in range(self.max_steps):
            reply = self.llm.chat(AGENT_SYSTEM, messages, TOOLS)
            messages.append(reply.as_message())
            if not reply.tool_calls:
                return self._finish(reply.text, errors, steps)
            for call in reply.tool_calls:
                steps.append(f"{call.name}({', '.join(f'{k}={v!r}' for k, v in call.arguments.items())})")
                messages.append(Message("tool", self.run_tool(call), tool_call_id=call.id, name=call.name))

        # Out of steps: ask for the answer with what was found. Tools stay declared because
        # some providers reject a history containing tool calls when no tools are defined.
        messages.append(Message("user", FINAL_ANSWER_NUDGE))
        reply = self.llm.chat(AGENT_SYSTEM, messages, TOOLS)
        return self._finish(reply.text or "The agent ran out of steps without an answer.", errors, steps)

    def _finish(self, text: str, errors: list[GodotError], steps: list[str]) -> Diagnosis:
        text = text.strip()
        report = check_text(text, self.api) if self.api else None
        verdict, draft = None, text
        if self.verifier and self._numbered:
            summary = report.summary() if report else "API check: not run."
            question = f"Explain this Godot error and how to fix it:\n{self._log.strip()}"
            verdict = judge(question, self._numbered, text, summary, self.verifier)
            text = verdict.corrected_answer.strip() or text
        numbers = sorted({int(n) for n in CITATION_RE.findall(text)})
        sources = [(n, self._numbered[n - 1]) for n in numbers if 0 < n <= len(self._numbered)]
        return Diagnosis(text, errors, sources, steps, draft, report, verdict)
