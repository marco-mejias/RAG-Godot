"""Provider-neutral chat interface.

The rest of the app only talks to `LLM.chat()` with these small dataclasses;
each provider module translates them to and from its own SDK types. That's
what makes Gemini, Claude and Ollama interchangeable.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


class LLMError(RuntimeError):
    """A provider failed or declined to answer."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply (tolerates ```json fences and extra prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("The model did not return JSON.")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model returned invalid JSON: {exc}") from exc


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]     # JSON Schema of the arguments


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)   # assistant only
    tool_call_id: str | None = None                            # tool only
    name: str | None = None                                    # tool only: which tool produced it
    # Assistant turns keep the provider's native response so it can be sent back
    # unchanged (needed for thinking blocks / signatures on Claude and Gemini).
    raw: Any = None


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall]
    raw: Any = None

    def as_message(self) -> Message:
        return Message("assistant", self.text, list(self.tool_calls), raw=self.raw)


class LLM(ABC):
    provider: str
    model: str

    @abstractmethod
    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None) -> Reply:
        """Send the conversation and return the model's next turn."""

    def chat_json(self, system: str, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        """Ask for a JSON object matching `schema`. Providers override this with their native
        structured-output mode; this fallback asks for JSON in the prompt and parses it."""
        prompt = f"{system}\n\nReply ONLY with one JSON object matching this JSON Schema:\n{json.dumps(schema)}"
        return parse_json_object(self.chat(prompt, messages).text)

    def __repr__(self) -> str:
        return f"{self.provider}:{self.model}"
