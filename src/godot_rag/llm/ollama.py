"""Ollama provider: open models running locally (no API key, no internet)."""

from __future__ import annotations

import os

import ollama

from godot_rag.llm.base import LLM, LLMError, Message, Reply, ToolCall, ToolSpec


class OllamaLLM(LLM):
    provider = "ollama"

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    @staticmethod
    def _to_api(system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "content": m.content, "tool_name": m.name})
            elif m.role == "assistant":
                out.append({
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls],
                })
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None) -> Reply:
        api_tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools or []
        ]
        try:
            response = self.client.chat(
                model=self.model, messages=self._to_api(system, messages), tools=api_tools or None
            )
        except ollama.ResponseError as exc:
            raise LLMError(f"Ollama: {exc.error} (did you run `ollama pull {self.model}`?)") from exc
        except ConnectionError as exc:
            raise LLMError("Ollama is not running. Install it from https://ollama.com and start it.") from exc

        msg = response.message
        calls = [
            ToolCall(f"call_{i}", tc.function.name, dict(tc.function.arguments or {}))
            for i, tc in enumerate(msg.tool_calls or [])
        ]
        return Reply(msg.content or "", calls)
