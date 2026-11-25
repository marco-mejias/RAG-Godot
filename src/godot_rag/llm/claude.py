"""Claude (Anthropic API) provider."""

from __future__ import annotations

import anthropic

from godot_rag.llm.base import LLM, LLMError, Message, Reply, ToolCall, ToolSpec

# Models that accept server-side refusal fallbacks ("default" routes by refusal category).
FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5")


class ClaudeLLM(LLM):
    provider = "claude"

    def __init__(self, model: str, effort: str = "medium"):
        self.model = model
        self.effort = effort
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

    @staticmethod
    def _to_api(messages: list[Message]) -> list[dict]:
        out: list[dict] = []

        def add_user_block(block: dict) -> None:
            # Tool results (and any text after them) must share one user turn.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})

        for m in messages:
            if m.role == "user":
                add_user_block({"type": "text", "text": m.content})
            elif m.role == "tool":
                add_user_block({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content})
            elif m.raw is not None:
                out.append({"role": "assistant", "content": m.raw})   # unchanged, keeps thinking blocks
            else:
                content: list[dict] = [{"type": "text", "text": m.content}] if m.content else []
                content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                out.append({"role": "assistant", "content": content})
        return out

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None) -> Reply:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": self._to_api(messages),
            "output_config": {"effort": self.effort},
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        if self.model.startswith(FALLBACK_MODELS):
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        try:
            response = self.client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude: invalid or missing ANTHROPIC_API_KEY (see .env.example)") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude: rate limit reached, try again in a moment") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Claude: could not reach the API (network?)") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this request.")

        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        return Reply(text, calls, raw=response.content)
