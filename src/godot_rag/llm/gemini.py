"""Gemini (Google AI Studio) provider."""

from __future__ import annotations

import os

from google import genai
from google.genai import errors, types

from godot_rag.llm.base import LLM, LLMError, Message, Reply, ToolCall, ToolSpec


class GeminiLLM(LLM):
    provider = "gemini"

    def __init__(self, model: str):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("Gemini: GEMINI_API_KEY is not set (see .env.example)")
        self.model = model
        self.client = genai.Client(api_key=key)

    @staticmethod
    def _to_api(messages: list[Message]) -> list[types.Content]:
        out: list[types.Content] = []
        for m in messages:
            if m.role == "user":
                out.append(types.Content(role="user", parts=[types.Part(text=m.content)]))
            elif m.role == "tool":
                part = types.Part.from_function_response(name=m.name or "tool", response={"result": m.content})
                # Answers to parallel calls go together in one user turn.
                if out and out[-1].role == "user" and out[-1].parts and out[-1].parts[0].function_response:
                    out[-1].parts.append(part)
                else:
                    out.append(types.Content(role="user", parts=[part]))
            elif m.raw is not None:
                out.append(m.raw)   # unchanged: keeps Gemini's thought signatures
            else:
                parts = [types.Part(text=m.content)] if m.content else []
                parts += [
                    types.Part(function_call=types.FunctionCall(name=c.name, args=c.arguments)) for c in m.tool_calls
                ]
                out.append(types.Content(role="model", parts=parts))
        return out

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec] | None = None) -> Reply:
        config = types.GenerateContentConfig(
            system_instruction=system,
            # We run the tool loop ourselves (same loop for every provider).
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tools=[
                types.Tool(function_declarations=[
                    types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                    for t in tools
                ])
            ] if tools else None,
        )
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=self._to_api(messages), config=config
            )
        except errors.ClientError as exc:
            raise LLMError(f"Gemini request failed ({exc.code}): {exc.message}") from exc
        except errors.ServerError as exc:
            raise LLMError(f"Gemini server error ({exc.code}), try again later") from exc

        if not response.candidates or response.candidates[0].content is None:
            raise LLMError("Gemini returned no answer (possibly blocked by its safety filters).")
        content = response.candidates[0].content
        parts = content.parts or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [
            ToolCall(p.function_call.id or f"{p.function_call.name}_{i}", p.function_call.name, dict(p.function_call.args or {}))
            for i, p in enumerate(parts) if p.function_call
        ]
        return Reply(text, calls, raw=content)
