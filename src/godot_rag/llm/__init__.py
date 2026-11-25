"""Swappable LLM providers behind one interface (see base.py)."""

from __future__ import annotations

from godot_rag.config import Settings
from godot_rag.llm.base import LLM, LLMError, Message, Reply, ToolCall, ToolSpec

PROVIDERS = ("gemini", "claude", "ollama")


def get_llm(settings: Settings, provider: str | None = None) -> LLM:
    """Build the configured provider. SDKs are imported lazily so you only need the one you use."""
    provider = (provider or settings.provider).lower()
    model = settings.model_for(provider)
    if provider == "gemini":
        from godot_rag.llm.gemini import GeminiLLM
        return GeminiLLM(model)
    first = model[0] if isinstance(model, list) else model   # only Gemini uses a fallback chain
    if provider == "claude":
        from godot_rag.llm.claude import ClaudeLLM
        return ClaudeLLM(first)
    if provider == "ollama":
        from godot_rag.llm.ollama import OllamaLLM
        return OllamaLLM(first)
    raise ValueError(f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}")


__all__ = ["LLM", "LLMError", "Message", "Reply", "ToolCall", "ToolSpec", "PROVIDERS", "get_llm"]
