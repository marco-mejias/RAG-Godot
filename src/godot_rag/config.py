"""Loads config.yaml and .env into one immutable Settings object."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class Settings:
    docs_repo: str
    docs_branch: str
    docs_base_url: str
    docs_include: tuple[str, ...]
    raw_dir: Path
    index_dir: Path
    embedding_model: str
    chunk_chars: int
    chunk_overlap: int
    top_k: int
    provider: str
    models: dict[str, str | list[str]]     # a list = fallback chain (Gemini)
    max_agent_steps: int
    results_per_search: int
    verify_provider: str = "claude"

    def model_for(self, provider: str) -> str | list[str]:
        try:
            return self.models[provider]
        except KeyError:
            raise ValueError(f"No model configured for provider '{provider}' in config.yaml") from None


def load_settings(path: Path | None = None) -> Settings:
    path = path or DEFAULT_CONFIG
    load_dotenv(path.parent / ".env")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent

    return Settings(
        docs_repo=cfg["docs"]["repo"],
        docs_branch=cfg["docs"]["branch"],
        docs_base_url=cfg["docs"]["base_url"].rstrip("/"),
        docs_include=tuple(cfg["docs"]["include"]),
        raw_dir=base / cfg["paths"]["raw"],
        index_dir=base / cfg["paths"]["index"],
        embedding_model=cfg["retrieval"]["embedding_model"],
        chunk_chars=int(cfg["retrieval"]["chunk_chars"]),
        chunk_overlap=int(cfg["retrieval"]["chunk_overlap"]),
        top_k=int(cfg["retrieval"]["top_k"]),
        provider=os.getenv("LLM_PROVIDER") or cfg["llm"]["provider"],
        models=dict(cfg["llm"]["models"]),
        max_agent_steps=int(cfg["agent"]["max_steps"]),
        results_per_search=int(cfg["agent"]["results_per_search"]),
        verify_provider=(cfg.get("verify") or {}).get("provider", "claude"),
    )
