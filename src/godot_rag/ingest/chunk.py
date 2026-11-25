"""Splits parsed pages into chunks small enough to embed and to quote to the LLM.

Rules:
- A chunk never crosses a section boundary, so every chunk has one clear
  "Page > Section" location and one URL.
- Long sections are split on paragraph boundaries; the last paragraph(s) of a
  chunk are repeated at the start of the next one (overlap) so an idea that
  straddles the cut is still found.
- Every chunk starts with its breadcrumb, which gives the embedding model
  context ("CharacterBody3D > Methods" says a lot about a bare code line).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from godot_rag.ingest.parse import Page


@dataclass
class Chunk:
    id: str
    text: str          # breadcrumb + body, what gets embedded and shown
    page: str          # rel_path, e.g. "classes/class_node"
    title: str         # page title
    section: str       # breadcrumb of the section
    url: str           # link to the exact section
    kind: str          # "class" | "manual"


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split a paragraph bigger than max_chars on line breaks, then hard-cut."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    parts, current = [], ""
    for line in paragraph.split("\n"):
        while len(line) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if current and len(current) + 1 + len(line) > max_chars:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    paragraphs: list[str] = []
    for p in text.split("\n\n"):
        if p.strip():
            paragraphs.extend(_split_long(p.strip("\n"), max_chars))

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for p in paragraphs:
        if current and size + len(p) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            # carry the tail paragraphs that fit in the overlap budget
            tail, tail_size = [], 0
            for prev in reversed(current):
                if tail_size + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 2
            current, size = tail, tail_size
        current.append(p)
        size += len(p) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_page(page: Page, max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in page.sections:
        breadcrumb = " > ".join(section.path)
        url = f"{page.url}#{section.anchor}" if section.anchor else page.url
        for n, body in enumerate(split_text(section.text, max_chars, overlap)):
            text = f"{breadcrumb}\n\n{body}"
            digest = hashlib.sha1(f"{page.rel_path}|{breadcrumb}|{n}|{body}".encode()).hexdigest()[:16]
            chunks.append(Chunk(digest, text, page.rel_path, page.title, breadcrumb, url, page.kind))
    return chunks
