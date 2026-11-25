"""Finds the chunks most similar in meaning to a query."""

from __future__ import annotations

import re
from dataclasses import dataclass

from godot_rag.config import Settings
from godot_rag.index import Embedder, open_collection


@dataclass(frozen=True)
class Hit:
    text: str
    title: str
    section: str
    url: str
    kind: str
    page: str
    score: float      # cosine similarity, 1.0 = identical meaning

    @property
    def body(self) -> str:
        """The chunk text without its breadcrumb line."""
        return self.text.split("\n\n", 1)[-1]


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        self.collection = open_collection(settings)
        indexed_with = (self.collection.metadata or {}).get("embedding_model")
        model = indexed_with or settings.embedding_model
        if indexed_with and indexed_with != settings.embedding_model:
            print(f"Note: the index was built with {indexed_with}; using it for queries too.")
        self.embedder = embedder or Embedder(model)

    def search(self, query: str, k: int | None = None, where: dict | None = None) -> list[Hit]:
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k or self.settings.top_k,
            where=where,
        )
        hits = []
        for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hits.append(Hit(text, meta["title"], meta["section"], meta["url"], meta["kind"], meta["page"], 1 - dist))
        return hits

    def search_class(self, class_name: str, topic: str = "", k: int | None = None) -> list[Hit]:
        """Search inside one class reference page (e.g. 'CharacterBody3D')."""
        page = "classes/class_" + re.sub(r"[^a-z0-9_]", "", class_name.lower())
        query = f"{class_name} {topic}".strip()
        return self.search(query, k=k, where={"page": page})
