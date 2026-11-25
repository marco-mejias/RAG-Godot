"""Embeds chunks and stores them in a local Chroma vector database."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from godot_rag.config import Settings
from godot_rag.ingest.chunk import Chunk

COLLECTION = "godot_docs"

# BGE models are trained to expect this prefix on *queries* (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Wraps a sentence-transformers model: text -> normalized vector."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # slow import, keep it lazy

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=show_progress
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(self.query_prefix + query, normalize_embeddings=True).tolist()


def open_client(settings: Settings):
    import chromadb

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.index_dir))


def open_collection(settings: Settings):
    client = open_client(settings)
    try:
        return client.get_collection(COLLECTION)
    except Exception as exc:  # chroma raises different errors across versions
        raise RuntimeError("The index does not exist yet. Run: python -m godot_rag ingest") from exc


def build_index(settings: Settings, chunks: Iterable[Chunk], embedder: Embedder, batch_size: int = 512) -> int:
    client = open_client(settings)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
    )

    chunks = list(chunks)
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embedder.embed_documents([c.text for c in batch]),
            metadatas=[
                {"page": c.page, "title": c.title, "section": c.section, "url": c.url, "kind": c.kind}
                for c in batch
            ],
        )
        print(f"  indexed {min(start + batch_size, total)}/{total} chunks", flush=True)
    return total
