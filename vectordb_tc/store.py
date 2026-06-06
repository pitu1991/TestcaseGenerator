"""ChromaDB persistence. Adds a keyword_search (literal substring match via
where_document) used by the hybrid retriever, and surfaces which embedding
model the stored chunks were built with (for model-change detection)."""
from __future__ import annotations

from models import Chunk, SearchResult


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str = "project_knowledge"):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[self._meta(c) for c in chunks],
        )

    def search(self, query_embedding, top_k=10, where=None) -> list[SearchResult]:
        res = self._col.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_results(res, match_type="dense", distance=True)

    def keyword_search(self, term: str, top_k=20) -> list[SearchResult]:
        """Literal substring match -> reliable for identifiers like PY075 / TWCJ-6184."""
        res = self._col.get(
            where_document={"$contains": term}, limit=top_k,
            include=["documents", "metadatas"],
        )
        out = []
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            out.append(SearchResult(self._chunk(cid, doc, meta), 1.0, "keyword"))
        return out

    def delete_by_source(self, source_path: str) -> int:
        existing = self._col.get(where={"source_path": source_path}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def hash_for_source(self, source_path: str) -> str | None:
        r = self._col.get(where={"source_path": source_path}, limit=1, include=["metadatas"])
        metas = r.get("metadatas") or []
        return metas[0].get("content_hash") if metas else None

    def stored_model_versions(self) -> set[str]:
        r = self._col.get(include=["metadatas"])
        return {m.get("model_version", "") for m in (r.get("metadatas") or [])}

    def issue_keys_by_category(self, category: str) -> set[str]:
        r = self._col.get(where={"category": category}, include=["metadatas"])
        return {m.get("issue_key") for m in (r.get("metadatas") or []) if m.get("issue_key")}

    def get_all_sources(self) -> set[str]:
        """Return every distinct source_path stored in the collection."""
        r = self._col.get(include=["metadatas"])
        return {m.get("source_path", "") for m in (r.get("metadatas") or []) if m.get("source_path")}

    def get_stats(self) -> dict:
        r = self._col.get(include=["metadatas"])
        metas = r.get("metadatas") or []
        by_cat: dict[str, int] = {}
        for m in metas:
            by_cat[m.get("category", "?")] = by_cat.get(m.get("category", "?"), 0) + 1
        return {"total_chunks": len(metas), "by_category": by_cat}

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _meta(c: Chunk) -> dict:
        m = {
            "source_path": c.source_path, "document_title": c.document_title,
            "chunk_index": c.chunk_index, "total_chunks": c.total_chunks,
            "category": c.category, "ingestion_timestamp": c.ingestion_timestamp,
            "content_hash": c.content_hash, "embedding_model": c.embedding_model,
            "model_version": c.model_version,
        }
        m.update({k: v for k, v in c.metadata.items() if v is not None})
        return m

    def _to_results(self, res, match_type, distance=False) -> list[SearchResult]:
        out = []
        ids = res["ids"][0]; docs = res["documents"][0]
        metas = res["metadatas"][0]; dists = res.get("distances", [[None]*len(ids)])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - dist if (distance and dist is not None) else 1.0
            out.append(SearchResult(self._chunk(cid, doc, meta), score, match_type))
        return out

    @staticmethod
    def _chunk(cid, doc, meta) -> Chunk:
        return Chunk(
            id=cid, text=doc, source_path=meta.get("source_path", ""),
            document_title=meta.get("document_title", ""),
            chunk_index=meta.get("chunk_index", 0), total_chunks=meta.get("total_chunks", 1),
            category=meta.get("category", "?"), ingestion_timestamp=meta.get("ingestion_timestamp", ""),
            content_hash=meta.get("content_hash", ""), embedding_model=meta.get("embedding_model", ""),
            model_version=meta.get("model_version", ""),
            metadata={k: v for k, v in meta.items() if k not in {
                "source_path", "document_title", "chunk_index", "total_chunks", "category",
                "ingestion_timestamp", "content_hash", "embedding_model", "model_version"}},
        )
