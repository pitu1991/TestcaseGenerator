"""ChromaDB persistence. Adds a keyword_search (literal substring match via
where_document) used by the hybrid retriever, and surfaces which embedding
model the stored chunks were built with (for model-change detection)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from models import Chunk, SearchResult, authority_for


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def keyword_search(self, term: str, top_k=20, where=None) -> list[SearchResult]:
        """Literal substring match -> reliable for identifiers like PY075 / TWCJ-6184.
        `where` applies the same metadata filter (e.g. is_latest) as dense search."""
        res = self._col.get(
            where=where, where_document={"$contains": term}, limit=top_k,
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

    # --- versioning (Phase A) ------------------------------------------------
    def latest_version_for_document(self, document_id: str) -> tuple[int, str] | None:
        """Return (version, content_hash) of the is_latest chunks for a document,
        or None if the document has never been ingested."""
        r = self._col.get(
            where={"$and": [{"document_id": document_id}, {"is_latest": True}]},
            limit=1, include=["metadatas"],
        )
        metas = r.get("metadatas") or []
        if not metas:
            return None
        m = metas[0]
        return int(m.get("version", 1)), m.get("content_hash", "")

    def mark_not_latest(self, document_id: str) -> int:
        """Flip is_latest -> False (and stamp superseded_at) on a document's current
        latest chunks. Metadata-only update; does NOT re-embed and does NOT delete."""
        r = self._col.get(
            where={"$and": [{"document_id": document_id}, {"is_latest": True}]},
            include=["metadatas"],
        )
        ids = r.get("ids", [])
        if not ids:
            return 0
        now = _now()
        new_metas = []
        for m in (r.get("metadatas") or []):
            mm = dict(m)
            mm["is_latest"] = False
            mm["superseded_at"] = now
            new_metas.append(mm)
        self._col.update(ids=ids, metadatas=new_metas)
        return len(ids)

    def versions_for_document(self, document_id: str) -> list[int]:
        """All stored version numbers for a document, newest first."""
        r = self._col.get(where={"document_id": document_id}, include=["metadatas"])
        return sorted({int(m.get("version", 1)) for m in (r.get("metadatas") or [])},
                      reverse=True)

    def chunks_for_document(self, document_id: str, version: int | None = None) -> list[Chunk]:
        """Chunks for one version of a document (latest if version is None), ordered
        by chunk_index. Used by delta diffing."""
        if version is None:
            where = {"$and": [{"document_id": document_id}, {"is_latest": True}]}
        else:
            where = {"$and": [{"document_id": document_id}, {"version": version}]}
        r = self._col.get(where=where, include=["documents", "metadatas"])
        chunks = [self._chunk(cid, doc, meta) for cid, doc, meta in
                  zip(r.get("ids", []), r.get("documents") or [], r.get("metadatas") or [])]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def prune_old_versions(self, document_id: str, max_versions: int) -> int:
        """Delete chunks belonging to all but the `max_versions` most-recent versions
        of a document. Returns the number of chunks removed."""
        if max_versions <= 0:
            return 0
        r = self._col.get(where={"document_id": document_id}, include=["metadatas"])
        ids = r.get("ids", [])
        metas = r.get("metadatas") or []
        versions = sorted({int(m.get("version", 1)) for m in metas}, reverse=True)
        if len(versions) <= max_versions:
            return 0
        keep = set(versions[:max_versions])
        drop = [cid for cid, m in zip(ids, metas) if int(m.get("version", 1)) not in keep]
        if drop:
            self._col.delete(ids=drop)
        return len(drop)

    def backfill_versioning(self) -> int:
        """One-time, idempotent migration: stamp version/is_latest/document_id/
        chunk_hash/authority_score on any chunk that predates Phase A. Metadata-only
        (no re-embed). Safe to run repeatedly — already-stamped chunks are skipped.
        MUST be run before the is_latest default filter is relied upon."""
        r = self._col.get(include=["documents", "metadatas"])
        ids = r.get("ids", [])
        docs = r.get("documents") or []
        metas = r.get("metadatas") or []
        up_ids, up_metas = [], []
        for cid, doc, m in zip(ids, docs, metas):
            if m.get("version") is not None:
                continue  # already migrated
            mm = dict(m)
            mm["version"] = 1
            mm["is_latest"] = True
            mm["document_id"] = m.get("source_path", "")
            mm["module"] = mm.get("module", "default")
            mm["section"] = mm.get("section", "")
            mm["chunk_hash"] = hashlib.sha256((doc or "").encode()).hexdigest()
            mm["superseded_at"] = ""
            mm["authority_score"] = authority_for(m.get("category", ""))
            up_ids.append(cid)
            up_metas.append(mm)
        if up_ids:
            self._col.update(ids=up_ids, metadatas=up_metas)
        return len(up_ids)

    def update_chunk_metadata(self, chunk_id: str, updates: dict) -> bool:
        """Merge `updates` into one chunk's metadata (no re-embed). Used by the
        failure analyzer to attach root_cause/fix_commit after triage. Returns
        False if the chunk does not exist. None values are ignored."""
        r = self._col.get(ids=[chunk_id], include=["metadatas"])
        metas = r.get("metadatas") or []
        if not metas:
            return False
        mm = dict(metas[0])
        mm.update({k: v for k, v in updates.items() if v is not None})
        self._col.update(ids=[chunk_id], metadatas=[mm])
        return True

    def stored_model_versions(self) -> set[str]:
        r = self._col.get(include=["metadatas"])
        return {m.get("model_version", "") for m in (r.get("metadatas") or [])}

    def chunks_by_category(self, category: str, where_extra: dict | None = None) -> list[Chunk]:
        """All chunks in a category (optionally narrowed by extra metadata equality,
        e.g. {'failure_project': 'web'}). Used by failure-stats aggregation."""
        if where_extra:
            where = {"$and": [{"category": category}] + [{k: v} for k, v in where_extra.items()]}
        else:
            where = {"category": category}
        r = self._col.get(where=where, include=["documents", "metadatas"])
        return [self._chunk(cid, doc, meta) for cid, doc, meta in
                zip(r.get("ids", []), r.get("documents") or [], r.get("metadatas") or [])]

    def issue_keys_by_category(self, category: str) -> set[str]:
        r = self._col.get(where={"category": category}, include=["metadatas"])
        return {m.get("issue_key") for m in (r.get("metadatas") or []) if m.get("issue_key")}

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Fetch specific chunks by id (used to resolve conflict chunk references)."""
        if not ids:
            return []
        r = self._col.get(ids=ids, include=["documents", "metadatas"])
        return [self._chunk(cid, doc, meta) for cid, doc, meta in
                zip(r.get("ids", []), r.get("documents") or [], r.get("metadatas") or [])]

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
            "document_id": c.document_id, "version": c.version, "is_latest": c.is_latest,
            "module": c.module, "section": c.section, "chunk_hash": c.chunk_hash,
            "superseded_at": c.superseded_at, "authority_score": c.authority_score,
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

    _RESERVED = {
        "source_path", "document_title", "chunk_index", "total_chunks", "category",
        "ingestion_timestamp", "content_hash", "embedding_model", "model_version",
        "document_id", "version", "is_latest", "module", "section", "chunk_hash",
        "superseded_at", "authority_score",
    }

    @classmethod
    def _chunk(cls, cid, doc, meta) -> Chunk:
        return Chunk(
            id=cid, text=doc, source_path=meta.get("source_path", ""),
            document_title=meta.get("document_title", ""),
            chunk_index=meta.get("chunk_index", 0), total_chunks=meta.get("total_chunks", 1),
            category=meta.get("category", "?"), ingestion_timestamp=meta.get("ingestion_timestamp", ""),
            content_hash=meta.get("content_hash", ""), embedding_model=meta.get("embedding_model", ""),
            model_version=meta.get("model_version", ""),
            metadata={k: v for k, v in meta.items() if k not in cls._RESERVED},
            document_id=meta.get("document_id", ""), version=int(meta.get("version", 1)),
            is_latest=bool(meta.get("is_latest", True)), module=meta.get("module", "default"),
            section=meta.get("section", ""), chunk_hash=meta.get("chunk_hash", ""),
            superseded_at=meta.get("superseded_at", ""),
            authority_score=int(meta.get("authority_score", 0)),
        )
