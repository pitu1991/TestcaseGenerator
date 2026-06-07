"""Failure intelligence (Phase E).

Turns dead test-failure artifacts (error messages, stack traces, screenshots)
into a searchable memory: when a new test fails, retrieve the most similar past
failures so the host LLM can surface a likely root cause and fix instead of
re-debugging the same race condition for the sixth time.

Design notes (how this fits the rest of the platform):
- LOCAL-FIRST, no cloud/API keys. Same embedder + ChromaStore the rest of the
  pipeline uses; no Astra/OpenAI/LangChain. Screenshots/blobs are NOT stored in
  the vector DB — only their path/URL goes in metadata (the vector is the index,
  the blob lives in object storage).
- ONE vector per failure. Unlike documents, a failure is not chunked: fragmenting
  a stack trace would split the signal across rows and ruin similarity. Each
  failure is a single Chunk in category "Failure".
- The embedding key is the ERROR SIGNATURE (test_name + error_message +
  stack_trace). root_cause / fix_commit / assignee are human annotations attached
  after triage and held as metadata only, so an outdated annotation can never
  drift the search vector (Trap 4 in the field: stale embeddings are worse than
  none).
- IDEMPOTENT + recurrence-aware. Identical failures hash to the same failure_id,
  so re-recording bumps `occurrences` and `last_seen` rather than duplicating —
  that count is exactly the "this keeps reappearing" signal.
- Authority 10 (see models.AUTHORITY_DEFAULTS): failures are diagnostic, never
  ground truth, so they can't pollute test-case generation context.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from config import AppConfig
from embedder import EmbeddingService
from models import Chunk, FailureArtifact, authority_for
from store import ChromaStore

_CATEGORY = "Failure"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FailureAnalyzer:
    def __init__(self, embedder: EmbeddingService, store: ChromaStore, config: AppConfig):
        self.embedder, self.store, self.config = embedder, store, config

    # --- write ---------------------------------------------------------------
    def record_failure(self, test_name: str, error_message: str, stack_trace: str = "",
                        run_id: str = "", status: str = "failed", screenshot_path: str = "",
                        project: str = "default", extra: dict | None = None) -> FailureArtifact:
        """Store (or update) a failure as a single Failure chunk. Recurrences of an
        identical error collapse onto the same failure_id and increment occurrences,
        preserving the first_seen timestamp and any triage annotations."""
        sig = self._signature(test_name, error_message, stack_trace)
        failure_id = f"fail_{sig[:12]}"
        now = _now()

        prior = self.get_failure(failure_id)
        if prior is not None:
            occurrences = prior.occurrences + 1
            first_seen = prior.first_seen or now
            # Keep triage annotations across recurrences (the search vector is the
            # error signature, so they ride along unchanged in metadata).
            root_cause = prior.root_cause
            fix_commit = prior.fix_commit
            assignee = prior.assignee
        else:
            occurrences, first_seen = 1, now
            root_cause = fix_commit = assignee = ""

        art = FailureArtifact(
            failure_id=failure_id, test_name=test_name, error_message=error_message,
            stack_trace=stack_trace, run_id=run_id, status=status,
            screenshot_path=screenshot_path, project=project, root_cause=root_cause,
            fix_commit=fix_commit, assignee=assignee, occurrences=occurrences,
            first_seen=first_seen, last_seen=now,
        )
        self._upsert(art, sig, extra or {})
        return art

    def annotate(self, failure_id: str, root_cause: str = "", fix_commit: str = "",
                 assignee: str = "") -> bool:
        """Attach triage outcome (root cause / fix commit / assignee) to a stored
        failure. Metadata-only — does not change the search vector. Returns False
        if the failure_id is unknown."""
        updates = {"root_cause": root_cause or None, "fix_commit": fix_commit or None,
                   "assignee": assignee or None}
        if not any(v for v in updates.values()):
            return self.get_failure(failure_id) is not None
        return self.store.update_chunk_metadata(failure_id, updates)

    # --- read ----------------------------------------------------------------
    def find_similar(self, error_text: str, top_k: int = 3, min_similarity: float | None = None,
                     project: str | None = None) -> list[tuple[FailureArtifact, float]]:
        """Most similar past failures to a new error, newest-quality first. Returns
        (artifact, similarity) pairs at or above the threshold so the host LLM can
        reuse a known root cause/fix. Filtered to category=Failure (and project)."""
        threshold = self.config.failure_similarity_threshold if min_similarity is None else min_similarity
        if self.store.get_stats()["total_chunks"] == 0:
            return []
        conds: list[dict] = [{"category": _CATEGORY}, {"is_latest": True}]
        if project:
            conds.append({"failure_project": project})
        where = conds[0] if len(conds) == 1 else {"$and": conds}
        hits = self.store.search(self.embedder.embed_query(error_text), top_k=top_k, where=where)
        out = [(self._to_artifact(r.chunk), round(r.relevance_score, 4))
               for r in hits if r.relevance_score >= threshold]
        return out[:top_k]

    def get_failure(self, failure_id: str) -> FailureArtifact | None:
        chunks = self.store.get_chunks_by_ids([failure_id])
        return self._to_artifact(chunks[0]) if chunks else None

    def stats(self, project: str | None = None) -> dict:
        """Aggregate failure stats: total distinct failures and the most-recurring
        ones (the chronic offenders worth fixing first)."""
        extra = {"failure_project": project} if project else None
        arts = [self._to_artifact(c) for c in self.store.chunks_by_category(_CATEGORY, extra)]
        top = sorted(arts, key=lambda a: a.occurrences, reverse=True)[:5]
        return {
            "total_failures": len(arts),
            "total_occurrences": sum(a.occurrences for a in arts),
            "top_recurring": [{"failure_id": a.failure_id, "test_name": a.test_name,
                               "occurrences": a.occurrences,
                               "root_cause": a.root_cause or "(untriaged)"} for a in top],
        }

    # --- internals -----------------------------------------------------------
    @staticmethod
    def _signature(test_name: str, error_message: str, stack_trace: str) -> str:
        return hashlib.sha256(f"{test_name}\n{error_message}\n{stack_trace}".encode()).hexdigest()

    @staticmethod
    def _embed_text(test_name: str, error_message: str, stack_trace: str) -> str:
        # The searchable error signature. Stack trace tail carries the most signal;
        # keep it bounded so one giant trace doesn't dominate the embedding budget.
        trace = stack_trace.strip()
        if len(trace) > 2000:
            trace = trace[-2000:]
        return "\n".join(p for p in (test_name, error_message, trace) if p)

    def _upsert(self, art: FailureArtifact, sig: str, extra: dict) -> None:
        text = self._embed_text(art.test_name, art.error_message, art.stack_trace)
        meta = {
            "source_path": f"failure:{art.failure_id}",
            "document_title": art.test_name,
            "content_hash": sig,
            "category": _CATEGORY,
            "ingestion_timestamp": art.last_seen,
            "embedding_model": self.embedder.model_name,
            "model_version": self.embedder.model_version,
            "document_id": art.failure_id, "version": 1, "is_latest": True,
            "module": art.project, "authority_score": authority_for(_CATEGORY),
            # failure-specific fields (non-reserved -> round-trip into chunk.metadata)
            "metadata": {
                "test_name": art.test_name, "error_message": art.error_message,
                "stack_trace": art.stack_trace, "run_id": art.run_id,
                "failure_status": art.status, "screenshot_path": art.screenshot_path,
                "failure_project": art.project, "root_cause": art.root_cause,
                "fix_commit": art.fix_commit, "assignee": art.assignee,
                "occurrences": art.occurrences, "first_seen": art.first_seen,
                "last_seen": art.last_seen, **extra,
            },
        }
        chunk = Chunk(
            id=art.failure_id, text=text, source_path=meta["source_path"],
            document_title=art.test_name, chunk_index=0, total_chunks=1,
            category=_CATEGORY, ingestion_timestamp=art.last_seen, content_hash=sig,
            embedding_model=self.embedder.model_name, model_version=self.embedder.model_version,
            metadata=meta["metadata"], document_id=art.failure_id, version=1, is_latest=True,
            module=art.project, chunk_hash=sig, authority_score=authority_for(_CATEGORY),
        )
        embeddings = self.embedder.embed_texts([text])
        self.store.upsert_chunks([chunk], embeddings)

    @staticmethod
    def _to_artifact(chunk: Chunk) -> FailureArtifact:
        m = chunk.metadata or {}
        return FailureArtifact(
            failure_id=chunk.id, test_name=m.get("test_name", chunk.document_title),
            error_message=m.get("error_message", ""), stack_trace=m.get("stack_trace", ""),
            run_id=m.get("run_id", ""), status=m.get("failure_status", "failed"),
            screenshot_path=m.get("screenshot_path", ""),
            project=m.get("failure_project", chunk.module or "default"),
            root_cause=m.get("root_cause", ""), fix_commit=m.get("fix_commit", ""),
            assignee=m.get("assignee", ""), occurrences=int(m.get("occurrences", 1)),
            first_seen=m.get("first_seen", ""), last_seen=m.get("last_seen", ""),
        )
