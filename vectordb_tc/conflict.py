"""Conflict candidate detection (Phase C).

No LLM here — by design. With no in-server model, detection is the cheap,
deterministic half: pure vector similarity finds chunks from *different* sources
that talk about the same thing, and records them as 'suspected' candidates. The
host LLM (in the IDE) does the expensive half later — deciding whether the two
chunks actually contradict — via the record_verdict MCP tool.

High similarity != conflict. Two chunks can be near-identical and agree. So this
stage only proposes pairs for adjudication; it never confirms truth."""
from __future__ import annotations


class ConflictDetector:
    def __init__(self, store, governance, threshold: float = 0.83,
                 project_id: str = "default", top_k: int = 5, notifier=None):
        self.store = store
        self.governance = governance
        self.threshold = threshold
        self.project_id = project_id
        self.top_k = top_k
        # Phase D: alert reviewers when a conflict is flagged. Defaults to no-op.
        if notifier is None:
            from notifier import NullNotifier
            notifier = NullNotifier()
        self.notifier = notifier

    def scan(self, chunks, embeddings) -> list:
        """For each freshly stored chunk, find similar latest chunks from other
        documents above the threshold and record suspected conflicts. Resolutions
        are never scanned (they are the authoritative answer, not a candidate)."""
        created = []
        for c, emb in zip(chunks, embeddings):
            if c.category == "Business_Resolution":
                continue
            for hit in self.store.search(emb, top_k=self.top_k, where={"is_latest": True}):
                other = hit.chunk
                if other.id == c.id or other.document_id == c.document_id:
                    continue
                if other.category == "Business_Resolution":
                    continue
                if hit.relevance_score < self.threshold:
                    continue
                if self.governance.exists_open_conflict(c.id, other.id):
                    continue
                rec = self.governance.create_conflict(
                    project_id=self.project_id,
                    module=c.module or other.module,
                    chunk_a_id=c.id, chunk_b_id=other.id,
                    source_a=c.source_path, source_b=other.source_path,
                    similarity=round(float(hit.relevance_score), 4),
                )
                self.notifier.conflict_created(rec)
                created.append(rec)
        return created
