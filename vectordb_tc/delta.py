"""Delta generation (Phase B).

Compares two versions of a document at the SECTION level using the per-chunk
hashes stamped in Phase A. The whole-file content_hash only tells you *that* a
document changed; chunk_hash grouped by section tells you *which* parts changed,
so test regeneration can target just the added/changed sections instead of
re-running the entire document.

This module does no generation itself (consistent with the project principle) —
it produces a structured DocumentDelta the host LLM acts on via the MCP tools."""
from __future__ import annotations

import hashlib

from models import DeltaSection, DocumentDelta
from store import ChromaStore


class DeltaEngine:
    def __init__(self, store: ChromaStore):
        self.store = store

    def diff_versions(self, document_id: str, from_version: int | None = None,
                      to_version: int | None = None) -> DocumentDelta:
        """Section-level diff between two versions. Defaults to (previous -> latest).
        If the document has only one version, every section is reported as added."""
        versions = self.store.versions_for_document(document_id)
        if not versions:
            return DocumentDelta(document_id, 0, 0, [], [])

        to_v = to_version or versions[0]
        if from_version is not None:
            from_v = from_version
        else:
            older = [v for v in versions if v < to_v]
            from_v = older[0] if older else 0          # 0 -> no prior version

        new_secs = self._sections(self.store.chunks_for_document(document_id, to_v))
        old_secs = (self._sections(self.store.chunks_for_document(document_id, from_v))
                    if from_v else {})

        sections: list[DeltaSection] = []
        unchanged: list[str] = []
        for name in sorted(set(new_secs) | set(old_secs)):
            old = old_secs.get(name)
            new = new_secs.get(name)
            if new and not old:
                sections.append(DeltaSection(name, "added", new_text=new[0]))
            elif old and not new:
                sections.append(DeltaSection(name, "removed", old_text=old[0]))
            elif old[1] != new[1]:                      # combined-hash differs
                sections.append(DeltaSection(name, "changed",
                                             old_text=old[0], new_text=new[0]))
            else:
                unchanged.append(name)
        return DocumentDelta(document_id, from_v, to_v, sections, unchanged)

    @staticmethod
    def _sections(chunks) -> dict[str, tuple[str, str]]:
        """Group chunks by section -> (combined_text, combined_hash), ordered by
        chunk_index so a reorder within a section is detectable as a change."""
        bucket: dict[str, list] = {}
        for c in chunks:
            bucket.setdefault(c.section, []).append(c)
        out: dict[str, tuple[str, str]] = {}
        for name, cs in bucket.items():
            cs.sort(key=lambda c: c.chunk_index)
            text = "\n".join(c.text for c in cs)
            out[name] = (text, hashlib.sha256(text.encode()).hexdigest())
        return out
