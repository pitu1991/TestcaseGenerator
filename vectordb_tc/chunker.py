"""Structure-aware chunker.

Two fixes vs the design:
1. It is COUPLED to the embedder's max_seq_length and clamps + warns if the
   configured chunk size exceeds it (Req 1.5). No more silent truncation.
2. Tables are kept whole when they fit the budget, else split by rows with the
   header row repeated in each sub-chunk (Req 1.4)."""
from __future__ import annotations

import logging
import re

from embedder import EmbeddingService
from models import Chunk

log = logging.getLogger(__name__)

_HEADING = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_FENCE = re.compile(r"^```")


class Chunker:
    def __init__(self, embedder: EmbeddingService, chunk_size: int = 400, overlap: int = 80):
        self.embedder = embedder
        self.overlap = overlap
        # Clamp to what the model can embed. This is the guard the design lacked.
        if chunk_size > embedder.max_seq_length:
            log.warning(
                "chunk_size=%d exceeds model max_seq_length=%d; clamping to %d",
                chunk_size, embedder.max_seq_length, embedder.max_seq_length,
            )
        self.max_tokens = min(chunk_size, embedder.max_seq_length)

    def _ntok(self, text: str) -> int:
        return self.embedder.count_tokens(text)

    def chunk_document(self, text: str, base_meta: dict) -> list[Chunk]:
        pieces: list[str] = []
        for block in self._split_structural(text):
            if self._is_table(block):
                pieces.extend(self._chunk_table(block))
            elif block.startswith("```"):
                pieces.append(block)  # keep code fences atomic
            elif self._ntok(block) <= self.max_tokens:
                pieces.append(block)
            else:
                pieces.extend(self._token_split(block))

        pieces = [p for p in (s.strip() for s in pieces) if p]
        total = len(pieces)
        return [self._mk_chunk(p, i, total, base_meta) for i, p in enumerate(pieces)]

    # --- structural splitting ------------------------------------------------
    def _split_structural(self, text: str) -> list[str]:
        """Split on headings / fenced blocks / table regions, keeping each intact.
        (Simplified; production version should track heading hierarchy.)"""
        blocks, buf, in_fence = [], [], False
        for line in text.splitlines():
            if _FENCE.match(line):
                in_fence = not in_fence
                buf.append(line)
                if not in_fence:
                    blocks.append("\n".join(buf)); buf = []
                continue
            if not in_fence and _HEADING.match(line) and buf:
                blocks.append("\n".join(buf)); buf = []
            buf.append(line)
        if buf:
            blocks.append("\n".join(buf))
        return blocks

    def _is_table(self, block: str) -> bool:
        rows = [r for r in block.splitlines() if r.strip()]
        return len(rows) >= 2 and sum(bool(_TABLE_ROW.match(r)) for r in rows) >= 2

    def _chunk_table(self, table: str) -> list[str]:
        rows = [r for r in table.splitlines() if r.strip()]
        if self._ntok(table) <= self.max_tokens:
            return [table]                       # keep whole
        header = "\n".join(rows[:2])             # header + separator
        body = rows[2:]
        out, cur = [], [header]
        for row in body:
            trial = "\n".join(cur + [row])
            if self._ntok(trial) > self.max_tokens and len(cur) > 1:
                out.append("\n".join(cur)); cur = [header, row]   # repeat header
            else:
                cur.append(row)
        if len(cur) > 1:
            out.append("\n".join(cur))
        return out

    def _token_split(self, text: str) -> list[str]:
        """Word-window fallback for long prose, approximating the token budget."""
        words, out, step = text.split(), [], max(1, self.max_tokens - self.overlap)
        # ~0.75 words/token heuristic to size the window in words
        win = max(1, int(self.max_tokens * 0.75))
        stride = max(1, int(step * 0.75))
        for i in range(0, len(words), stride):
            out.append(" ".join(words[i:i + win]))
        return out

    def _mk_chunk(self, text: str, idx: int, total: int, base: dict) -> Chunk:
        return Chunk(
            id=f"{base['content_hash'][:12]}_{idx}",
            text=text,
            source_path=base["source_path"],
            document_title=base.get("document_title", ""),
            chunk_index=idx,
            total_chunks=total,
            category=base["category"],
            ingestion_timestamp=base["ingestion_timestamp"],
            content_hash=base["content_hash"],
            embedding_model=base["embedding_model"],
            model_version=base["model_version"],
            metadata=base.get("metadata", {}),
        )
