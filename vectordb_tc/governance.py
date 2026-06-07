"""Governance store (Phase C).

Conflicts and human-approved resolutions live in SQLite — relational records, not
vectors — so they stay queryable, auditable, and portable to Postgres later with
no code change beyond the connection string. Local-first: a single file under
project_root, stdlib sqlite3, no server.

The LLM never writes truth here. Detection writes 'suspected' candidates; the host
LLM adjudicates (confirmed/dismissed) via record_verdict; a human approves the
resolution. This module only persists those decisions."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from models import ConflictRecord, ResolutionArtifact

OPEN_STATUSES = ("suspected", "confirmed")   # conflicts that still gate generation


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class GovernanceStore:
    def __init__(self, db_path: str):
        # check_same_thread=False: the MCP server and a future review UI may touch
        # this from different threads; writes are short and serialized by SQLite.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conflicts (
                conflict_id      TEXT PRIMARY KEY,
                project_id       TEXT NOT NULL,
                module           TEXT,
                status           TEXT NOT NULL,
                priority         TEXT,
                created_at       TEXT NOT NULL,
                chunk_a_id       TEXT NOT NULL,
                chunk_b_id       TEXT NOT NULL,
                source_a         TEXT,
                source_b         TEXT,
                similarity       REAL,
                assigned_reviewer TEXT DEFAULT '',
                rationale        TEXT DEFAULT '',
                resolution_id    TEXT DEFAULT '',
                pair_key         TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status);
            CREATE INDEX IF NOT EXISTS idx_conflicts_pair   ON conflicts(pair_key);

            CREATE TABLE IF NOT EXISTS resolutions (
                resolution_id       TEXT PRIMARY KEY,
                conflict_id         TEXT NOT NULL,
                decision            TEXT NOT NULL,
                text                TEXT NOT NULL,
                approver            TEXT,
                authority_level     INTEGER,
                effective_date      TEXT,
                created_at          TEXT NOT NULL,
                resolution_chunk_id TEXT DEFAULT ''
            );
            """
        )
        self._conn.commit()

    # --- conflicts -----------------------------------------------------------
    def exists_open_conflict(self, chunk_a: str, chunk_b: str) -> bool:
        """True if this unordered chunk pair already has a non-dismissed record
        (avoids re-creating the same candidate on every re-ingest)."""
        a, b = _pair_key(chunk_a, chunk_b)
        row = self._conn.execute(
            "SELECT 1 FROM conflicts WHERE pair_key = ? AND status != 'dismissed' LIMIT 1",
            (f"{a}|{b}",),
        ).fetchone()
        return row is not None

    def create_conflict(self, project_id: str, module: str, chunk_a_id: str,
                        chunk_b_id: str, source_a: str, source_b: str,
                        similarity: float, priority: str = "medium",
                        status: str = "suspected") -> ConflictRecord:
        a, b = _pair_key(chunk_a_id, chunk_b_id)
        rec = ConflictRecord(
            conflict_id=uuid.uuid4().hex[:12], project_id=project_id, module=module,
            status=status, priority=priority, created_at=_now(),
            chunk_a_id=chunk_a_id, chunk_b_id=chunk_b_id,
            source_a=source_a, source_b=source_b, similarity=similarity,
        )
        self._conn.execute(
            """INSERT INTO conflicts (conflict_id, project_id, module, status, priority,
                 created_at, chunk_a_id, chunk_b_id, source_a, source_b, similarity,
                 assigned_reviewer, rationale, resolution_id, pair_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec.conflict_id, rec.project_id, rec.module, rec.status, rec.priority,
             rec.created_at, rec.chunk_a_id, rec.chunk_b_id, rec.source_a, rec.source_b,
             rec.similarity, rec.assigned_reviewer, rec.rationale, rec.resolution_id,
             f"{a}|{b}"),
        )
        self._conn.commit()
        return rec

    def get_conflict(self, conflict_id: str) -> ConflictRecord | None:
        row = self._conn.execute(
            "SELECT * FROM conflicts WHERE conflict_id = ?", (conflict_id,)).fetchone()
        return self._to_conflict(row) if row else None

    def get_conflicts(self, status: str | None = None,
                      project_id: str | None = None) -> list[ConflictRecord]:
        sql = "SELECT * FROM conflicts"
        clauses, params = [], []
        if status:
            clauses.append("status = ?"); params.append(status)
        if project_id:
            clauses.append("project_id = ?"); params.append(project_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        return [self._to_conflict(r) for r in self._conn.execute(sql, params).fetchall()]

    def record_verdict(self, conflict_id: str, is_conflict: bool,
                       rationale: str = "") -> bool:
        """Host-LLM adjudication: confirmed (real contradiction) or dismissed."""
        status = "confirmed" if is_conflict else "dismissed"
        cur = self._conn.execute(
            "UPDATE conflicts SET status = ?, rationale = ? WHERE conflict_id = ?",
            (status, rationale, conflict_id))
        self._conn.commit()
        return cur.rowcount > 0

    def open_conflict_chunk_ids(self, project_id: str | None = None) -> set[str]:
        """Chunk ids still entangled in suspected/confirmed conflicts — generation
        should be blocked or flagged on these until resolved."""
        sql = ("SELECT chunk_a_id, chunk_b_id FROM conflicts WHERE status IN "
               "('suspected','confirmed')")
        params: list = []
        if project_id:
            sql += " AND project_id = ?"; params.append(project_id)
        out: set[str] = set()
        for row in self._conn.execute(sql, params).fetchall():
            out.add(row["chunk_a_id"]); out.add(row["chunk_b_id"])
        return out

    # --- resolutions ---------------------------------------------------------
    def create_resolution(self, conflict_id: str, decision: str, text: str,
                          approver: str, authority_level: int,
                          resolution_chunk_id: str = "", effective_date: str = "",
                          resolution_id: str | None = None) -> ResolutionArtifact:
        art = ResolutionArtifact(
            resolution_id=resolution_id or uuid.uuid4().hex[:12], conflict_id=conflict_id,
            decision=decision, text=text, approver=approver,
            authority_level=authority_level,
            effective_date=effective_date or _now(), created_at=_now(),
            resolution_chunk_id=resolution_chunk_id,
        )
        self._conn.execute(
            """INSERT INTO resolutions (resolution_id, conflict_id, decision, text,
                 approver, authority_level, effective_date, created_at, resolution_chunk_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (art.resolution_id, art.conflict_id, art.decision, art.text, art.approver,
             art.authority_level, art.effective_date, art.created_at,
             art.resolution_chunk_id),
        )
        # Mark the conflict resolved and link it to the artifact.
        self._conn.execute(
            "UPDATE conflicts SET status = 'resolved', resolution_id = ? WHERE conflict_id = ?",
            (art.resolution_id, conflict_id))
        self._conn.commit()
        return art

    def get_resolution(self, resolution_id: str) -> ResolutionArtifact | None:
        row = self._conn.execute(
            "SELECT * FROM resolutions WHERE resolution_id = ?", (resolution_id,)).fetchone()
        return self._to_resolution(row) if row else None

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _to_conflict(row) -> ConflictRecord:
        return ConflictRecord(
            conflict_id=row["conflict_id"], project_id=row["project_id"],
            module=row["module"] or "", status=row["status"], priority=row["priority"] or "",
            created_at=row["created_at"], chunk_a_id=row["chunk_a_id"],
            chunk_b_id=row["chunk_b_id"], source_a=row["source_a"] or "",
            source_b=row["source_b"] or "", similarity=row["similarity"] or 0.0,
            assigned_reviewer=row["assigned_reviewer"] or "", rationale=row["rationale"] or "",
            resolution_id=row["resolution_id"] or "",
        )

    @staticmethod
    def _to_resolution(row) -> ResolutionArtifact:
        return ResolutionArtifact(
            resolution_id=row["resolution_id"], conflict_id=row["conflict_id"],
            decision=row["decision"], text=row["text"], approver=row["approver"] or "",
            authority_level=row["authority_level"] or 0,
            effective_date=row["effective_date"] or "", created_at=row["created_at"],
            resolution_chunk_id=row["resolution_chunk_id"] or "",
        )
