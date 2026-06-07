# Phases B / C / D — Delta, Governance, Review UI (Design + Status)

**Status:** Implemented (57 tests passing, incl. real-ChromaDB + SQLite end-to-end).
Builds on [Phase A](phase_a_versioning_design.md). Local-first, no cloud, no
in-server LLM, no API keys — all reasoning happens in the host IDE LLM via MCP tools.

---

## Phase B — Delta Generation

**Goal:** when a document changes, regenerate only the test cases impacted by the
changed sections instead of re-running the whole document.

**How it works**
- Phase A stamped a per-chunk `chunk_hash`. The whole-file `content_hash` tells you
  *that* a document changed; `chunk_hash` grouped by `section` tells you *which*
  parts changed.
- `delta.py::DeltaEngine.diff_versions(document_id, from_v?, to_v?)` defaults to
  *previous → latest*, groups each version's chunks by `section`, and emits a
  `DocumentDelta` of **added / changed / removed** sections (combined-hash compare,
  ordered by `chunk_index` so a reorder counts as a change). Single-version docs
  report everything as `added` (from_version = 0).
- `RetrievalEngine.relevant_test_cases(query, issue_key)` fetches existing
  `Test_Case` chunks most relevant to the changed text (latest-only).

**MCP tools:** `get_document_versions`, `search_delta_changes`,
`gather_delta_context` (returns changed sections + existing test cases + an
`instruction_block` telling the LLM to regenerate only impacted cases, keep the
rest, then call `export_test_cases` with the merged set so validation still holds).

**`_with_latest` change:** now flattens an existing `$and` instead of nesting
(ChromaDB rejects nested `$and`) — needed so `category + issue_key + is_latest`
combine in one filter.

---

## Phase C — Knowledge Governance & Conflict Resolution

**Goal:** detect when independent documents contradict each other, route them to a
human, and make approved resolutions authoritative.

**Two-phase detection (no in-server LLM):**
1. **Candidate generation (server, at ingest, deterministic):** `conflict.py`
   re-uses each freshly stored chunk's embedding to find similar **cross-document**
   latest chunks above `conflict_similarity_threshold` (default 0.83) and records a
   **`suspected`** conflict. Resolutions are never scanned. Gated by
   `conflict_detection` (default **off** — it costs one similarity query per chunk).
2. **Adjudication (host IDE LLM, on demand):** `get_conflict_candidates` returns both
   chunk texts; the LLM decides contradiction and calls `record_conflict_verdict`
   → `confirmed` / `dismissed`. *The server never decides truth.*

**Governance store (`governance.py`):** SQLite (stdlib, single local file at
`{project_root}/governance.db`, portable to Postgres later). Tables: `conflicts`
(status: suspected → confirmed/dismissed → resolved; unordered `pair_key` dedup)
and `resolutions`.

**Resolution → authoritative knowledge:** `resolve_conflict` re-ingests the approved
text as a **`Business_Resolution`** chunk (authority 100) via
`IngestionPipeline.ingest_resolution`, links it to the conflict, and marks it
resolved.

**Authority ranking:** `RetrievalEngine.search(..., authority_boost=True)` (used by
`search_authoritative_knowledge`) multiplies the fused RRF score by
`(1 + authority_score/100)` and re-sorts, so a Business_Resolution (100) outranks a
Story (80) outranks an MOM (40) at equal relevance. Off by default elsewhere to keep
Phase A/B retrieval behavior unchanged.

**Traceability chain:** TestCase.`resolution_ids` → ResolutionArtifact.`conflict_id`
→ ConflictRecord.`chunk_a/b_id` (+ `resolution_chunk_id` → the re-ingested chunk's
`document_id`). `search_unresolved_conflicts` lists chunks still entangled in open
conflicts so generation can avoid grounding on them.

**MCP tools:** `get_conflicts`, `get_conflict_candidates`, `record_conflict_verdict`,
`resolve_conflict`, `search_authoritative_knowledge`, `search_unresolved_conflicts`.

---

## Phase D — Human-in-the-Loop Review UI + Notifications

**Review UI (`review_app.py`):** stdlib `http.server` only (no FastAPI/uvicorn),
**localhost-bound, single-user, no auth** (deferred to the hosted phase). All logic
lives in `ReviewService` (pure, unit-tested without a socket); the HTTP handler is a
thin adapter. Pages: conflict list, conflict detail (Source A vs B side-by-side with
confirm/dismiss and an approve-resolution form). All output HTML-escaped.
`resolve_fn` / `verdict_fn` are injected, decoupling the UI from ingestion/governance.
Launched on a daemon thread via the `start_review_ui` MCP tool (shares the live
store/governance objects).

**Notifications (`notifier.py`):** `Notifier` interface + `LogNotifier` (default,
logs a line) + `NullNotifier`. Wired into `ConflictDetector` so a flagged conflict
fires `conflict_created`. Swap in an Email/Slack subclass later with no call-site
change — the "wire the placeholder now" decision.

---

## Config added (Phases B–D)

| Key | Default | Meaning |
|---|---|---|
| `conflict_detection` | `false` | Enable similarity-based candidate detection at ingest |
| `conflict_similarity_threshold` | `0.83` | Cosine score to flag a candidate |
| `project_id` | `"default"` | Multi-project scoping for conflicts |
| `governance_db_path` | `{project_root}/governance.db` | SQLite location (derived) |

## Component responsibilities (reaffirmed)

- **Embedding model / Vector DB:** similarity only. No notion of version, authority,
  or contradiction.
- **Retriever + governance (server):** all business logic — latest filtering, delta
  diffing, conflict candidate generation, authority ranking. Deterministic, no LLM.
- **Host IDE LLM:** adjudicates conflicts, performs impact analysis, generates test
  cases. Never runs vector search; never decides which source is authoritative.
- **Human reviewer:** approves the authoritative resolution.
