# Phase A — Versioning Foundation (Design)

**Status:** Implemented (41 tests passing, incl. real-ChromaDB end-to-end versioning test)
**Scope:** Document/chunk versioning (Option B — keep history), enriched metadata, and an `is_latest`-by-default retrieval contract.
**Why first:** Every other phase (delta generation, governance, traceability) reads version metadata. Changing the metadata schema *after* chunks exist forces a re-index, so we lock the schema now.

---

## 1. Goals & Non-Goals

### Goals
- Keep historical chunks instead of deleting them on update (Option B).
- Give every chunk a stable logical identity (`document_id`) that survives content changes.
- Track `version` and `is_latest` per chunk.
- Make `is_latest = true` the **default** for all production retrieval; historical search is opt-in.
- Add the metadata fields that Phase B (delta) and Phase C (governance) will consume, so we never re-index twice.
- Provide a one-time, idempotent migration that stamps existing chunks.

### Non-Goals (deferred)
- Chunk-level diffing / impact analysis → **Phase B** (this doc only *stores* the data Phase B needs).
- Conflict detection, authority scoring, resolution artifacts → **Phase C**.
- Review UI, notifications → **Phase D**.
- Cross-document `module` taxonomy management — `module` is captured but not curated here.

---

## 2. Metadata Schema (locked for Phases A–C)

New fields added to `Chunk` and persisted in ChromaDB metadata. Fields marked **(B)** / **(C)** are populated now but consumed in later phases.

| Field | Type | Source | Purpose |
|---|---|---|---|
| `document_id` | str | derived (§3) | Stable logical identity across versions — the versioning anchor |
| `version` | int | ingestion state machine | Monotonic per `document_id`, starts at 1 |
| `is_latest` | bool | ingestion state machine | Default retrieval filter |
| `module` | str | param / folder / "default" | Coarse grouping (e.g. "Authentication") |
| `section` | str | chunker (nearest heading) | Section the chunk came from; best-effort |
| `chunk_hash` | str | per-chunk sha256 | **(B)** Detect which individual chunks changed between versions |
| `superseded_at` | str (ISO) \| "" | state machine | When this chunk lost `is_latest` (audit) |
| `authority_score` | int | category default (§7) | **(C)** Reserved; defaults by category, used by governance ranking |

Existing fields kept unchanged: `id`, `text`, `source_path`, `document_title`, `chunk_index`, `total_chunks`, `category`, `ingestion_timestamp`, `content_hash`, `embedding_model`, `model_version`, plus the free-form `metadata` dict (issue_key, epic_key, etc.).

**Note on `content_hash` vs `chunk_hash`:**
- `content_hash` = sha256 of the **whole source** (file/Jira body). Drives document-level change detection. *Already exists.*
- `chunk_hash` = sha256 of the **individual chunk text**. New. Lets Phase B tell that section 2 of a doc is unchanged even though section 1 changed (the whole-file `content_hash` changes whenever *any* section changes, so it can't answer that).

---

## 3. Identity & ID Schemes

### `document_id` (stable logical identity)
- **Local files:** POSIX-normalized path **relative to `knowledge_dir`** (forward slashes, original case). Falls back to the normalized absolute path if the file is ingested from outside `knowledge_dir`.
  - e.g. `auth/login_flow.md`
- **Jira:** `jira:{ISSUE_KEY}` (e.g. `jira:TWCJ-6184`).

`document_id` is **content-independent** — the same file path keeps the same `document_id` across every version. This is the anchor the state machine groups on.

### Chunk `id` (must be unique across versions)
Current format `{content_hash[:12]}_{chunk_index}` is **insufficient** for Option B: if v3 happens to revert to v1's exact content, the hash (and thus the IDs) collide with v1's retained (is_latest=false) chunks, and the upsert silently overwrites history.

**New format:** `{content_hash[:12]}_{chunk_index}_v{version}`
- Guarantees uniqueness across versions even on content revert.
- Still stable for a given (content, version) pair → idempotent re-runs remain no-ops.

---

## 4. Ingestion State Machine

Replaces today's "delete old → insert new" in `ingestion.py`. Grouping key is `document_id`.

```
ingest(source):
    content_hash = sha256(source_bytes)
    document_id  = derive_document_id(source)        # §3

    latest = store.latest_version_for_document(document_id)   # → (version, content_hash) | None

    # Case 1: brand new document
    if latest is None:
        version = 1
        emit chunks with version=1, is_latest=true
        return CREATED(version=1)

    # Case 2: unchanged (idempotent re-run) — PRESERVES current behavior
    if latest.content_hash == content_hash:
        return SKIPPED_UNCHANGED(version=latest.version)

    # Case 3: changed → new version (Option B: retain old)
    version = latest.version + 1
    store.mark_not_latest(document_id)               # flip prior is_latest=false, stamp superseded_at
    emit chunks with version=version, is_latest=true
    return UPDATED(version=version)
```

**Key property:** Case 3 never deletes. Old chunks remain queryable for history/delta, but are excluded from default retrieval by the `is_latest` filter.

`IngestionResult` gains `document_id: str` and `version: int` so callers (and the audit log in Phase C) can see exactly what happened.

---

## 5. Store API Additions (`store.py`)

New / changed methods on `ChromaStore`:

| Method | Behavior |
|---|---|
| `latest_version_for_document(document_id)` | Returns `(version, content_hash)` of the `is_latest=true` chunks for that doc, else `None`. |
| `mark_not_latest(document_id)` | Metadata-only update: sets `is_latest=false` and `superseded_at=now` on the doc's current latest chunks. Uses Chroma `.update()` (**no re-embedding**). |
| `search(..., where=...)` | **Changed:** callers pass an `is_latest` filter via the where-merge helper (§6). |
| `keyword_search(..., where=...)` | **Changed:** add metadata `where=` alongside the existing `where_document` substring filter. |
| `delete_by_source` / `remove_source` | **Changed semantics:** purges **all versions** of a `document_id` by default (full removal). Document this clearly. |
| `_meta(chunk)` | Emits the new fields from §2. |

**Migration helper (one-time, idempotent):**
`backfill_versioning()` — scans all chunks lacking `version`; stamps `version=1`, `is_latest=true`, `document_id=` (derived from `source_path`), `chunk_hash=sha256(text)`, `authority_score=` (category default). Metadata-only `.update()`, no re-embed. Safe to run repeatedly (skips already-stamped chunks).

---

## 6. Retrieval Contract Change (`retrieval.py`)

**Rule:** `is_latest = true` is injected by default into every `search()` / `gather_context()` call. Callers opt into history with `include_historical=True`.

Where-clause merge (ChromaDB requires `$and` to combine conditions):

```
def _with_latest(filters, include_historical):
    latest = {} if include_historical else {"is_latest": True}
    if filters and latest:
        return {"$and": [filters, latest]}
    return filters or latest or None
```

- `RetrievalEngine.search(query, filters, top_k, include_historical=False)`
- `gather_context(...)` always uses `include_historical=False`.
- `keyword_search` path gets the same merged `where`.

This is the single change that closes the retrieval-pollution problem (P1.6): with both v1 and v2 in the store, default search returns only v2.

---

## 7. `authority_score` defaults (reserved for Phase C)

Stamped now so the schema is stable; ranking that consumes it lands in Phase C.

| Category | Default `authority_score` |
|---|---|
| `Business_Resolution` *(added in C)* | 100 |
| `Story`, `Business_Rule` | 80 |
| `UI_Flow`, `Error_Code` | 70 |
| `Test_Case` | 60 |
| `MOM`, `Q_and_A` | 40 |

Configurable via `config.yaml` in Phase C; hardcoded defaults for now.

---

## 8. Migration & Rollout

1. Ship schema + state-machine code.
2. Run `backfill_versioning()` once against the existing ChromaDB (metadata-only, no re-embed, no model download).
3. **Only then** enable the `is_latest` default filter — otherwise pre-migration chunks (missing the field) would be filtered out and retrieval would return empty.

Fallback if backfill is ever in doubt: wipe `vectordb-data/` and re-ingest from `KnowledgeBase/` (clean, but re-embeds everything).

---

## 9. Test Plan

Unit (FakeEmbedder, no downloads — matches existing `test_unit.py` style):
- `document_id` derivation: file under/outside `knowledge_dir`, Jira key.
- Chunk ID uniqueness across versions, including the v3-reverts-to-v1 case.
- State machine: new → CREATED v1; same hash → SKIPPED; changed → UPDATED v2 + old chunks flipped `is_latest=false`, not deleted.
- Where-merge: filters-only, latest-only, both (`$and`), historical opt-in.
- `backfill_versioning()` idempotency (run twice → second is a no-op).

Integration (skipped if chromadb absent):
- Ingest v1, ingest v2, assert default search returns only v2; assert `include_historical=True` returns both.

---

## 10. Files Touched

| File | Change |
|---|---|
| `models.py` | Add new fields to `Chunk`; add `document_id`, `version` to `IngestionResult` |
| `store.py` | New version/latest methods, `mark_not_latest`, where-merge support, `_meta` fields, `backfill_versioning` |
| `ingestion.py` | New state machine; `document_id` derivation; per-chunk `chunk_hash`; version-aware chunk IDs |
| `chunker.py` | Emit nearest-heading `section` (best-effort) |
| `retrieval.py` | `is_latest` default filter + `include_historical` opt-in; where-merge helper |
| `server.py` | Thread `include_historical` through `search_knowledge`; surface `version` in status |
| `tests/test_unit.py` | New cases (§9) |
| `docs/phase_a_versioning_design.md` | This document |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Forgetting `is_latest` on a future retrieval path reintroduces pollution | Default-on in the engine; opt-out is explicit. No raw `_col.query` outside `store.py`. |
| Pre-migration chunks filtered out | Enforce migration-before-filter ordering (§8). |
| Storage growth from retained versions | Acceptable locally; add a retention/prune policy when scaling (deferred). |
| `section` extraction too invasive in chunker | Best-effort; defaults to `""` if heading not resolvable — does not block Phase A. |
| Chunk-ID collision on content revert | Version-suffixed IDs (§3). |

---

## 12. Resolved Decisions

1. **`module` source of truth:** **explicit ingest param > top-level folder under `KnowledgeBase/` > `"default"`.** The caller may pass `module=` explicitly; if omitted, derive from the first path segment under `knowledge_dir`; else `"default"`.
2. **`remove_source` semantics:** **purge ALL versions** of a `document_id`. A future `rollback_version` tool can handle "undo last update" separately so the two intents never share one name.
3. **Retention:** add a config knob `max_versions_retained` (0 = unlimited, the local default). Pruning hook wired now; enforced only when set > 0.
