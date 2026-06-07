# Phase E — Failure Intelligence (Flow + Design)

**Status:** Implemented (unit + real-ChromaDB integration tests passing).
Local-first, no cloud, no API keys. Turns dead test-failure artifacts into a
searchable memory so recurring failures are diagnosed instantly instead of
re-debugged.

This is the local-first answer to the "store test artifacts in a vector DB"
pattern (the Astra DB / Pinecone QA articles), built on the **same** embedder +
ChromaDB the rest of this platform already uses — no Astra, OpenAI, or LangChain.

---

## The problem

A CI run fails, an Allure/JUnit report is generated, trace files sit in a bucket
until they expire. Those artifacts contain patterns — the same login timeout that
keeps reappearing, the identical DOM mutation that breaks checkout — but a flat
file or a relational table cannot answer *"show me failures that look like this
one."* A vector store can.

## What Phase E adds

| Capability | How |
|---|---|
| **Instant failure diagnosis** | On a new failure, retrieve the most similar past failures + their root cause / fix |
| **Recurring-failure detection** | Identical errors collapse onto one `failure_id`; `occurrences` is the "keeps happening" signal |
| **Triage memory** | After a fix, annotate the failure with root_cause / fix_commit / assignee |
| **Chronic-offender report** | `stats` ranks the most frequently recurring failures |

---

## End-to-end flow

```
        CI / local test run (pytest, Playwright, JUnit, Cypress ...)
                              │  emits JUnit XML
                              ▼
        examples/ingest_failures.py  ingest --junit results.xml --project web
                              │  parse <testcase> with <failure>/<error>
                              ▼
        FailureAnalyzer.record_failure(test_name, error, stack_trace, ...)
                              │
              ┌───────────────┴────────────────┐
       first time?                       seen before?  (same sig hash)
              │                                 │
              ▼                                 ▼
     new Failure chunk                 occurrences += 1, last_seen updated,
     occurrences = 1                   triage annotations preserved
              │                                 │
              └───────────────┬─────────────────┘
                              ▼
              embed error signature  →  ChromaDB (category="Failure")
                              │   (screenshot path/URL in metadata; blob stays in S3/disk)
                              ▼
        ── later: a NEW red build ──────────────────────────────────────
                              │
        FailureAnalyzer.find_similar("checkout button click times out")
                              │  dense cosine search, category=Failure, is_latest
                              ▼
        top-k past failures above threshold, with root_cause + fix_commit
                              │
              ┌───────────────┴────────────────┐
        strong match?                     no match?
              │                                 │
              ▼                                 ▼
   "This is FAIL-x: cookie banner       "No strong match — manual triage."
    overlay; fixed in a1b2c3d"                 │
              │                                 ▼
              ▼                    (after fixing) annotate_failure(id,
   reuse known fix → MTTR drops           root_cause=..., fix_commit=...)
```

The host IDE LLM can drive the query/annotate step: feed it the new error, it
calls `find_similar_failures`, and synthesizes a root-cause hypothesis from the
retrieved history — pattern matching at scale, with the vector step filtering
noise before the LLM ever sees the data.

---

## Key design decisions

- **One vector per failure (no chunking).** A failure is an atomic event;
  fragmenting a stack trace would split the signal across rows and wreck
  similarity. Each failure is a single `Chunk` in category `Failure`. (Contrast
  with documents, which *are* chunked.)
- **The embedding key is the error signature** = `test_name + error_message +
  stack_trace` (trace tail bounded to ~2 KB so one giant trace can't dominate).
  `root_cause` / `fix_commit` / `assignee` are **metadata only** — attached after
  triage, never folded into the vector, so a stale annotation can't drift the
  search key. (Field trap: an outdated embedding is worse than none.)
- **Idempotent + recurrence-aware.** `failure_id = "fail_" + sha256(signature)[:12]`.
  Re-recording an identical failure upserts the same id, bumps `occurrences`, and
  preserves `first_seen` + annotations. That count is the recurrence signal.
- **Blobs live elsewhere.** Only the screenshot path/URL goes in metadata; the PNG
  stays on disk/object storage. The vector is the index, the blob is the payload.
- **Authority 10 (lowest).** Failures are diagnostic, never ground truth. The low
  authority + dedicated category means they never leak into
  `gather_test_context` (which filters to Story/UI_Flow/Error_Code/Business_Rule/
  MOM) or outrank real knowledge in authority-boosted search.
- **Same store, two writers.** CI writes failures via the script; the IDE reads
  them via MCP tools — both point at the same ChromaDB, so CI-ingested failures
  are queryable from the editor with zero sync.

---

## MCP tools (Phase E)

| Tool | Purpose |
|---|---|
| `record_failure` | Store/merge a failure; returns the `FailureArtifact` (incl. `occurrences`) |
| `find_similar_failures` | The analyzer: similar past failures + root cause/fix for a new error |
| `annotate_failure` | Attach root_cause / fix_commit / assignee after triage (metadata-only) |
| `get_failure_stats` | Distinct failures, total occurrences, top recurring offenders |

## CI hook (`examples/ingest_failures.py`)

```bash
# At the end of a CI job (run even on failure):
python examples/ingest_failures.py ingest --junit results.xml --project web --run-id "$CI_BUILD_ID"

# Triage a fresh error:
python examples/ingest_failures.py query --error "TimeoutError clicking #checkout" --project web

# Chronic offenders:
python examples/ingest_failures.py stats --project web
```

Parses JUnit XML with the stdlib (`xml.etree`) — works for pytest (`--junitxml`),
JUnit/Surefire, Playwright/Cypress reporters, Gradle, gotestsum, etc.

## Config

| Key | Default | Meaning |
|---|---|---|
| `failure_similarity_threshold` | `0.85` | Cosine score above which a new error is treated as the same as a past failure |

---

## Component responsibilities (consistent with Phases A–D)

- **Embedding model / Vector DB:** similarity only. No notion of recurrence, triage,
  or authority.
- **FailureAnalyzer (server):** signature hashing, dedup/occurrence counting,
  threshold gating, stats. Deterministic, no LLM.
- **Host IDE LLM:** synthesizes a root-cause hypothesis from retrieved history;
  decides whether a match is genuinely the same bug.
- **Engineer:** annotates the confirmed root cause / fix so the memory stays fresh.
