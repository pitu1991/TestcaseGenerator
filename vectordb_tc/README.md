# VectorDB Test Case Generator — Setup & Remaining Tasks

Local MCP server: ingests project knowledge into ChromaDB and assembles grounded
context so the host LLM in Kiro drafts test cases. The server does no generation.

## What's done (drop-in)

All core modules, tested where logic is deterministic:
`models.py`, `config.py`, `embedder.py`, `chunker.py`, `store.py`, `retrieval.py`,
`validator.py`, `exporter.py`, `ingestion.py` (local + xlsx), `server.py` (8 tools),
`jira_client.py` (ADF flattener complete; function mapping marked TODO).
`tests/test_unit.py` — 21 passing unit/property tests, no model download required.

## Setup (on the Windows machine)

1. Put this folder at `<project_root>/vectordb_tc/`.
2. `pip install -r requirements.txt`
   First run downloads `bge-small-en-v1.5` (~130 MB) once; it is then cached for
   offline use. To pre-bundle for fully offline installs, copy the HuggingFace
   cache folder onto the target machine.
3. Edit `config.yaml` → set `project_root`. All other paths derive from it.
4. Add the `vectordb-tc` block from `mcp.json` to your Kiro MCP config (it sits
   alongside your existing `jira-mcp` entry).
5. Run tests: `pytest tests -q` → expect `21 passed`.

## Remaining tasks (need your machine / data / Jira server)

- **Wire the Jira client (Task 5.1):** in `jira_client.py`, map `get_issue`,
  `get_epic_stories`, `get_all_epics_summary` to your existing server's real
  function names (3 lines, marked TODO). Then implement `ingestion.ingest_jira_story`
  / `ingest_jira_epic` to call the client, run results through `adf_to_text`
  (already done inside the client's `_normalize_issue`), chunk, and store. Add the
  best-effort vs `fail_fast` reporting.
- **Implement `ingestion.sync()`:** hash-diff the filesystem against the store,
  ingest new/changed, and delete chunks for removed sources (logic for both halves
  already exists — `hash_for_source` and `delete_by_source`).
- **Initial ingestion (Task 10.3):** once Jira is wired, run `ingest_documents`
  on `<project_root>/KnowledgeBase/`, `ingest_documents` on your existing test-case
  `.xlsx` files (parsed as `Test_Case`), and `ingest_jira_stories` for the 12 epics.
  Then `get_ingestion_status` should show non-zero `Story` and `Test_Case` counts.
- **Integration tests:** add a `tests/test_integration.py` that stands up a real
  `ChromaStore` in a temp dir with the real embedder, ingests a couple of fixture
  files, and asserts retrieval + a full `gather_test_context` → `export_test_cases`
  round-trip. Mark it to skip when chromadb/sentence-transformers aren't installed.

## Two things to tune against real data

- The identifier regex in `retrieval._extract_keywords` and the account-number
  pattern in `ingestion.redact_pii` are deliberately coarse. Adjust them once you
  see your actual error-code / field-name formats — the bank pattern in particular
  will over-match long digit runs.
- `_auto_category` guesses category from filename. If your KnowledgeBase naming
  doesn't follow those hints, pass an explicit `category` to `ingest_documents`.
