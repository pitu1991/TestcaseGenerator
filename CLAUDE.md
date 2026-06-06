# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VectorDB Test Case Generator** — A local MCP server that ingests project knowledge (Jira stories, CORE UI flows, MOMs, error codes, business rules, Q&A, existing test cases) into ChromaDB and assembles grounded context so the host LLM in Kiro can draft structured test cases.

**Key architectural principle:** This is a retrieval-and-formatting engine only—no generation, no runtime LLM, no network access (embedding model downloads are cached locally). The host LLM in Kiro handles test case generation.

## Build & Install

### Prerequisites

- Python 3.10+
- Windows (tested), but architecture is cross-platform
- pip (package manager)

### Installation

```bash
# Install dependencies
pip install -r vectordb_tc/requirements.txt

# First run downloads the embedding model (~130 MB); cached for offline use
# To pre-bundle for fully offline installs, copy the HuggingFace cache folder
```

### Configuration

1. Copy vectordb_tc/config.example.yaml to vectordb_tc/config.yaml
2. Set project_root to your actual project directory
3. All other paths derive from project_root:
   - vectordb_tc/chromadb_path → {project_root}/vectordb-data
   - vectordb_tc/output_dir → {project_root}/User Stories
   - vectordb_tc/knowledge_dir → {project_root}/KnowledgeBase
   - vectordb_tc/jira_server_path → {project_root}/jira-mcp/server.py

**Important:** `server.py` line 20 defaults to `AppConfig()`, which reads from the `VECTORDB_PROJECT_ROOT` environment variable — NOT from `config.yaml`. If you edit `config.yaml`, change that line to `AppConfig.from_file("config.yaml")` or set the env var, otherwise your YAML settings are silently ignored.

## Running Tests

```bash
# Run all unit + property tests (21 tests, no external downloads needed)
# These use a FakeEmbedder and do NOT require chromadb/sentence-transformers at import
pytest vectordb_tc/tests -q

# Run a single test file
pytest vectordb_tc/tests/test_unit.py -q

# Run tests with verbose output
pytest vectordb_tc/tests -v

# Run tests with coverage
pytest vectordb_tc/tests --cov=vectordb_tc
```

Test structure:
- test_unit.py: 21 unit/property tests covering chunker, retrieval (RRF fusion), validation, PII redaction, ADF flattening, and config (no model download required)
- test_integration.py: Integration tests for real ChromaDB + embedder; marked to skip if dependencies absent

Config: pytest.ini sets python_classes = *Tests for test discovery.

## Running the Server

```bash
# Start the FastMCP server (listens for tool calls from Kiro)
python vectordb_tc/server.py
```

The server exposes 8 MCP tools.

### Kiro MCP Wiring

Register both servers in Kiro's MCP config (vectordb_tc/mcp.json is a reference template):

```json
{
  "mcpServers": {
    "jira-mcp": {
      "command": "python",
      "args": ["/absolute/path/to/jira-mcp/server.py"]
    },
    "vectordb-tc": {
      "command": "python",
      "args": ["/absolute/path/to/vectordb_tc/server.py"],
      "env": { "VECTORDB_PROJECT_ROOT": "/absolute/path/to/project_root" }
    }
  }
}
```

The `VECTORDB_PROJECT_ROOT` env var is the primary config mechanism when running via Kiro (since `AppConfig()` reads it before any YAML file).

**Jira MCP is not called over the wire.** `JiraClient` loads `jira-mcp/server.py` directly via `importlib` and calls its Python functions. The Jira server's `server.py` must export callable functions (not just MCP handlers). The `if __name__ == "__main__": mcp.run()` guard prevents the MCP loop from firing on import.

## Code Architecture

### Core Data Flow

Input Files / Jira → Ingestion Pipeline → Chunks (embedded) → ChromaDB
                                                                  |
                                           Retrieval Engine ← Query
                                                  |
                                          Context Bundle
                                                  |
                                           Host LLM (Kiro)
                                                  |
                                           Test Cases → Validator → Excel

### Key Modules

#### models.py — Data Contracts
Defines all data classes shared across the system:
- Chunk: persisted unit of knowledge with embeddings, content hash, category, metadata
- SearchResult: chunk + relevance score + match type (dense/keyword/hybrid)
- ContextBundle: structured payload to host LLM (ACs, few-shot examples, UI flows, error codes, etc.)
- TestCase / TestStep: output test case structure
- ValidationResult / ValidationError: export validation report
- IngestionResult: result of a single file/epic ingest
- Category: enum of knowledge types (MOM, UI_Flow, Error_Code, Business_Rule, Q_and_A, Story, Test_Case)

#### config.py — Configuration Management
- AppConfig: dataclass with YAML/JSON loader; all paths derive from project_root
- Defaults: embedding_model=BAAI/bge-small-en-v1.5, chunk_size=400, token_budget=35000, pii_guard=True
- Lazy import of pyyaml (JSON-only setups don't require it)

#### embedder.py — Sentence-Transformers Wrapper
- EmbeddingService: wraps a SentenceTransformer model
- Exposes max_seq_length (e.g., 512 for bge-small-en-v1.5) — used by chunker to clamp chunks
- Handles BGE query prefix ("Represent this sentence...") for search queries
- Token counter via tokenizer (enforces chunk budget throughout pipeline)

#### chunker.py — Structure-Aware Document Chunking
- Chunker: couples to embedder's max_seq_length; clamps chunk_size if it exceeds model limit
- Structural splitting: respects headings, code fences, tables (no silent truncation)
- Table handling: keeps tables whole if they fit; else splits by row with header repeated
- Token splitting fallback: word-window chunking for long prose (0.75 words/token heuristic)
- Chunk ID format: {content_hash[:12]}_{chunk_index} (stable across re-runs if content unchanged)

#### ingestion.py — Document & Jira Ingestion Pipeline
- IngestionPipeline: coordinates chunking, embedding, and store operations
- Local files: `SUPPORTED_TEXT = {".md", ".txt", ".json"}` — **xlsx is NOT general knowledge**; Excel files are only parsed as test cases via `ingest_test_cases()`. Dropping an .xlsx into KnowledgeBase will be silently skipped by `ingest_directory()`.
- Change detection: SHA256 content hash; re-ingest only if hash differs (not mtime)
- Jira integration: fetches stories/epics, normalizes Jira issue dicts, flattens ADF → text
- PII guard (default ON): redacts SSN and bank account patterns; logging redaction count but not values
- Model-change detection: signals if embedding model differs from stored chunks (requires full re-index)
- Test case parsing: .xlsx sheets → Test_Case chunks linked to issue keys
- Best-effort epic ingestion: collects per-story failures, reports them, continues unless fail_fast=True

#### store.py — ChromaDB Persistence
- ChromaStore: wraps ChromaDB PersistentClient with cosine similarity
- Dense search: via embedding vectors
- Keyword search: literal substring match (where_document=\) — reliable for identifiers
- Metadata storage: source_path, document_title, category, content_hash, embedding_model, model_version
- Query helpers: hash_for_source, stored_model_versions, issue_keys_by_category, get_all_sources, get_stats

#### retrieval.py — Hybrid Search & Context Assembly
- RetrievalEngine: combines dense + keyword retrieval via Reciprocal Rank Fusion (RRF)
- Hybrid scoring: RRF formula 1 / (k + rank + 1) per list (k=60); chunks in both lists score highest
- Keyword extraction: domain identifiers (error codes, issue keys, CamelCase field names)
- gather_context(): assembles a ContextBundle within token budget
- AC enumeration: splits AC input on newlines/bullets, assigns stable IDs (AC-1, AC-2, etc.)
- Token budgeting: caps each context group so total stays within limit

#### validator.py — Export Validation
- ExportValidator: HARD + SOFT checks before Excel export
- HARD checks (reject & return errors):
  - Every AC covered by at least one test case
  - At least one negative/error case exists
  - Test case names match {ISSUE_KEY}_TC-{NN} : {description}
  - Required fields non-empty (name, description, precondition, requirement_id, steps)
- SOFT checks (warnings, don't reject):
  - Citation IDs resolve to session context (best-effort grounding)
  - AC coverage only by cases with unresolved citations flagged as "low confidence"
- Produces coverage matrix (AC ID → test case IDs)

#### exporter.py — Excel Output
- ExcelExporter: writes team-standard template with 13 columns
- One sheet per story (issue key); test steps are rows
- Column widths configured (Name=70, Description=40, etc.)
- Defaults: Status="Not Run", Type="Manual", Application="CORE"

#### jira_client.py — Jira Integration
- JiraClient: imports the existing Jira MCP server module directly
- ADF → text flattener (adf_to_text): handles nested structures (lists, tables, code blocks, mentions, panels)
- Preserves tables as | cell | cell | rows (identifier–meaning pairs stay together)
- Headings become markdown (###), lists become bullets/numbered
- Issue normalization: maps raw Jira issue dicts to ingestion-friendly format
- TODO mappings: three function names need mapping to your Jira server (marked in code)

#### server.py — FastMCP Entry Point
- Wires 8 tools and returns STRUCTURED errors (empty KB, Jira unavailable, validation failures)
- Globals: CONFIG, EMBEDDER, STORE, CHUNKER, INGEST, RETRIEVE, EXPORTER, VALIDATOR
- Tools: ingest_documents, ingest_jira_stories, search_knowledge, gather_test_context, export_test_cases, get_ingestion_status, list_stories_without_tests, remove_source
- **Two-tool workflow:** `gather_test_context` appends an `instruction_block` and `template_spec` to its JSON response telling the host LLM to produce test cases and then call `export_test_cases`. This is how the server directs generation without doing it itself — the instruction is baked into the tool response, not into a system prompt.

### Coupling & Design Decisions

1. **Embedder coupling:** Chunker is coupled to embedder's max_seq_length; no silent truncation
2. **Content hash (not mtime):** Change detection uses SHA256; re-running on unchanged files is a no-op
3. **Model-change detection:** Chunk metadata stores embedding model name/version; mismatch signals full re-index
4. **Token counting throughout:** EmbeddingService.count_tokens() called by chunker and retrieval
5. **RRF fusion:** Dense + keyword results fused by score; hybrid scoring makes identifiers retrievable
6. **Best-effort epic ingest:** Failures collected and returned; use fail_fast=True to abort on first error
7. **PII redaction (default ON):** Lightweight pattern guard for SSN + bank account numbers

## Key Design Patterns

### Change Detection & Idempotency
- Content hash: Every source (file or Jira story) is hashed and stored in chunk metadata
- Skip unchanged: If hash matches stored hash, ingest is skipped (fast re-runs)
- Replace on change: If hash differs, old chunks deleted and new ones stored (no orphans)

### Token Budgeting
- Retrieval engine caps context per category: few-shot gets 1/3 of budget, error codes get 1/5, etc.
- Chunker respects embedder's max_seq_length; no chunk exceeds model's capacity
- Server.py's gather_context() returns total_tokens so host LLM knows budget used

### Hybrid Retrieval via RRF
- Dense search: semantic similarity via embeddings (catch paraphrases, related concepts)
- Keyword search: literal substring match (reliable for identifiers, error codes, issue keys)
- RRF fusion: score = sum of 1 / (k + rank_in_list + 1) across lists
- Reduces need for reranking in Phase 1

### Structured Error Handling
- All errors returned as JSON: {"error": "<kind>", "message": "...", **extra}
- Kinds: model_changed, jira_unavailable, not_implemented, empty_knowledge_base, etc.
- Host LLM can programmatically handle specific errors

## Common Development Tasks

### Add a new knowledge category
1. Add to Category enum in models.py (CATEGORIES tuple must also be updated)
2. Update _auto_category() in ingestion.py if you want filename hints
3. Update gather_context() in retrieval.py to include the category
4. Update requirements/design doc if user-facing

### Extend ADF → text flattening
- Edit adf_to_text() in jira_client.py
- Add cases for new node types (e.g., ntype == "newType")
- Keep the fallback at the end to avoid silent drops

### Tune retrieval for your data
- Keyword extraction: Adjust _IDENT regex in retrieval.py (error codes, issue keys, CamelCase fields)
- PII redaction: Update _SSN and _BANK patterns in ingestion.py if account format differs
- Auto-categorization: Update filename patterns in _auto_category()

### Implement Jira client mapping (required setup task)
Three TODO lines in jira_client.py need your Jira server function names:
- self._mod.get_issue(issue_key)
- self._mod.get_epic_issues(epic_key) (fetch stories under epic)
- self._mod.get_all_epics() (for batch operations)

## Notes on Phase 1 vs Phase 2

**Phase 1 (current scope):**
- Local-only, no network access (embedding model cached)
- Lightweight PII guard (patterns, not full framework)
- Retrieval + formatting only; generation is host LLM's job
- Citation verification is soft (warning, not rejection)
- Reranking optional, default off
- Text-based attachments only, no OCR
- Manual quality evaluation via spot-checks

**Phase 2 (deferred):**
- OCR for image attachments
- Full configurable redaction framework
- Evaluation framework (golden sets, recall@K, MRR metrics)
- Potential shared/hosted deployment with auth

## Debugging Tips

- **Check embedding model version mismatch:** get_ingestion_status() shows model_version; mismatch triggers model_changed error
- **Keyword retrieval issues:** Ensure identifiers match _IDENT regex in retrieval.py; adjust pattern if needed
- **Token budget exceeded:** Check gather_context() output's total_tokens vs token_budget; reduce chunk_size or increase budget
- **PII redactions:** Enable log_level: "DEBUG" in config; check log file (no actual values logged)
- **Jira unavailable:** Verify Jira MCP server path in config.yaml; ensure jira-mcp/server.py exports mapped functions
