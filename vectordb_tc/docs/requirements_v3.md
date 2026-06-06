# Requirements Document (v3 — Phase 1 scoped)

## Introduction

A local VectorDB-powered test case generation system for the Transamerica Payroll Modernization QA project. ChromaDB indexes accumulated project knowledge (Jira stories, CORE UI flows, meeting MOMs, error codes, batch job logic, Q&A pairs, and the team's existing test cases). MCP tools retrieve relevant context and assemble grounded prompts so the **host LLM in Kiro** drafts structured test cases for a new user story.

**Architecture decision:** Generation is performed by the host LLM in Kiro, not by the server. The VectorDB_Server is a retrieval-and-formatting engine only — no generation, no runtime LLM, no network access (beyond a one-time, cacheable embedding-model download).

## Phase 1 vs Phase 2 scope

Phase 1 targets a working single-developer build (~2–4 days). The following are intentionally deferred or reduced for Phase 1; the design accommodates them but does not implement them fully:

- **Req 11 (Evaluation framework):** Phase 2. Phase 1 uses a manual spot-check against the team's existing stories/test cases (see Req 11.1).
- **Req 12 (PII):** Phase 1 = documented "no production data" policy + a lightweight default-on pattern guard. Full configurable redaction framework is Phase 2.
- **OCR for image attachments (Req 3):** Phase 2. Phase 1 ingests text-based attachments only.
- **Reranking (Req 4):** Optional, default off in Phase 1. Enable later if retrieval precision is insufficient.

Changes from v2: added `list_stories_without_tests` tool; added cold-start, table-chunking, and Jira-unavailable handling; set default `token_budget = 35000`; citation verification downgraded from hard-fail to a warning (Req 6/7); PII reduced to a Phase-1 guard (Req 12).

## Glossary

(Unchanged from v2.) Key terms: **VectorDB_Server** (retrieval/format engine, no generation), **Host_LLM** (Kiro's LLM, does generation), **Context_Bundle** (structured payload to the Host_LLM), **Few_Shot_Example** (historical story→test-cases pair), **Coverage_Matrix** (AC ID → test case IDs), **Test_Case_Template** (Name, Description, Precondition, Test Step #, Test Step Description, Expected, Assigned To, Requirement Id, Status, Type, Workplace Capability, Priority, Application), **ADF** (Atlassian Document Format), **ECR** (Error Corrections Required page in CORE).

---

## Requirements

### Requirement 1: Document Ingestion from Local Files

**User Story:** As a QA engineer, I want to ingest project knowledge files into the vector database, so that the system has context for generating relevant test cases.

#### Acceptance Criteria

1. WHEN a directory path is provided, THE Ingestion_Pipeline SHALL recursively scan and ingest all .md, .txt, and .json files.
2. WHEN a single file path is provided, THE Ingestion_Pipeline SHALL ingest that file.
3. THE Ingestion_Pipeline SHALL split documents using structure-aware chunking that respects headings, lists, and table boundaries, then cap chunk size so no chunk, tokenized by the Embedding_Model's tokenizer, exceeds the model's max sequence length. Default target: 350–400 tokens per chunk, 80-token overlap.
4. WHEN chunking a table, THE Ingestion_Pipeline SHALL keep the table whole if it fits within the chunk token budget; IF it does not fit, THEN it SHALL split the table by rows and repeat the header row in each sub-chunk so that row-to-meaning mappings are preserved.
5. IF a configured chunk size exceeds the model's max sequence length, THEN THE Ingestion_Pipeline SHALL log a warning and reduce the effective chunk size so no content is silently truncated at embedding time.
6. THE Ingestion_Pipeline SHALL store metadata per chunk: source file path, document title, chunk index, ingestion timestamp, source content hash, embedding model name and version, and category (MOM, UI_Flow, Error_Code, Business_Rule, Q_and_A, Story, Test_Case).
7. WHEN a file already in ChromaDB is re-ingested, THE Ingestion_Pipeline SHALL replace its chunks (keyed by source path), not duplicate them.
8. THE Ingestion_Pipeline SHALL generate embeddings with a local sentence-transformers model, no external API calls at runtime.
9. IF a file cannot be read or parsed, THEN THE Ingestion_Pipeline SHALL log the path and reason and continue with remaining files.

### Requirement 2: Historical Test Case Ingestion

**User Story:** As a QA engineer, I want the team's existing test cases ingested and linked to their source stories, so they serve as few-shot examples matching our style.

#### Acceptance Criteria

1. WHEN a test-case spreadsheet (.xlsx) is provided, THE Ingestion_Pipeline SHALL parse each sheet and extract test cases in the Test_Case_Template structure.
2. THE Ingestion_Pipeline SHALL link each test case set to its source story via the issue key (from sheet name or the `{ISSUE_KEY}_TC-{NN}` naming convention) and store that linkage in metadata.
3. THE Ingestion_Pipeline SHALL store each (story → test cases) pair as a retrievable unit categorized as Test_Case, preserving step structure and expected results.
4. IF a test case cannot be linked to an issue key, THEN THE Ingestion_Pipeline SHALL still ingest it, tag it unlinked, and log the affected rows.
5. WHEN a spreadsheet is re-ingested, THE Ingestion_Pipeline SHALL replace its prior chunks, not duplicate them.

### Requirement 3: Jira Story Ingestion via MCP Integration

**User Story:** As a QA engineer, I want to ingest Jira stories from the existing Jira MCP server, so story details are searchable alongside other knowledge.

#### Acceptance Criteria

1. WHEN an epic key is provided, THE Ingestion_Pipeline SHALL fetch all stories under that epic via the Jira_MCP_Server and ingest description, acceptance criteria, definition of done, and comments.
2. THE Ingestion_Pipeline SHALL parse Jira ADF into clean text (preserving tables, lists, code blocks) before chunking.
3. WHEN a single issue key is provided, THE Ingestion_Pipeline SHALL ingest description, acceptance criteria, definition of done, and comments as separately indexed chunks.
4. THE Ingestion_Pipeline SHALL store Jira metadata per chunk: issue key, epic key, summary, assignee, status, story points, last-updated timestamp.
5. BY DEFAULT (best-effort), IF some stories fail to fetch during epic ingestion, THEN THE Ingestion_Pipeline SHALL ingest what it can and report failures with reasons. A `fail_fast` flag SHALL instead abort on first failure.
6. THE Ingestion_Pipeline SHALL handle Jira pagination and rate limits when fetching all stories under an epic.
7. IF the Jira_MCP_Server is unavailable or not running, THEN `ingest_jira_stories` SHALL return a clear, structured error (server unreachable) and SHALL NOT crash the VectorDB_Server.
8. WHEN a story already in ChromaDB is re-ingested, THE Ingestion_Pipeline SHALL update its chunks.
9. THE Ingestion_Pipeline SHALL support batch ingestion of all 12 project epics in one operation.
10. THE Ingestion_Pipeline SHALL ingest text-based attachments only. (Phase 2: optional OCR for image attachments; default remains text-only.)

### Requirement 4: Semantic Search and Context Retrieval

**User Story:** As a QA engineer, I want to search with natural language and exact identifiers, so I can find context without knowing file locations.

#### Acceptance Criteria

1. WHEN a query is provided, THE Retrieval_Engine SHALL perform hybrid retrieval (dense embedding similarity + keyword/exact-term matching) so identifiers like error codes (e.g., PY075), ECR references, and issue keys (e.g., TWCJ-6184) are reliably matched.
2. THE Retrieval_Engine SHALL support an optional reranking step over top candidates, configurable, default OFF in Phase 1.
3. THE Retrieval_Engine SHALL support filtering by category (MOM, UI_Flow, Error_Code, Business_Rule, Q_and_A, Story, Test_Case).
4. THE Retrieval_Engine SHALL support filtering by epic key, issue key, date range, and file-path pattern.
5. WHEN results are returned, each SHALL include text content, relevance score, source path, and category. Optional metadata MAY be null and SHALL NOT cause a result to be dropped.
6. THE Retrieval_Engine SHALL reject only results missing text content, relevance score, or source path.
7. THE Retrieval_Engine SHALL return results within 3 seconds for up to 50,000 chunks.
8. THE Retrieval_Engine SHALL allow configuring K (default 10) and SHALL cap returned text to a configurable token budget to protect the host context window.

### Requirement 5: Context Assembly for Test Case Generation

**User Story:** As a QA engineer, I want a tool that gathers everything the host LLM needs to draft test cases, so generation is grounded in real project knowledge.

#### Acceptance Criteria

1. WHEN a description and acceptance criteria are provided, THE Context_Assembler SHALL return a Context_Bundle with: similar past stories and their test cases (Few_Shot_Examples), applicable UI flows, related error codes, and relevant business rules.
2. THE Context_Assembler SHALL enumerate the acceptance criteria, assign each a stable AC ID, and include that list in the bundle for the Host_LLM to map cases against.
3. THE Context_Bundle SHALL include the Test_Case_Template column spec, default metadata values, the naming convention, and an instruction block describing required output structure (positive, negative, edge coverage; at least one negative case referencing applicable error/validation codes).
4. THE Context_Assembler SHALL keep the bundle within a configurable token budget (**default 35,000 tokens**), prioritizing highest-relevance context and closest Few_Shot_Examples.
5. IF an issue key is provided, THEN THE Context_Assembler SHALL also fetch linked issues and comments from the Jira_MCP_Server.
6. IF the knowledge base is empty (no chunks ingested), THEN THE Context_Assembler SHALL return a clear "knowledge base empty — run ingestion first" message rather than an empty bundle.
7. IF the knowledge base cannot be accessed or retrieval fails, THEN THE Context_Assembler SHALL fail and report the access error rather than return a partial bundle silently.
8. THE Context_Bundle SHALL tag every context item with a stable source chunk ID so generated cases can cite them and the Export_Validator can verify those citations.

### Requirement 6: Test Case Export and Validation

**User Story:** As a QA engineer, I want host-produced test cases validated and exported to Excel in the team format, so I can upload/share them and trust their coverage.

#### Acceptance Criteria

1. WHEN host-produced structured test cases are submitted, THE Export_Validator SHALL validate them before writing output and return a structured report.
2. THE Export_Validator SHALL **reject and return correctable errors** (so the Host_LLM can retry) IF any HARD check fails: every acceptance criterion is covered by at least one test case; at least one negative test case exists; all names match `{ISSUE_KEY}_TC-{NN} : {Short description}`; and all required template columns are present.
3. THE Export_Validator SHALL verify each cited source chunk ID against this session's assembled context. Unresolvable citations are a **WARNING, not a rejection**: the case still exports and is flagged "unverified citation" in the report. *(Phase 1 decision — soft-fail. To make citations a hard requirement, move this check into Req 6.2.)*
4. THE Export_Validator SHALL produce an .xlsx file with one sheet per user story.
5. THE Export_Validator SHALL use headers: Name, Description, Precondition, Test Step #, Test Step Description, Expected, Assigned To, Requirement Id, Status, Type, Workplace Capability, Priority, Application.
6. THE Export_Validator SHALL set defaults: Status = "Not Run", Type = "Manual", Application = "CORE".
7. THE Export_Validator SHALL set column widths: Name=70, Description=40, Precondition=40, Step#=8, Step Description=55, Expected=50.
8. WHEN multiple stories are submitted, THE Export_Validator SHALL create one sheet per story in one workbook.
9. THE Export_Validator SHALL save output to a configurable directory (default `<project_root>/User Stories/`).
10. THE Export_Validator SHALL include the Coverage_Matrix (AC ID → test case IDs) in its report and mark exported cases as drafts requiring SME review.

### Requirement 7: Traceability and Grounding

**User Story:** As a QA engineer, I want every test case traceable to an AC and grounded in real knowledge, so I can prove coverage during review.

#### Acceptance Criteria

1. THE system SHALL ensure each acceptance criterion maps to at least one positive test case and, where the AC implies failure/validation behavior, at least one negative test case. (Enforced as a HARD check in Req 6.2.)
2. THE system SHALL populate the Requirement Id column with the AC ID(s) each test case covers.
3. THE system SHALL request a cited source chunk ID for each test case; the Export_Validator SHALL verify citations and flag (not reject) any that do not resolve. *(Best-effort grounding — see Req 6.3.)*
4. THE system SHALL draw error codes and validation rules in negative-case expected results from cited sources rather than invented values where such sources exist.
5. WHEN no relevant Few_Shot_Examples or grounding context exist for an AC, THE system SHALL flag that AC as low-confidence in the Coverage_Matrix.

### Requirement 8: MCP Server Integration

**User Story:** As a QA engineer, I want the system exposed as MCP tools, so I can use it from Kiro alongside the Jira MCP server.

#### Acceptance Criteria

1. THE VectorDB_Server SHALL expose `ingest_documents` (file/dir path → ingest, including test-case spreadsheets).
2. THE VectorDB_Server SHALL expose `ingest_jira_stories` (epic key or issue key → ingest Jira content).
3. THE VectorDB_Server SHALL expose `search_knowledge` (query + filters → ranked chunks with metadata).
4. THE VectorDB_Server SHALL expose `gather_test_context` (description + AC + optional issue key → Context_Bundle).
5. THE VectorDB_Server SHALL expose `export_test_cases` (host-produced cases + story key(s) → validation report + Excel path).
6. THE VectorDB_Server SHALL expose `get_ingestion_status` (→ totals, by-category counts, last ingestion, embedding model/version).
7. THE VectorDB_Server SHALL expose `list_stories_without_tests` (→ ingested Story issue keys that have no linked Test_Case set, for sprint planning).
8. THE VectorDB_Server SHALL expose `remove_source` (source path or issue key → delete its chunks).
9. THE VectorDB_Server SHALL run as a standalone FastMCP server compatible with the existing Kiro MCP configuration format.
10. THE VectorDB_Server SHALL persist all data locally so knowledge survives restarts without re-ingestion.
11. THE VectorDB_Server SHALL return structured tool errors (error type + message), including an empty-knowledge-base condition, so the Host_LLM can act on them.

### Requirement 9: Incremental Knowledge Updates

**User Story:** As a QA engineer, I want to add and update knowledge incrementally, so the system stays current without full re-indexing.

#### Acceptance Criteria

1. WHEN new files are added, THE Ingestion_Pipeline SHALL ingest only the new files.
2. WHEN an existing file changes, THE Ingestion_Pipeline SHALL detect it via stored content hash (not mtime alone) and re-ingest only that file.
3. THE Ingestion_Pipeline SHALL support a `sync` mode comparing current file system and Jira state against ChromaDB and ingesting only new/modified items.
4. WHEN a source file is deleted or a story is removed from its epic, THE `sync` mode SHALL delete the orphaned chunks.
5. WHEN a Jira story is updated, THE Ingestion_Pipeline SHALL update its chunks.
6. THE VectorDB_Server SHALL maintain an ingestion log: source id, timestamp, chunk count, content hash, operation type (new, update, delete).

### Requirement 10: System Configuration and Scalability

**User Story:** As a QA engineer, I want the system configurable and ready for future team-wide deployment.

#### Acceptance Criteria

1. THE VectorDB_Server SHALL read YAML/JSON config specifying: ChromaDB path, embedding model, chunk size, overlap, default K, retrieval token budget, reranking on/off, Jira attachment policy, and output directory.
2. THE VectorDB_Server SHALL resolve all paths from a configurable project root, not hardcoded user-specific absolute paths.
3. THE VectorDB_Server SHALL run on Windows with Python 3.10+ without Docker or external services.
4. THE VectorDB_Server SHALL default to a sentence-transformers model whose max sequence length accommodates the chunk size (recommended: a 512-token model such as `bge-small-en-v1.5`), model name configurable.
5. THE VectorDB_Server SHALL store embedding model name/version in chunk metadata; WHEN the configured model changes, THE system SHALL detect the mismatch and require a full re-index rather than mixing embeddings.
6. THE VectorDB_Server SHALL cache the embedding model locally on first run so later runtime needs no network; the model MAY be pre-bundled for fully offline install.
7. THE VectorDB_Server SHALL store ChromaDB in a configurable local directory (default `<project_root>/vectordb-data/`).
8. THE VectorDB_Server SHALL log to a local file with configurable level (DEBUG, INFO, WARNING, ERROR).
9. WHILE in local mode, THE VectorDB_Server SHALL operate without network access for embeddings or storage.

### Requirement 11: Evaluation and Quality Measurement (Phase 2 — Phase 1 stub)

**User Story:** As a QA engineer, I want to measure quality so I can trust the system and catch regressions.

#### Acceptance Criteria

1. **(Phase 1)** After finalizing chunk size and embedding model, THE team SHALL run a manual spot-check: query a handful of known topics and confirm the expected chunks are retrieved, and generate cases for at least one known story and compare against its existing human-authored test cases.
2. **(Phase 2)** THE system SHALL support a golden set of stories paired with expected test cases.
3. **(Phase 2)** THE system SHALL report retrieval recall@K and mean reciprocal rank against (query → expected chunk) pairs.
4. **(Phase 2)** THE system SHALL report AC coverage percentage and proportion of cases with resolvable citations, and compare against a prior baseline after config/model changes.

### Requirement 12: Data Security and PII Handling (Phase 1 guard + Phase 2 framework)

**User Story:** As a QA engineer on a payroll system, I want sensitive data kept out of the vector store, so the system stays compliant as it scales.

#### Acceptance Criteria

1. **(Phase 1 policy)** THE project SHALL adopt and document a "no production data" rule: only requirements docs, MOMs, UI flows, error-code references, Q&A, Jira tickets, and sanitized test cases are ingested — never production exports or files such as `.env`.
2. **(Phase 1 guard, default ON)** THE Ingestion_Pipeline SHALL run a lightweight pattern guard for high-risk values (e.g., SSN and bank-account number formats); on a match it SHALL redact the value from the stored chunk and log that a redaction occurred (without logging the value). This guard MAY be disabled in config only as a deliberate choice after the corpus is confirmed clean.
3. THE VectorDB_Server SHALL keep all stored data local and SHALL NOT transmit knowledge-base content over the network in local mode.
4. **(Phase 2)** A shared/hosted deployment SHALL add per-user access controls and an authenticated boundary; the local data model SHALL NOT preclude this.

---

## Appendix A: Tool Input/Output Contracts

Functional contracts (field intent), not final schemas.

### ingest_documents
- **In:** `path` (file/dir), optional `category`, optional `recursive` (default true).
- **Out:** `{ files_processed, chunks_added, chunks_replaced, redactions, errors:[{path, reason}] }`.

### ingest_jira_stories
- **In:** `epic_key` OR `issue_key`, optional `fail_fast` (default false).
- **Out (ok):** `{ stories_ingested, chunks_added, failed:[{issue_key, reason}] }`.
- **Out (Jira down):** `{ error:"jira_unavailable", message }`.

### search_knowledge
- **In:** `query`, optional `k` (10), optional `filters` (category, epic_key, issue_key, date_range, path_pattern), optional `rerank` (default false).
- **Out:** `results:[{ text, score, source_path, category, metadata, chunk_id }]`, token-budget capped.

### gather_test_context
- **In:** `description`, `acceptance_criteria`, optional `issue_key`, optional `token_budget` (default 35000).
- **Out (ok):** `{ acceptance_criteria:[{ac_id, text}], few_shot_examples:[{story, test_cases, chunk_id}], ui_flows, error_codes, business_rules, template_spec, default_values, naming_convention, instruction_block }`.
- **Out (empty KB):** `{ error:"empty_knowledge_base", message:"run ingestion first" }`.

### export_test_cases
- **In:** `cases:[{ name, description, precondition, steps:[{num, description, expected}], assigned_to, requirement_id (ac_ids), priority, workplace_capability, citations:[chunk_id] }]`, `story_key` (or per-case), optional `output_dir`.
- **Out (ok):** `{ file_path, sheets, coverage_matrix:{ac_id:[tc_ids]}, low_confidence_acs, citation_warnings:[{tc_name, unresolved_id}] }`.
- **Out (hard-fail):** `{ valid:false, errors:[{type, ac_id|tc_name, message}] }`.

### get_ingestion_status
- **In:** none.
- **Out:** `{ total_documents, total_chunks, by_category, last_ingestion, embedding_model, model_version }`.

### list_stories_without_tests
- **In:** optional `epic_key` filter.
- **Out:** `{ stories_without_tests:[{issue_key, summary, epic_key}], count }`.

### remove_source
- **In:** `source_path` OR `issue_key`.
- **Out:** `{ chunks_removed }`.
