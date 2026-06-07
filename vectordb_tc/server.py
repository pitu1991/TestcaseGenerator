"""FastMCP entry point. Wires the 8 tools and returns STRUCTURED errors the host
LLM can act on (empty KB, Jira unavailable, validation failures)."""
from __future__ import annotations

import json
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from config import AppConfig
from chunker import Chunker
from conflict import ConflictDetector
from delta import DeltaEngine
from embedder import EmbeddingService
from exporter import ExcelExporter
from governance import GovernanceStore
from ingestion import IngestionPipeline
from models import AcceptanceCriterion, TestCase, TestStep
from notifier import LogNotifier
from retrieval import EmptyKnowledgeBase, RetrievalEngine
from store import ChromaStore
from validator import ExportValidator

CONFIG = AppConfig()  # or AppConfig.from_file("config.yaml")
EMBEDDER = EmbeddingService(CONFIG.embedding_model)
STORE = ChromaStore(CONFIG.chromadb_path)
CHUNKER = Chunker(EMBEDDER, CONFIG.chunk_size, CONFIG.chunk_overlap)
GOVERNANCE = GovernanceStore(CONFIG.governance_db_path)
DETECTOR = ConflictDetector(STORE, GOVERNANCE, CONFIG.conflict_similarity_threshold,
                            CONFIG.project_id, notifier=LogNotifier())
INGEST = IngestionPipeline(CONFIG, CHUNKER, EMBEDDER, STORE, conflict_detector=DETECTOR)
RETRIEVE = RetrievalEngine(EMBEDDER, STORE, CONFIG)
DELTA = DeltaEngine(STORE)
EXPORTER = ExcelExporter()
VALIDATOR = ExportValidator()

mcp = FastMCP("vectordb-tc-server")


def _err(kind: str, message: str, **extra) -> str:
    return json.dumps({"error": kind, "message": message, **extra})


@mcp.tool()
def ingest_documents(path: str, category: str = "auto", module: str = "auto") -> str:
    if INGEST.model_needs_reindex():
        return _err("model_changed", "Embedding model differs from stored data; re-index required.")
    import os
    if os.path.isdir(path):
        results = INGEST.ingest_directory(path, category, module)
    else:
        results = [INGEST.ingest_file(path, category, module)]
    return json.dumps([asdict(r) for r in results])


@mcp.tool()
def migrate_versioning() -> str:
    """One-time, idempotent: stamp version/is_latest/document_id on pre-Phase-A
    chunks. Run once after upgrading, BEFORE relying on is_latest filtering."""
    migrated = STORE.backfill_versioning()
    return json.dumps({"chunks_migrated": migrated})


@mcp.tool()
def ingest_jira_stories(key: str, key_type: str = "epic", fail_fast: bool = False) -> str:
    from jira_client import JiraClient  # local import keeps server import-light
    try:
        client = JiraClient(CONFIG.jira_server_path)
    except Exception as e:  # noqa: BLE001 - surface a clean error, don't crash
        return _err("jira_unavailable", f"Jira MCP not reachable: {e}")
    try:
        if key_type == "epic":
            res = INGEST.ingest_jira_epic(key, client, fail_fast=fail_fast)
        else:
            res = INGEST.ingest_jira_story(key, client)
    except NotImplementedError:
        return _err("not_implemented", "Jira ingestion not yet wired.")
    return json.dumps(asdict(res))


@mcp.tool()
def search_knowledge(query: str, filters: dict | None = None, top_k: int = 10,
                     include_historical: bool = False) -> str:
    try:
        results = RETRIEVE.search(query, filters=filters, top_k=top_k,
                                  include_historical=include_historical)
    except EmptyKnowledgeBase as e:
        return _err("empty_knowledge_base", str(e))
    return json.dumps([{
        "text": r.chunk.text, "score": r.relevance_score,
        "source_path": r.chunk.source_path, "category": r.chunk.category,
        "chunk_id": r.chunk.id, "match_type": r.match_type,
        "metadata": r.chunk.metadata,
    } for r in results])


@mcp.tool()
def get_document_versions(document_id: str) -> str:
    """List stored version numbers for a document (newest first)."""
    return json.dumps({"document_id": document_id,
                       "versions": STORE.versions_for_document(document_id)})


@mcp.tool()
def search_delta_changes(document_id: str, from_version: int | None = None,
                         to_version: int | None = None) -> str:
    """Section-level diff between two versions (defaults to previous -> latest).
    Returns added/changed/removed sections so generation can target only those."""
    delta = DELTA.diff_versions(document_id, from_version, to_version)
    return json.dumps(asdict(delta))


@mcp.tool()
def gather_delta_context(document_id: str, issue_key: str | None = None,
                         from_version: int | None = None,
                         to_version: int | None = None) -> str:
    """Assemble context for DELTA regeneration: the changed sections plus the
    existing test cases most relevant to them. Directs the host LLM to regenerate
    only impacted cases, preserve the rest, then call export_test_cases."""
    delta = DELTA.diff_versions(document_id, from_version, to_version)
    if not delta.sections:
        return json.dumps({"document_id": document_id, "no_changes": True,
                           "from_version": delta.from_version, "to_version": delta.to_version})

    query = "\n".join((s.new_text or s.old_text) for s in delta.sections)
    try:
        existing = RETRIEVE.relevant_test_cases(query, issue_key=issue_key, top_k=8)
    except EmptyKnowledgeBase:
        existing = []

    return json.dumps({
        "document_id": document_id,
        "from_version": delta.from_version, "to_version": delta.to_version,
        "changed_sections": [asdict(s) for s in delta.sections],
        "unchanged_sections": delta.unchanged_sections,
        "existing_test_cases": [{"text": r.chunk.text, "chunk_id": r.chunk.id,
                                 "issue_key": r.chunk.metadata.get("issue_key", ""),
                                 "score": r.relevance_score} for r in existing],
        "template_spec": {"headers": ExcelExporter.HEADERS, "defaults": ExcelExporter.DEFAULTS,
                          "naming_convention": "{ISSUE_KEY}_TC-{NN} : {short description}"},
        "instruction_block": (
            "Only these sections changed between the two versions. Regenerate ONLY "
            "the test cases impacted by the added/changed sections; keep existing test "
            "cases for unchanged behavior as-is. Include at least one negative case for "
            "new or changed behavior, cite source chunk_ids, follow the naming "
            "convention, then call export_test_cases with the full merged set so "
            "validation (AC coverage + negative case) still holds."
        ),
    })


@mcp.tool()
def gather_test_context(description: str, acceptance_criteria: str,
                        issue_key: str | None = None) -> str:
    try:
        bundle = RETRIEVE.gather_context(description, acceptance_criteria, issue_key)
    except EmptyKnowledgeBase as e:
        return _err("empty_knowledge_base", str(e))
    bundle.template_spec = {
        "headers": ExcelExporter.HEADERS, "defaults": ExcelExporter.DEFAULTS,
        "naming_convention": "{ISSUE_KEY}_TC-{NN} : {short description}",
    }
    bundle.instruction_block = (
        "Produce test cases as JSON. Cover EVERY acceptance criterion (set ac_ids), "
        "include at least one negative case, cite source chunk_ids in 'citations', "
        "and follow the naming convention. Then call export_test_cases."
    )
    def srs(items):
        return [{"text": r.chunk.text, "chunk_id": r.chunk.id,
                 "source_path": r.chunk.source_path, "score": r.relevance_score} for r in items]
    return json.dumps({
        "acceptance_criteria": [asdict(a) for a in bundle.acceptance_criteria],
        "few_shot_examples": srs(bundle.similar_stories),
        "ui_flows": srs(bundle.ui_flows), "error_codes": srs(bundle.error_codes),
        "business_rules": srs(bundle.business_rules), "meeting_notes": srs(bundle.meeting_notes),
        "linked_issues": bundle.linked_issues, "template_spec": bundle.template_spec,
        "instruction_block": bundle.instruction_block,
        "total_tokens": bundle.total_tokens, "token_budget": bundle.token_budget,
    })


@mcp.tool()
def export_test_cases(test_cases: list, acceptance_criteria: list,
                      issue_key: str, context_chunk_ids: list | None = None,
                      output_dir: str | None = None) -> str:
    """test_cases: list of dicts from the host LLM. Validated before writing."""
    acs = [AcceptanceCriterion(a["ac_id"], a["text"]) for a in acceptance_criteria]
    cases = [_to_testcase(d) for d in test_cases]
    result = VALIDATOR.validate(cases, acs, issue_key, set(context_chunk_ids or []))
    if not result.valid:
        return json.dumps({"valid": False,
                           "errors": [asdict(e) for e in result.errors]})
    path = EXPORTER.export(cases, issue_key, output_dir or CONFIG.output_dir)
    return json.dumps({
        "valid": True, "file_path": path,
        "coverage_matrix": result.coverage_matrix,
        "low_confidence_acs": result.low_confidence_acs,
        "citation_warnings": result.citation_warnings,
    })


@mcp.tool()
def get_ingestion_status() -> str:
    stats = STORE.get_stats()
    stats.update({"embedding_model": EMBEDDER.model_name,
                  "model_version": EMBEDDER.model_version,
                  "open_conflicts": len(GOVERNANCE.open_conflict_chunk_ids(CONFIG.project_id))})
    return json.dumps(stats)


@mcp.tool()
def list_stories_without_tests(epic_key: str | None = None) -> str:
    stories = STORE.issue_keys_by_category("Story")
    tested = STORE.issue_keys_by_category("Test_Case")
    missing = sorted(stories - tested)
    return json.dumps({"stories_without_tests": missing, "count": len(missing)})


@mcp.tool()
def get_conflicts(status: str | None = None) -> str:
    """List governance conflict records, optionally filtered by status
    (suspected | confirmed | dismissed | resolved)."""
    conflicts = GOVERNANCE.get_conflicts(status=status, project_id=CONFIG.project_id)
    return json.dumps([asdict(c) for c in conflicts])


@mcp.tool()
def get_conflict_candidates() -> str:
    """Suspected conflicts awaiting LLM adjudication, with BOTH chunk texts so the
    host LLM can judge whether they actually contradict. The LLM then calls
    record_conflict_verdict. (The detector proposes pairs; it never decides truth.)"""
    out = []
    for c in GOVERNANCE.get_conflicts(status="suspected", project_id=CONFIG.project_id):
        chunks = {ch.id: ch for ch in STORE.get_chunks_by_ids([c.chunk_a_id, c.chunk_b_id])}
        a, b = chunks.get(c.chunk_a_id), chunks.get(c.chunk_b_id)
        out.append({
            "conflict_id": c.conflict_id, "module": c.module, "similarity": c.similarity,
            "source_a": c.source_a, "text_a": a.text if a else "",
            "source_b": c.source_b, "text_b": b.text if b else "",
        })
    return json.dumps(out)


@mcp.tool()
def record_conflict_verdict(conflict_id: str, is_conflict: bool, rationale: str = "") -> str:
    """Record the host LLM's adjudication: is_conflict=true -> 'confirmed',
    false -> 'dismissed'. Stores the rationale for audit."""
    ok = GOVERNANCE.record_verdict(conflict_id, is_conflict, rationale)
    if not ok:
        return _err("not_found", f"No conflict with id {conflict_id}")
    return json.dumps({"conflict_id": conflict_id,
                       "status": "confirmed" if is_conflict else "dismissed"})


@mcp.tool()
def resolve_conflict(conflict_id: str, resolution_text: str, decision: str = "new_rule",
                     approver: str = "", authority_level: int = 100) -> str:
    """Record a human-approved resolution: re-ingest the authoritative text as a
    high-authority Business_Resolution chunk, link it to the conflict, and mark the
    conflict resolved. decision: source_a | source_b | merge | new_rule."""
    conflict = GOVERNANCE.get_conflict(conflict_id)
    if conflict is None:
        return _err("not_found", f"No conflict with id {conflict_id}")
    import uuid
    rid = uuid.uuid4().hex[:12]
    ing = INGEST.ingest_resolution(resolution_text, rid, module=conflict.module,
                                   extra={"conflict_id": conflict_id})
    art = GOVERNANCE.create_resolution(
        conflict_id, decision, resolution_text, approver, authority_level,
        resolution_chunk_id=ing.document_id, resolution_id=rid)
    return json.dumps({"resolution": asdict(art), "conflict_id": conflict_id,
                       "resolution_chunk": ing.document_id})


@mcp.tool()
def search_authoritative_knowledge(query: str, top_k: int = 10) -> str:
    """Search with governance authority ranking on: approved Business_Resolutions
    outrank source documents at equal relevance."""
    try:
        results = RETRIEVE.search(query, top_k=top_k, authority_boost=True)
    except EmptyKnowledgeBase as e:
        return _err("empty_knowledge_base", str(e))
    return json.dumps([{"text": r.chunk.text, "score": r.relevance_score,
                        "category": r.chunk.category, "chunk_id": r.chunk.id,
                        "authority_score": r.chunk.authority_score,
                        "source_path": r.chunk.source_path} for r in results])


@mcp.tool()
def start_review_ui(port: int = 8765) -> str:
    """Launch the local human-in-the-loop conflict review UI (localhost only) on a
    background thread. Reviewers confirm/dismiss and approve authoritative
    resolutions; resolutions are re-ingested as high-authority knowledge."""
    import uuid
    from review_app import ReviewService, run_in_thread

    def resolve_fn(conflict_id, decision, text, approver):
        conflict = GOVERNANCE.get_conflict(conflict_id)
        rid = uuid.uuid4().hex[:12]
        ing = INGEST.ingest_resolution(text, rid,
                                       module=conflict.module if conflict else "default",
                                       extra={"conflict_id": conflict_id})
        return GOVERNANCE.create_resolution(conflict_id, decision, text, approver, 100,
                                            resolution_chunk_id=ing.document_id,
                                            resolution_id=rid)

    def verdict_fn(conflict_id, is_conflict, rationale):
        return GOVERNANCE.record_verdict(conflict_id, is_conflict, rationale)

    service = ReviewService(GOVERNANCE, STORE, resolve_fn, verdict_fn)
    run_in_thread(service, port=port)
    return json.dumps({"status": "started", "url": f"http://127.0.0.1:{port}/"})


@mcp.tool()
def search_unresolved_conflicts() -> str:
    """Chunks still entangled in suspected/confirmed conflicts. Generation should
    avoid grounding new test cases on these until the conflict is resolved."""
    ids = sorted(GOVERNANCE.open_conflict_chunk_ids(CONFIG.project_id))
    chunks = STORE.get_chunks_by_ids(ids)
    return json.dumps([{"chunk_id": c.id, "text": c.text, "source_path": c.source_path,
                        "category": c.category} for c in chunks])


@mcp.tool()
def remove_source(source: str) -> str:
    # Accepts a file path OR an issue key ("jira:KEY" stored form is handled too).
    removed = STORE.delete_by_source(source)
    if removed == 0 and not source.startswith("jira:"):
        removed = STORE.delete_by_source(f"jira:{source}")
    return json.dumps({"chunks_removed": removed})


def _to_testcase(d: dict) -> TestCase:
    return TestCase(
        name=d.get("name", ""), description=d.get("description", ""),
        precondition=d.get("precondition", ""),
        steps=[TestStep(str(s.get("step_number", i + 1)), s.get("description", ""),
                        s.get("expected_result", "")) for i, s in enumerate(d.get("steps", []))],
        assigned_to=d.get("assigned_to", ""), requirement_id=d.get("requirement_id", ""),
        ac_ids=d.get("ac_ids", []), case_type=d.get("case_type", "positive"),
        citations=d.get("citations", []), resolution_ids=d.get("resolution_ids", []),
        workplace_capability=d.get("workplace_capability", ""), priority=d.get("priority", ""),
    )


if __name__ == "__main__":
    mcp.run()
