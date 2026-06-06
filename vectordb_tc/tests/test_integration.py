"""Integration tests - require chromadb and sentence-transformers.

Automatically skipped when either dependency is absent so the CI unit-test
suite never breaks on a lean install. Run explicitly with:

    pytest tests/test_integration.py -v

The tests stand up a real ChromaStore + real embedder in a temp directory,
ingest a couple of fixture documents, and assert:
  - chunks land in the store after ingestion
  - keyword and hybrid retrieval find the right content
  - gather_context assembles a valid ContextBundle
  - export_test_cases round-trip produces a valid .xlsx
  - sync() ingests new files and purges orphaned chunks
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip entire module when heavy deps are absent
chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")
sentence_transformers = pytest.importorskip(
    "sentence_transformers", reason="sentence-transformers not installed"
)

from config import AppConfig
from embedder import EmbeddingService
from chunker import Chunker
from store import ChromaStore
from ingestion import IngestionPipeline
from retrieval import RetrievalEngine
from exporter import ExcelExporter
from validator import ExportValidator
from models import AcceptanceCriterion, TestCase, TestStep


# ---------------------------------------------------------------------------
# Module-scoped fixtures — one shared embedder / store across all tests.
# The embedder download is ~130 MB; sharing it avoids repeating that cost.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("integration_project")
    (root / "KnowledgeBase").mkdir()
    (root / "User Stories").mkdir()
    (root / "vectordb-data").mkdir()
    return root


@pytest.fixture(scope="module")
def config(tmp_project):
    return AppConfig(
        project_root=str(tmp_project),
        chunk_size=200,
        chunk_overlap=40,
        pii_guard=True,
    )


@pytest.fixture(scope="module")
def embedder(config):
    return EmbeddingService(config.embedding_model)


@pytest.fixture(scope="module")
def store(config):
    return ChromaStore(config.chromadb_path)


@pytest.fixture(scope="module")
def pipeline(config, embedder, store):
    chunker = Chunker(embedder, config.chunk_size, config.chunk_overlap)
    return IngestionPipeline(config, chunker, embedder, store)


@pytest.fixture(scope="module")
def retriever(embedder, store, config):
    return RetrievalEngine(embedder, store, config)


@pytest.fixture(scope="module", autouse=True)
def seed_knowledge_base(tmp_project, pipeline):
    """Ingest two fixture documents once before any test in the module runs."""
    kb = tmp_project / "KnowledgeBase"

    (kb / "business_rules.md").write_text(
        "# Payment Rules\n\n"
        "Payments above $10,000 require manager approval.\n"
        "Payments in foreign currency are converted at the daily rate.\n\n"
        "## Error Codes\n\n"
        "| Code | Meaning |\n"
        "| --- | --- |\n"
        "| PY075 | Invalid SSN |\n"
        "| PY076 | Missing required field |\n"
        "| PY077 | Bad date format |",
        encoding="utf-8",
    )
    (kb / "ui_flow_login.md").write_text(
        "# Login Flow\n\n"
        "1. User navigates to /login.\n"
        "2. User enters username and password.\n"
        "3. System validates credentials against the identity provider.\n"
        "4. On success, redirect to dashboard with a session token.\n"
        "5. On failure, show error message and increment the lockout counter.",
        encoding="utf-8",
    )
    pipeline.ingest_directory(str(kb))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_store_has_chunks_after_ingest(store):
    stats = store.get_stats()
    assert stats["total_chunks"] > 0, "No chunks were ingested"


def test_keyword_search_finds_error_code(store):
    results = store.keyword_search("PY075", top_k=5)
    assert results, "keyword_search returned nothing"
    assert any("PY075" in r.chunk.text for r in results)


def test_hybrid_retrieval_finds_error_identifier(retriever):
    results = retriever.search("what does PY075 mean", top_k=5)
    assert any("PY075" in r.chunk.text for r in results)


def test_dense_search_finds_payment_rule(retriever):
    results = retriever.search("large payment approval threshold", top_k=5)
    assert results
    assert any("approval" in r.chunk.text.lower() for r in results)


def test_hash_change_detection_skips_unchanged(pipeline, tmp_project, store):
    path = str(tmp_project / "KnowledgeBase" / "business_rules.md")
    result = pipeline.ingest_file(path)
    assert result.skipped_unchanged, "Unchanged file should be skipped"
    assert result.chunks_created == 0


def test_pii_redacted_before_store(pipeline, tmp_project, store):
    pii_file = tmp_project / "KnowledgeBase" / "pii_test.txt"
    pii_file.write_text("Customer SSN is 123-45-6789 and account 12345678901234.",
                        encoding="utf-8")
    pipeline.ingest_file(str(pii_file))
    results = store.keyword_search("123-45-6789", top_k=5)
    assert not results, "Raw SSN should not appear in the store after redaction"
    pii_file.unlink()


def test_gather_context_assembles_bundle(retriever):
    bundle = retriever.gather_context(
        description="User submits a cross-border payment form",
        acceptance_criteria=(
            "1. System validates payment amount\n"
            "2. Manager approval required for amounts over $10,000\n"
            "3. Currency conversion applied at daily rate"
        ),
        issue_key="TEST-001",
    )
    assert bundle.story_description
    assert len(bundle.acceptance_criteria) == 3
    assert bundle.acceptance_criteria[0].ac_id == "AC-1"
    assert bundle.acceptance_criteria[2].ac_id == "AC-3"
    assert bundle.total_tokens >= 0
    assert bundle.token_budget > 0


def test_export_round_trip(tmp_project):
    acs = [
        AcceptanceCriterion("AC-1", "System validates payment amount"),
        AcceptanceCriterion("AC-2", "Manager approval for amounts over $10,000"),
    ]
    cases = [
        TestCase(
            name="TEST-001_TC-01 : valid payment accepted",
            description="Submit a valid domestic payment under the threshold",
            precondition="User is logged in",
            steps=[TestStep("1", "Enter payment amount $500", "Payment processed")],
            assigned_to="qa", requirement_id="TEST-001",
            ac_ids=["AC-1"], case_type="positive", citations=[],
        ),
        TestCase(
            name="TEST-001_TC-02 : large payment triggers approval",
            description="Submit payment above manager-approval threshold",
            precondition="User is logged in",
            steps=[TestStep("1", "Enter payment amount $15,000", "Approval request sent")],
            assigned_to="qa", requirement_id="TEST-001",
            ac_ids=["AC-2"], case_type="positive", citations=[],
        ),
        TestCase(
            name="TEST-001_TC-03 : invalid amount rejected",
            description="Submit payment with a negative amount",
            precondition="User is logged in",
            steps=[TestStep("1", "Enter amount -$100", "Error message displayed")],
            assigned_to="qa", requirement_id="TEST-001",
            ac_ids=["AC-1"], case_type="negative", citations=[],
        ),
    ]

    result = ExportValidator().validate(cases, acs, "TEST-001", set())
    assert result.valid, [str(e) for e in result.errors]
    assert set(result.coverage_matrix) >= {"AC-1", "AC-2"}

    out_path = ExcelExporter().export(cases, "TEST-001",
                                      str(tmp_project / "User Stories"))
    assert Path(out_path).exists()
    assert out_path.endswith(".xlsx")


def test_sync_ingests_new_file(pipeline, store, tmp_project):
    kb = tmp_project / "KnowledgeBase"
    new_file = kb / "sync_new.txt"
    new_file.write_text("New business rule: all wire transfers must be logged.",
                        encoding="utf-8")

    before = store.get_stats()["total_chunks"]
    pipeline.sync(str(kb))
    after = store.get_stats()["total_chunks"]
    assert after > before, "sync should have added chunks for the new file"

    new_file.unlink()


def test_sync_purges_orphaned_chunks(pipeline, store, tmp_project):
    kb = tmp_project / "KnowledgeBase"
    orphan = kb / "soon_deleted.txt"
    orphan.write_text("Temporary document to be removed.", encoding="utf-8")
    pipeline.ingest_file(str(orphan))
    assert store.hash_for_source(str(orphan)) is not None, "orphan not ingested"

    orphan.unlink()
    pipeline.sync(str(kb))

    assert store.hash_for_source(str(orphan)) is None, \
        "sync should have deleted chunks for the removed file"
