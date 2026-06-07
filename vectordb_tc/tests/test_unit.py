"""Unit + property tests for the deterministic, dependency-light logic.

These run WITHOUT chromadb or sentence-transformers (those are imported lazily
inside __init__, not at module import time), using a FakeEmbedder. Integration
tests that need a real store/model live separately and are skipped if deps are
absent."""
from __future__ import annotations

from pathlib import Path

import pytest

from models import (
    AcceptanceCriterion, Chunk, SearchResult, TestCase, TestStep,
)
from models import authority_for
from validator import ExportValidator
from chunker import Chunker
from conflict import ConflictDetector
from delta import DeltaEngine
from governance import GovernanceStore
from notifier import LogNotifier, Notifier
from retrieval import RetrievalEngine
from review_app import ReviewService
from ingestion import IngestionPipeline, redact_pii
from jira_client import adf_to_text
from config import AppConfig


class FakeEmbedder:
    """Counts tokens as whitespace words; good enough to exercise clamping/splitting."""
    model_name = "fake"
    model_version = "fake@dim8"

    def __init__(self, max_seq_length: int = 512):
        self.max_seq_length = max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_texts(self, texts):
        return [[float(len(t))] * 8 for t in texts]

    def embed_query(self, q):
        return [float(len(q))] * 8


def _chunk(cid: str, text: str = "x") -> Chunk:
    return Chunk(cid, text, "src", "t", 0, 1, "Story", "ts", "h", "fake", "fake@dim8", {})


def _result(cid: str, mt="dense", score=1.0) -> SearchResult:
    return SearchResult(_chunk(cid), score, mt)


def _valid_case(key="TWCJ-6184", n=1, ctype="positive", acs=("AC-1",), cites=("h_0",)):
    return TestCase(
        name=f"{key}_TC-{n:02d} : does a thing", description="d", precondition="p",
        steps=[TestStep("1", "do", "expect")], assigned_to="qa", requirement_id=key,
        ac_ids=list(acs), case_type=ctype, citations=list(cites),
    )


# --- chunker: model coupling (the v1 bug must stay fixed) --------------------
def test_chunker_clamps_to_model_max():
    ch = Chunker(FakeEmbedder(max_seq_length=512), chunk_size=750, overlap=80)
    assert ch.max_tokens == 512  # never exceeds what the model can embed


def test_chunker_respects_smaller_chunk_size():
    ch = Chunker(FakeEmbedder(max_seq_length=512), chunk_size=400, overlap=80)
    assert ch.max_tokens == 400


def test_prose_chunks_never_exceed_budget():
    emb = FakeEmbedder(max_seq_length=20)
    ch = Chunker(emb, chunk_size=20, overlap=4)
    text = " ".join(f"word{i}" for i in range(200))
    chunks = ch.chunk_document(text, _base())
    assert chunks
    assert all(emb.count_tokens(c.text) <= 20 for c in chunks)


def test_table_split_repeats_header():
    emb = FakeEmbedder(max_seq_length=14)
    ch = Chunker(emb, chunk_size=14, overlap=2)
    table = "\n".join([
        "| Code | Meaning |", "| --- | --- |",
        "| PY075 | Invalid SSN |", "| PY076 | Missing field |",
        "| PY077 | Bad date |", "| PY078 | Future date |",
    ])
    chunks = ch.chunk_document(table, _base())
    assert len(chunks) >= 2
    assert all("Code" in c.text for c in chunks)  # header repeated in each


def _base() -> dict:
    return {"source_path": "src", "document_title": "t", "content_hash": "abcdef123456",
            "category": "Business_Rule", "ingestion_timestamp": "ts",
            "embedding_model": "fake", "model_version": "fake@dim8", "metadata": {}}


# --- retrieval: RRF fusion ---------------------------------------------------
def test_rrf_ranks_shared_ids_highest():
    dense = [_result("a"), _result("b"), _result("c")]
    keyword = [_result("c", "keyword"), _result("d", "keyword")]
    fused = RetrievalEngine._rrf([dense, keyword])
    ids = [r.chunk.id for r in fused]
    assert set(ids) == {"a", "b", "c", "d"}     # no dropped results
    assert ids[0] == "c"                          # appears in both -> top
    assert any(r.match_type == "hybrid" for r in fused if r.chunk.id == "c")


def test_extract_keywords_finds_identifiers():
    eng = RetrievalEngine(FakeEmbedder(), None, AppConfig())
    kws = eng._extract_keywords("error PY075 on issue TWCJ-6184 in fileUpload")
    assert "PY075" in kws and "TWCJ-6184" in kws


def test_enumerate_acs_from_blob_and_list():
    eng = RetrievalEngine(FakeEmbedder(), None, AppConfig())
    blob = eng._enumerate_acs("1. first\n2. second\n3. third")
    assert [a.ac_id for a in blob] == ["AC-1", "AC-2", "AC-3"]
    lst = eng._enumerate_acs(["only one"])
    assert lst[0].ac_id == "AC-1" and lst[0].text == "only one"


# --- validator: hard rules + soft citation ----------------------------------
def test_validator_passes_complete_set():
    acs = [AcceptanceCriterion("AC-1", "a")]
    cases = [_valid_case(acs=("AC-1",)), _valid_case(n=2, ctype="negative", acs=("AC-1",))]
    res = ExportValidator().validate(cases, acs, "TWCJ-6184", {"h_0"})
    assert res.valid and res.coverage_matrix["AC-1"]


def test_validator_flags_uncovered_ac():
    acs = [AcceptanceCriterion("AC-1", "a"), AcceptanceCriterion("AC-2", "b")]
    cases = [_valid_case(acs=("AC-1",)), _valid_case(n=2, ctype="negative", acs=("AC-1",))]
    res = ExportValidator().validate(cases, acs, "TWCJ-6184", {"h_0"})
    assert not res.valid
    assert any(e.type == "uncovered_ac" and e.target == "AC-2" for e in res.errors)


def test_validator_requires_negative_case():
    acs = [AcceptanceCriterion("AC-1", "a")]
    cases = [_valid_case(acs=("AC-1",))]  # only positive
    res = ExportValidator().validate(cases, acs, "TWCJ-6184", {"h_0"})
    assert any(e.type == "no_negative" for e in res.errors)


def test_validator_flags_bad_name():
    acs = [AcceptanceCriterion("AC-1", "a")]
    bad = _valid_case(acs=("AC-1",)); bad.name = "wrong-name"
    neg = _valid_case(n=2, ctype="negative", acs=("AC-1",))
    res = ExportValidator().validate([bad, neg], acs, "TWCJ-6184", {"h_0"})
    assert any(e.type == "bad_name" for e in res.errors)


def test_validator_citation_is_warning_not_error():
    acs = [AcceptanceCriterion("AC-1", "a")]
    cases = [_valid_case(acs=("AC-1",), cites=("ghost_id",)),
             _valid_case(n=2, ctype="negative", acs=("AC-1",), cites=("ghost_id",))]
    res = ExportValidator().validate(cases, acs, "TWCJ-6184", {"h_0"})
    assert res.valid                                   # soft: still exports
    assert res.citation_warnings                       # but flagged
    assert "AC-1" in res.low_confidence_acs


@pytest.mark.parametrize("n_acs", [1, 3, 5])
def test_property_every_ac_in_coverage_matrix(n_acs):
    acs = [AcceptanceCriterion(f"AC-{i+1}", f"crit {i}") for i in range(n_acs)]
    cases = [_valid_case(acs=tuple(a.ac_id for a in acs)),
             _valid_case(n=2, ctype="negative", acs=tuple(a.ac_id for a in acs))]
    res = ExportValidator().validate(cases, acs, "TWCJ-6184", {"h_0"})
    assert set(res.coverage_matrix) >= {a.ac_id for a in acs}
    assert res.valid


# --- PII guard ---------------------------------------------------------------
def test_pii_redaction():
    text, n = redact_pii("SSN 123-45-6789 and account 12345678901 here")
    assert "[REDACTED-SSN]" in text and "[REDACTED-ACCT]" in text
    assert n == 2
    assert "123-45-6789" not in text


# --- issue key extraction ----------------------------------------------------
def test_issue_key_from_sheet_title():
    assert IngestionPipeline._issue_key_for_sheet("TWCJ-6184", [("h",)]) == "TWCJ-6184"


def test_issue_key_from_tc_name_cell():
    rows = [("Name", "Desc"), ("TWCJ-6184_TC-01 : login", "x")]
    assert IngestionPipeline._issue_key_for_sheet("Sheet1", rows) == "TWCJ-6184"


# --- ADF flattener -----------------------------------------------------------
def test_adf_basic_paragraph_and_heading():
    doc = {"type": "doc", "content": [
        {"type": "heading", "attrs": {"level": 2},
         "content": [{"type": "text", "text": "Title"}]},
        {"type": "paragraph", "content": [{"type": "text", "text": "Hello world"}]},
    ]}
    out = adf_to_text(doc)
    assert "## Title" in out and "Hello world" in out


def test_adf_table_rows_joined():
    table = {"type": "table", "content": [
        {"type": "tableRow", "content": [
            {"type": "tableHeader", "content": [{"type": "text", "text": "Code"}]},
            {"type": "tableHeader", "content": [{"type": "text", "text": "Meaning"}]}]},
        {"type": "tableRow", "content": [
            {"type": "tableCell", "content": [{"type": "text", "text": "PY075"}]},
            {"type": "tableCell", "content": [{"type": "text", "text": "Invalid SSN"}]}]},
    ]}
    out = adf_to_text(table)
    assert "| Code | Meaning |" in out
    assert "| PY075 | Invalid SSN |" in out


def test_adf_handles_none_and_str():
    assert adf_to_text(None) == ""
    assert adf_to_text("plain") == "plain"


# --- Phase A: versioning -----------------------------------------------------
class FakeStore:
    """In-memory store exercising the versioning state machine without ChromaDB."""

    def __init__(self):
        self.chunks: dict[str, Chunk] = {}   # id -> Chunk

    def latest_version_for_document(self, document_id):
        for c in self.chunks.values():
            if c.document_id == document_id and c.is_latest:
                return (c.version, c.content_hash)
        return None

    def mark_not_latest(self, document_id):
        n = 0
        for c in self.chunks.values():
            if c.document_id == document_id and c.is_latest:
                c.is_latest = False
                c.superseded_at = "ts"
                n += 1
        return n

    def upsert_chunks(self, chunks, embeddings):
        for c in chunks:
            self.chunks[c.id] = c

    def prune_old_versions(self, document_id, max_versions):
        versions = sorted({c.version for c in self.chunks.values()
                           if c.document_id == document_id}, reverse=True)
        if len(versions) <= max_versions:
            return 0
        keep = set(versions[:max_versions])
        drop = [cid for cid, c in self.chunks.items()
                if c.document_id == document_id and c.version not in keep]
        for cid in drop:
            del self.chunks[cid]
        return len(drop)

    def versions_for_document(self, document_id):
        return sorted({c.version for c in self.chunks.values()
                       if c.document_id == document_id}, reverse=True)

    def chunks_for_document(self, document_id, version=None):
        if version is None:
            cs = [c for c in self.chunks.values()
                  if c.document_id == document_id and c.is_latest]
        else:
            cs = [c for c in self.chunks.values()
                  if c.document_id == document_id and c.version == version]
        return sorted(cs, key=lambda c: c.chunk_index)


def _pipeline(tmp_path, store=None, max_versions=0):
    cfg = AppConfig(project_root=str(tmp_path), pii_guard=False,
                    max_versions_retained=max_versions)
    emb = FakeEmbedder()
    return IngestionPipeline(cfg, Chunker(emb), emb, store or FakeStore())


def _write(tmp_path, rel, text):
    p = Path(tmp_path) / "KnowledgeBase" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_new_document_is_version_1(tmp_path):
    store = FakeStore()
    pipe = _pipeline(tmp_path, store)
    p = _write(tmp_path, "auth/login.md", "# Login\nUsername and password")
    r = pipe.ingest_file(str(p))
    assert r.version == 1 and not r.skipped_unchanged
    assert all(c.is_latest and c.version == 1 for c in store.chunks.values())
    assert all(c.document_id == "auth/login.md" for c in store.chunks.values())
    assert all(c.module == "auth" for c in store.chunks.values())   # folder-derived


def test_unchanged_reingest_skips(tmp_path):
    store = FakeStore()
    pipe = _pipeline(tmp_path, store)
    p = _write(tmp_path, "a.md", "same content here")
    pipe.ingest_file(str(p))
    n_before = len(store.chunks)
    r = pipe.ingest_file(str(p))
    assert r.skipped_unchanged and r.version == 1
    assert len(store.chunks) == n_before          # nothing added


def test_changed_document_creates_v2_and_retains_v1(tmp_path):
    store = FakeStore()
    pipe = _pipeline(tmp_path, store)
    p = _write(tmp_path, "a.md", "first version text")
    pipe.ingest_file(str(p))
    p.write_text("second version different text", encoding="utf-8")
    r = pipe.ingest_file(str(p))
    assert r.version == 2
    v1 = [c for c in store.chunks.values() if c.version == 1]
    v2 = [c for c in store.chunks.values() if c.version == 2]
    assert v1 and v2                               # Option B: both retained
    assert all(not c.is_latest for c in v1)        # old flipped
    assert all(c.is_latest for c in v2)            # new is latest


def test_module_explicit_param_overrides_folder(tmp_path):
    store = FakeStore()
    pipe = _pipeline(tmp_path, store)
    p = _write(tmp_path, "auth/login.md", "x")
    pipe.ingest_file(str(p), module="Authentication")
    assert all(c.module == "Authentication" for c in store.chunks.values())


def test_retention_prunes_oldest_versions(tmp_path):
    store = FakeStore()
    pipe = _pipeline(tmp_path, store, max_versions=2)
    p = _write(tmp_path, "a.md", "init")
    for text in ["v1 text", "v2 text", "v3 text"]:
        p.write_text(text, encoding="utf-8")
        pipe.ingest_file(str(p))
    assert {c.version for c in store.chunks.values()} == {2, 3}   # v1 pruned


def test_document_id_outside_knowledge_dir_is_absolute(tmp_path):
    pipe = _pipeline(tmp_path)
    outside = Path(tmp_path) / "elsewhere.md"
    outside.write_text("x", encoding="utf-8")
    assert pipe._document_id_for_file(outside) == outside.resolve().as_posix()


def test_chunk_id_includes_version():
    ch = Chunker(FakeEmbedder())
    base = _base(); base["version"] = 3; base["document_id"] = "d"
    chunks = ch.chunk_document("hello world", base)
    assert chunks[0].id.endswith("_v3") and chunks[0].version == 3
    assert chunks[0].chunk_hash                      # per-chunk hash stamped


def test_section_captured_from_heading():
    ch = Chunker(FakeEmbedder(max_seq_length=8), chunk_size=8)
    text = "# Auth\nsome login words\n# Payment\nother checkout words"
    secs = {c.section for c in ch.chunk_document(text, _base())}
    assert "Auth" in secs and "Payment" in secs


def test_with_latest_merges_filters():
    f = RetrievalEngine._with_latest
    assert f(None, False) == {"is_latest": True}
    assert f(None, True) is None
    assert f({"category": "Story"}, False) == {"$and": [{"category": "Story"}, {"is_latest": True}]}
    assert f({"category": "Story"}, True) == {"category": "Story"}
    # existing $and is flattened, not nested (ChromaDB rejects nested $and)
    assert f({"$and": [{"a": 1}, {"b": 2}]}, False) == \
        {"$and": [{"a": 1}, {"b": 2}, {"is_latest": True}]}


# --- Phase B: delta generation -----------------------------------------------
def test_delta_detects_changed_added_unchanged(tmp_path):
    store = FakeStore()
    pipe = _pipeline(tmp_path, store)
    p = _write(tmp_path, "auth.md",
               "# Auth\nUsername and password then OTP.\n"
               "# Profile\nView and edit profile details.")
    pipe.ingest_file(str(p))
    p.write_text(
        "# Auth\nUsername and password then CAPTCHA then OTP.\n"
        "# Profile\nView and edit profile details.\n"
        "# Notifications\nManage email notifications.", encoding="utf-8")
    pipe.ingest_file(str(p))

    delta = DeltaEngine(store).diff_versions("auth.md")
    assert (delta.from_version, delta.to_version) == (1, 2)
    changes = {s.section: s.change for s in delta.sections}
    assert changes.get("Auth") == "changed"
    assert changes.get("Notifications") == "added"
    assert "Profile" in delta.unchanged_sections
    auth = next(s for s in delta.sections if s.section == "Auth")
    assert "CAPTCHA" in auth.new_text and "CAPTCHA" not in auth.old_text


def test_delta_single_version_is_all_added(tmp_path):
    store = FakeStore()
    pipe = _pipeline(tmp_path, store)
    p = _write(tmp_path, "x.md", "# A\nalpha text\n# B\nbeta text")
    pipe.ingest_file(str(p))
    delta = DeltaEngine(store).diff_versions("x.md")
    assert delta.from_version == 0
    assert {s.change for s in delta.sections} == {"added"}
    assert {s.section for s in delta.sections} == {"A", "B"}


def test_delta_no_versions_returns_empty(tmp_path):
    delta = DeltaEngine(FakeStore()).diff_versions("missing.md")
    assert delta.sections == [] and delta.to_version == 0


# --- Phase C: governance & conflict ------------------------------------------
def _chunk_v(cid, document_id, text="OTP", module="auth", category="Story", auth=0):
    c = Chunk(cid, text, document_id, "t", 0, 1, category, "ts", "h", "fake", "fake@dim8", {})
    c.document_id = document_id
    c.module = module
    c.authority_score = auth
    return c


def test_governance_conflict_lifecycle():
    gov = GovernanceStore(":memory:")
    rec = gov.create_conflict("p", "auth", "a1", "b1", "A", "B", 0.9)
    assert rec.status == "suspected"
    assert gov.exists_open_conflict("b1", "a1")                 # unordered pair
    assert {c.conflict_id for c in gov.get_conflicts(status="suspected")} == {rec.conflict_id}
    assert gov.open_conflict_chunk_ids() == {"a1", "b1"}

    gov.record_verdict(rec.conflict_id, True, "genuine contradiction")
    assert gov.get_conflict(rec.conflict_id).status == "confirmed"

    art = gov.create_resolution(rec.conflict_id, "new_rule",
                                "OTP only for external users", "alice", 100,
                                resolution_chunk_id="resolution:r1", resolution_id="r1")
    assert art.resolution_id == "r1"
    resolved = gov.get_conflict(rec.conflict_id)
    assert resolved.status == "resolved" and resolved.resolution_id == "r1"
    assert gov.open_conflict_chunk_ids() == set()               # no longer open


def test_governance_dismissed_can_be_reflagged():
    gov = GovernanceStore(":memory:")
    rec = gov.create_conflict("p", "m", "a1", "b1", "A", "B", 0.9)
    gov.record_verdict(rec.conflict_id, False)                  # dismissed
    assert not gov.exists_open_conflict("a1", "b1")


def test_conflict_detector_flags_cross_document_similarity():
    gov = GovernanceStore(":memory:")
    new = _chunk_v("a1", "A")
    hits = [
        SearchResult(_chunk_v("b1", "B", "OTP not required"), 0.90, "dense"),  # flag
        SearchResult(_chunk_v("c1", "C", "unrelated"), 0.20, "dense"),         # too low
        SearchResult(_chunk_v("a2", "A", "OTP again"), 0.95, "dense"),         # same doc
    ]

    class StubStore:
        def search(self, emb, top_k, where):
            return hits

    det = ConflictDetector(StubStore(), gov, threshold=0.83, project_id="p")
    created = det.scan([new], [[0.1] * 8])
    assert len(created) == 1 and created[0].chunk_b_id == "b1"
    assert det.scan([new], [[0.1] * 8]) == []                   # dedup on re-scan


def test_detector_skips_business_resolution():
    gov = GovernanceStore(":memory:")
    res = _chunk_v("r1", "resolution:r1", category="Business_Resolution", auth=100)

    class StubStore:
        def search(self, emb, top_k, where):
            raise AssertionError("resolutions must not be scanned")

    det = ConflictDetector(StubStore(), gov, threshold=0.83)
    assert det.scan([res], [[0.1] * 8]) == []


def test_apply_authority_prioritizes_resolution():
    low = SearchResult(_chunk_v("x", "X", auth=0), 0.50, "dense")
    res = SearchResult(_chunk_v("r", "resolution:r", category="Business_Resolution", auth=100), 0.40, "dense")
    out = RetrievalEngine._apply_authority([low, res])
    assert out[0].chunk.id == "r"        # 0.40*(1+1.0)=0.80 beats 0.50*(1+0)


def test_business_resolution_is_top_authority():
    assert authority_for("Business_Resolution") == 100
    assert authority_for("Business_Resolution") > authority_for("Story")


# --- Phase D: review UI & notifications --------------------------------------
class _StubChunkStore:
    def __init__(self, chunks):
        self._c = {c.id: c for c in chunks}

    def get_chunks_by_ids(self, ids):
        return [self._c[i] for i in ids if i in self._c]


def test_review_service_lists_and_details_and_actions():
    gov = GovernanceStore(":memory:")
    rec = gov.create_conflict("p", "auth", "a1", "b1", "design.md", "req.md", 0.91)
    store = _StubChunkStore([_chunk_v("a1", "design.md", "OTP required"),
                             _chunk_v("b1", "req.md", "OTP not required")])
    calls = []
    svc = ReviewService(gov, store,
                        lambda *a: calls.append(("resolve",) + a),
                        lambda *a: calls.append(("verdict",) + a))

    listing = svc.list_html()
    assert rec.conflict_id in listing and "design.md" in listing
    detail = svc.detail_html(rec.conflict_id)
    assert "OTP required" in detail and "OTP not required" in detail

    svc.verdict(rec.conflict_id, True, "why")
    svc.resolve(rec.conflict_id, "new_rule", "OTP for externals", "alice")
    assert ("verdict", rec.conflict_id, True, "why") in calls
    assert ("resolve", rec.conflict_id, "new_rule", "OTP for externals", "alice") in calls


def test_review_detail_missing_conflict():
    svc = ReviewService(GovernanceStore(":memory:"), _StubChunkStore([]),
                        lambda *a: None, lambda *a: None)
    assert "Not found" in svc.detail_html("nope")


def test_review_escapes_html():
    gov = GovernanceStore(":memory:")
    rec = gov.create_conflict("p", "m", "a1", "b1", "s", "t", 0.5)
    store = _StubChunkStore([_chunk_v("a1", "s", "<script>alert(1)</script>"),
                             _chunk_v("b1", "t", "y")])
    svc = ReviewService(gov, store, lambda *a: None, lambda *a: None)
    detail = svc.detail_html(rec.conflict_id)
    assert "<script>" not in detail and "&lt;script&gt;" in detail


def test_log_notifier_does_not_raise():
    rec = GovernanceStore(":memory:").create_conflict("p", "m", "a", "b", "s1", "s2", 0.9)
    LogNotifier().conflict_created(rec)   # placeholder must not throw


def test_detector_notifies_on_conflict():
    gov = GovernanceStore(":memory:")
    seen = []

    class RecordingNotifier(Notifier):
        def conflict_created(self, c):
            seen.append(c.conflict_id)

    hits = [SearchResult(_chunk_v("b1", "B", "OTP not required"), 0.9, "dense")]

    class StubStore:
        def search(self, emb, top_k, where):
            return hits

    det = ConflictDetector(StubStore(), gov, threshold=0.83, notifier=RecordingNotifier())
    created = det.scan([_chunk_v("a1", "A")], [[0.1] * 8])
    assert len(created) == 1 and seen == [created[0].conflict_id]


def test_review_http_round_trip():
    import urllib.parse
    import urllib.request
    from review_app import run_in_thread

    gov = GovernanceStore(":memory:")
    rec = gov.create_conflict("p", "m", "a1", "b1", "sa", "sb", 0.8)
    store = _StubChunkStore([_chunk_v("a1", "sa", "alpha text"),
                             _chunk_v("b1", "sb", "beta text")])
    actions = []
    svc = ReviewService(gov, store,
                        lambda *a: actions.append(("resolve",) + a),
                        lambda *a: actions.append(("verdict",) + a))

    httpd, _thread = run_in_thread(svc, host="127.0.0.1", port=0)  # ephemeral port
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert rec.conflict_id in urllib.request.urlopen(base + "/").read().decode()
        assert "alpha text" in urllib.request.urlopen(
            base + f"/conflict/{rec.conflict_id}").read().decode()
        data = urllib.parse.urlencode(
            {"conflict_id": rec.conflict_id, "is_conflict": "1", "rationale": "r"}).encode()
        urllib.request.urlopen(base + "/verdict", data=data)   # POST -> 303 -> GET detail
        assert ("verdict", rec.conflict_id, True, "r") in actions
    finally:
        httpd.shutdown()
