"""Unit + property tests for the deterministic, dependency-light logic.

These run WITHOUT chromadb or sentence-transformers (those are imported lazily
inside __init__, not at module import time), using a FakeEmbedder. Integration
tests that need a real store/model live separately and are skipped if deps are
absent."""
from __future__ import annotations

import pytest

from models import (
    AcceptanceCriterion, Chunk, SearchResult, TestCase, TestStep,
)
from validator import ExportValidator
from chunker import Chunker
from retrieval import RetrievalEngine
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
