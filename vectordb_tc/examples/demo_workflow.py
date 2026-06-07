"""End-to-end walkthrough of the platform (Phases A-D), runnable locally.

    python examples/demo_workflow.py

Uses the REAL embedder + ChromaDB + SQLite in a throwaway temp directory, so it
demonstrates the actual pipeline (not fakes): versioning, delta detection,
conflict detection, human resolution, authority-ranked retrieval, and the review
UI render. Requires chromadb + sentence-transformers (first run downloads the
~130 MB model, then cached)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunker import Chunker
from conflict import ConflictDetector
from config import AppConfig
from delta import DeltaEngine
from embedder import EmbeddingService
from governance import GovernanceStore
from ingestion import IngestionPipeline
from notifier import LogNotifier
from retrieval import RetrievalEngine
from review_app import ReviewService
from store import ChromaStore


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="tcgen_demo_"))
    kb = root / "KnowledgeBase"
    kb.mkdir(parents=True)

    cfg = AppConfig(project_root=str(root), chunk_size=200, chunk_overlap=40,
                    pii_guard=False, conflict_detection=True,
                    conflict_similarity_threshold=0.7)
    embedder = EmbeddingService(cfg.embedding_model)
    store = ChromaStore(cfg.chromadb_path)
    gov = GovernanceStore(cfg.governance_db_path)
    detector = ConflictDetector(store, gov, cfg.conflict_similarity_threshold,
                                cfg.project_id, notifier=LogNotifier())
    pipe = IngestionPipeline(cfg, Chunker(embedder, cfg.chunk_size, cfg.chunk_overlap),
                             embedder, store, conflict_detector=detector)
    retr = RetrievalEngine(embedder, store, cfg)
    delta = DeltaEngine(store)

    # ---- Phase A: versioning ------------------------------------------------
    banner("PHASE A  Versioning (Option B: keep history, is_latest)")
    design = kb / "design.md"
    design.write_text("# Login Flow\n1. Username/Password\n2. OTP Validation\n"
                      "3. Redirect to Dashboard", encoding="utf-8")
    r1 = pipe.ingest_file(str(design))
    print(f"Ingested design.md  -> version {r1.version}  (document_id={r1.document_id})")

    design.write_text("# Login Flow\n1. Username/Password\n2. CAPTCHA Validation\n"
                      "3. OTP Validation\n4. Redirect to Dashboard", encoding="utf-8")
    r2 = pipe.ingest_file(str(design))
    print(f"Edited  design.md  -> version {r2.version}  (v1 retained, flipped is_latest=false)")
    print(f"Versions stored for 'design.md': {store.versions_for_document('design.md')}")

    # ---- Phase B: delta -----------------------------------------------------
    banner("PHASE B  Delta detection (what changed between versions)")
    d = delta.diff_versions("design.md")
    print(f"from v{d.from_version} -> v{d.to_version}")
    for s in d.sections:
        print(f"  [{s.change.upper()}] section '{s.section}'")
        print(f"     was: {s.old_text!r}")
        print(f"     now: {s.new_text!r}")
    print("-> only impacted test cases would be regenerated (gather_delta_context).")

    # ---- Phase C: conflict detection ---------------------------------------
    banner("PHASE C  Conflict detection across independent sources")
    (kb / "tech_design.md").write_text(
        "# OTP\nOTP validation is required for login.", encoding="utf-8")
    (kb / "business_req.md").write_text(
        "# OTP\nOTP validation is not required for login.", encoding="utf-8")
    pipe.ingest_file(str(kb / "tech_design.md"))
    pipe.ingest_file(str(kb / "business_req.md"))

    suspected = gov.get_conflicts(status="suspected")
    print(f"Suspected conflicts flagged by similarity: {len(suspected)}")
    conflict = max(suspected, key=lambda c: c.similarity)
    print(f"  conflict {conflict.conflict_id}: {conflict.source_a}  <->  {conflict.source_b}"
          f"  (similarity={conflict.similarity:.3f})")
    print("  NOTE: detection is deterministic similarity only - NO LLM in the server.")
    print("        The host IDE LLM adjudicates via get_conflict_candidates/record_verdict.")

    # ---- Phase C: adjudication + resolution --------------------------------
    banner("PHASE C  Human-in-the-loop resolution -> authoritative knowledge")
    gov.record_verdict(conflict.conflict_id, True, "required vs not required")
    print(f"LLM verdict recorded -> status '{gov.get_conflict(conflict.conflict_id).status}'")

    rid = "demo_res_1"
    res_text = "For login, OTP validation is required only for external users."
    ing = pipe.ingest_resolution(res_text, rid, module=conflict.module,
                                 extra={"conflict_id": conflict.conflict_id})
    gov.create_resolution(conflict.conflict_id, "new_rule", res_text, "reviewer@demo",
                          100, resolution_chunk_id=ing.document_id, resolution_id=rid)
    print(f"Human-approved resolution re-ingested as Business_Resolution (authority 100):")
    print(f"  \"{res_text}\"")
    print(f"Conflict status now -> '{gov.get_conflict(conflict.conflict_id).status}'")

    # ---- Phase C: authority-ranked retrieval -------------------------------
    banner("PHASE C  Retrieval: authoritative resolution outranks raw sources")
    q = "is OTP validation required for login"
    plain = retr.search(q, top_k=3)
    boosted = retr.search(q, top_k=3, authority_boost=True)
    print(f"Query: {q!r}\n")
    print("Without authority boost (top 3):")
    for r in plain:
        print(f"  [{r.chunk.category:>20}] {r.chunk.text[:55]!r}")
    print("\nWith authority boost (top 3)  <-- resolution wins:")
    for r in boosted:
        print(f"  [{r.chunk.category:>20}] {r.chunk.text[:55]!r}")

    # ---- Phase D: review UI render -----------------------------------------
    banner("PHASE D  Review UI (localhost) - rendered conflict list")
    svc = ReviewService(gov, store, lambda *a: None, lambda *a: None)
    listing = svc.list_html()
    row_lines = [ln.strip() for ln in listing.splitlines() if "<td>" in ln or "<tr>" in ln]
    print("ReviewService.list_html() produces a table of conflicts, e.g.:")
    print(f"  conflicts in DB: {len(gov.get_conflicts())}")
    print("  (launch live with the start_review_ui MCP tool -> http://127.0.0.1:8765/)")

    print(f"\nDemo store + db left in: {root}\n(delete it when done)")


if __name__ == "__main__":
    main()
