"""CI/CD hook: feed test failures into the failure-intelligence memory, or query
it for similar past failures. Local-first — uses the bundled embedder + ChromaDB,
no cloud, no API keys.

JUnit XML is the lingua franca of test runners (pytest --junitxml, JUnit/Surefire,
Playwright/Cypress reporters, Gradle, Go gotestsum, ...), so this hook parses that
one format with the stdlib and works for almost any stack.

Usage
-----
Ingest the failures from a run (call this at the END of a CI job, even on failure):

    python examples/ingest_failures.py ingest --junit results.xml \
        --project web --run-id "$CI_BUILD_ID"

Query the memory for a new error (e.g. while triaging a fresh red build):

    python examples/ingest_failures.py query \
        --error "TimeoutError: locator.click on #checkout" --project web

Show the chronic offenders:

    python examples/ingest_failures.py stats --project web

Config: reads config.yaml if VECTORDB_USE_CONFIG=1, else uses VECTORDB_PROJECT_ROOT
(same convention as the MCP server). The store written here is the SAME ChromaDB
the server reads, so failures ingested in CI are queryable from the IDE.
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig
from embedder import EmbeddingService
from failure import FailureAnalyzer
from store import ChromaStore


def _load_config() -> AppConfig:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if os.environ.get("VECTORDB_USE_CONFIG") == "1" and cfg_path.exists():
        return AppConfig.from_file(str(cfg_path))
    return AppConfig()  # honors VECTORDB_PROJECT_ROOT


def _analyzer(cfg: AppConfig) -> FailureAnalyzer:
    embedder = EmbeddingService(cfg.embedding_model)
    store = ChromaStore(cfg.chromadb_path)
    return FailureAnalyzer(embedder, store, cfg)


def parse_junit(path: str) -> list[dict]:
    """Extract failing/erroring test cases from a JUnit XML report.

    Handles both <testsuites> roots and a bare <testsuite>. A <testcase> counts as
    a failure if it has a <failure> or <error> child; passed/skipped are ignored.
    """
    root = ET.parse(path).getroot()
    suites = root.iter("testsuite")
    failures: list[dict] = []
    for suite in suites:
        for case in suite.findall("testcase"):
            bad = case.find("failure")
            status = "failed"
            if bad is None:
                bad = case.find("error")
                status = "error"
            if bad is None:
                continue  # passed or skipped
            classname = case.get("classname", "")
            name = case.get("name", "")
            test_name = f"{classname}::{name}" if classname else name
            failures.append({
                "test_name": test_name,
                "error_message": (bad.get("message") or "").strip(),
                "stack_trace": (bad.text or "").strip(),
                "status": status,
            })
    return failures


def cmd_ingest(args, fa: FailureAnalyzer) -> int:
    failures = parse_junit(args.junit)
    if not failures:
        print(f"No failures found in {args.junit} (all green?).")
        return 0
    print(f"Ingesting {len(failures)} failure(s) from {args.junit}...")
    new, recurring = 0, 0
    for f in failures:
        art = fa.record_failure(
            test_name=f["test_name"], error_message=f["error_message"],
            stack_trace=f["stack_trace"], run_id=args.run_id, status=f["status"],
            project=args.project,
        )
        tag = "NEW" if art.occurrences == 1 else f"RECURRING x{art.occurrences}"
        if art.occurrences == 1:
            new += 1
        else:
            recurring += 1
        note = f"  (known root cause: {art.root_cause})" if art.root_cause else ""
        print(f"  [{tag:>14}] {art.test_name}{note}")
    print(f"\nDone: {new} new, {recurring} recurring.")
    if recurring:
        print("Recurring failures already have history — query them for root cause.")
    return 0


def cmd_query(args, fa: FailureAnalyzer) -> int:
    matches = fa.find_similar(args.error, top_k=args.top_k,
                              min_similarity=args.min_similarity, project=args.project)
    if not matches:
        print("No strong match in failure memory. Manual triage needed.")
        return 0
    print(f"Top {len(matches)} similar past failure(s):\n")
    for art, sim in matches:
        print(f"  similarity {sim:.3f}  [{art.status}]  x{art.occurrences}  {art.test_name}")
        print(f"    error: {art.error_message[:100]}")
        print(f"    root cause: {art.root_cause or '(untriaged)'}")
        if art.fix_commit:
            print(f"    fix commit: {art.fix_commit}")
        if art.assignee:
            print(f"    last owner: {art.assignee}")
        print()
    return 0


def cmd_stats(args, fa: FailureAnalyzer) -> int:
    s = fa.stats(project=args.project)
    print(f"Distinct failures: {s['total_failures']}   "
          f"Total occurrences: {s['total_occurrences']}")
    if s["top_recurring"]:
        print("\nTop recurring:")
        for t in s["top_recurring"]:
            print(f"  x{t['occurrences']:>3}  {t['test_name']}  -> {t['root_cause']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Failure-intelligence CI hook (local-first).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest failures from a JUnit XML report.")
    p_ing.add_argument("--junit", required=True, help="Path to JUnit XML report.")
    p_ing.add_argument("--project", default="default")
    p_ing.add_argument("--run-id", default="", dest="run_id")

    p_q = sub.add_parser("query", help="Find similar past failures for an error.")
    p_q.add_argument("--error", required=True, help="New error message / signature.")
    p_q.add_argument("--project", default=None)
    p_q.add_argument("--top-k", type=int, default=3, dest="top_k")
    p_q.add_argument("--min-similarity", type=float, default=None, dest="min_similarity",
                     help="Override the config similarity cutoff (default 0.85).")

    p_s = sub.add_parser("stats", help="Show chronic recurring failures.")
    p_s.add_argument("--project", default=None)

    args = ap.parse_args()
    fa = _analyzer(_load_config())
    return {"ingest": cmd_ingest, "query": cmd_query, "stats": cmd_stats}[args.cmd](args, fa)


if __name__ == "__main__":
    raise SystemExit(main())
