"""Retrieval engine.

Fix: 'hybrid' now actually fuses dense vector hits with keyword (literal
substring) hits via Reciprocal Rank Fusion, instead of relying on metadata
filtering. This is what makes identifiers like error codes and issue keys
retrievable. gather_context enumerates ACs with stable IDs and assembles a
token-budgeted ContextBundle."""
from __future__ import annotations

import re

from config import AppConfig
from embedder import EmbeddingService
from models import AcceptanceCriterion, ContextBundle, SearchResult
from store import ChromaStore

# Domain identifiers: error codes (PY075), issue keys (TWCJ-6184), field-ish tokens.
_IDENT = re.compile(r"\b([A-Z]{2,}[-_]?\d{2,}|[A-Z]{2,}\d+|[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")


class EmptyKnowledgeBase(Exception):
    """Raised when retrieval is attempted before anything is ingested."""


class RetrievalEngine:
    def __init__(self, embedder: EmbeddingService, store: ChromaStore, config: AppConfig):
        self.embedder, self.store, self.config = embedder, store, config

    def search(self, query: str, filters: dict | None = None, top_k: int | None = None,
               include_historical: bool = False, authority_boost: bool = False) -> list[SearchResult]:
        top_k = top_k or self.config.default_top_k
        if self.store.get_stats()["total_chunks"] == 0:
            raise EmptyKnowledgeBase("knowledge base empty - run ingestion first")

        where = self._with_latest(filters, include_historical)
        dense = self.store.search(self.embedder.embed_query(query), top_k=top_k * 2, where=where)
        keyword: list[SearchResult] = []
        for term in self._extract_keywords(query):
            keyword.extend(self.store.keyword_search(term, top_k=top_k, where=where))
        fused = self._rrf([dense, keyword])
        if authority_boost:
            fused = self._apply_authority(fused)
        return fused[:top_k]

    @staticmethod
    def _apply_authority(results: list[SearchResult]) -> list[SearchResult]:
        """Re-rank by source authority (Phase C governance): a Business_Resolution
        (score 100) outranks a Story (80) outranks an MOM (40) at equal relevance.
        Multiplies fused score by (1 + authority/100) and re-sorts."""
        for r in results:
            a = getattr(r.chunk, "authority_score", 0) or 0
            r.relevance_score = round(r.relevance_score * (1 + a / 100.0), 6)
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)

    @staticmethod
    def _with_latest(filters: dict | None, include_historical: bool) -> dict | None:
        """Inject is_latest=True by default. Historical search is opt-in. ChromaDB
        requires $and to combine conditions; we flatten an existing $and rather than
        nesting (ChromaDB does not accept nested $and)."""
        if include_historical:
            return filters or None
        latest = {"is_latest": True}
        if not filters:
            return latest
        if "$and" in filters and isinstance(filters["$and"], list):
            return {"$and": filters["$and"] + [latest]}
        return {"$and": [filters, latest]}

    def relevant_test_cases(self, query: str, issue_key: str | None = None,
                            top_k: int = 8) -> list[SearchResult]:
        """Existing Test_Case chunks most relevant to a query (used for delta
        regeneration: what already exists that the change might affect)."""
        if issue_key:
            filters = {"$and": [{"category": "Test_Case"}, {"issue_key": issue_key}]}
        else:
            filters = {"category": "Test_Case"}
        return self.search_filtered(query, filters, top_k)

    def gather_context(self, description: str, acceptance_criteria, issue_key=None,
                       token_budget: int | None = None, linked_issues=None) -> ContextBundle:
        budget = token_budget or self.config.token_budget
        if self.store.get_stats()["total_chunks"] == 0:
            raise EmptyKnowledgeBase("knowledge base empty - run ingestion first")

        acs = self._enumerate_acs(acceptance_criteria)
        q = description + "\n" + "\n".join(a.text for a in acs)

        def grab(cat: str, k: int) -> list[SearchResult]:
            return self._cap(self.search_filtered(q, {"category": cat}, k), budget // 5)

        bundle = ContextBundle(
            story_description=description,
            acceptance_criteria=acs,
            issue_key=issue_key,
            linked_issues=linked_issues or [],
            similar_stories=self._cap(self.search_filtered(q, {"category": "Test_Case"}, 4), budget // 3),
            ui_flows=grab("UI_Flow", 4),
            error_codes=grab("Error_Code", 6),
            business_rules=grab("Business_Rule", 4),
            meeting_notes=grab("MOM", 3),
            template_spec={},          # filled by server from ExcelExporter constants
            instruction_block="",      # filled by server
            total_tokens=0,
            token_budget=budget,
        )
        bundle.total_tokens = self._count_bundle_tokens(bundle)
        return bundle

    def search_filtered(self, query, filters, top_k):
        try:
            return self.search(query, filters=filters, top_k=top_k)
        except EmptyKnowledgeBase:
            return []

    # --- internals -----------------------------------------------------------
    def _enumerate_acs(self, ac) -> list[AcceptanceCriterion]:
        if isinstance(ac, list):
            items = [str(x) for x in ac]
        else:
            # split a blob on numbered/bulleted lines or newlines
            items = [s.strip(" -*\t") for s in re.split(r"\n+|\r+", str(ac)) if s.strip()]
        return [AcceptanceCriterion(f"AC-{i+1}", t) for i, t in enumerate(items)]

    def _extract_keywords(self, text: str) -> list[str]:
        return list(dict.fromkeys(_IDENT.findall(text)))[:8]

    @staticmethod
    def _rrf(ranked_lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
        scores: dict[str, float] = {}
        best: dict[str, SearchResult] = {}
        for results in ranked_lists:
            for rank, r in enumerate(results):
                scores[r.chunk.id] = scores.get(r.chunk.id, 0.0) + 1.0 / (k + rank + 1)
                if r.chunk.id not in best or r.match_type == "keyword":
                    best[r.chunk.id] = r
        out = []
        for cid in sorted(scores, key=scores.get, reverse=True):
            r = best[cid]
            out.append(SearchResult(r.chunk, round(scores[cid], 5),
                                    "hybrid" if scores[cid] > 1.0 / (k + 1) else r.match_type))
        return out

    def _cap(self, results: list[SearchResult], token_cap: int) -> list[SearchResult]:
        out, used = [], 0
        for r in results:
            t = self.embedder.count_tokens(r.chunk.text)
            if used + t > token_cap:
                break
            out.append(r); used += t
        return out

    def _count_bundle_tokens(self, b: ContextBundle) -> int:
        groups = [b.similar_stories, b.ui_flows, b.error_codes, b.business_rules, b.meeting_notes]
        return sum(self.embedder.count_tokens(r.chunk.text) for g in groups for r in g)
