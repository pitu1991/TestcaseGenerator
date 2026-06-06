"""Data models / contracts shared across the VectorDB TC server.

Changes vs the original design doc:
- Chunk: added content_hash, embedding_model, model_version; Test_Case is now a
  valid category (it was missing, which broke the few-shot foundation).
- ContextBundle.acceptance_criteria is now an enumerated list with stable AC IDs
  (was a bare str, which made per-AC coverage impossible).
- TestCase: added case_type, ac_ids (traceability), and citations (grounding).
- Added ValidationError / ValidationResult for the restored Export_Validator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Single source of truth for categories. Test_Case was missing in the design.
Category = Literal[
    "MOM", "UI_Flow", "Error_Code", "Business_Rule", "Q_and_A", "Story", "Test_Case"
]
CATEGORIES: tuple[str, ...] = (
    "MOM", "UI_Flow", "Error_Code", "Business_Rule", "Q_and_A", "Story", "Test_Case",
)

CaseType = Literal["positive", "negative", "edge"]


@dataclass
class Chunk:
    id: str                    # "{content_hash[:12]}_{chunk_index}"
    text: str
    source_path: str           # file path or "jira:{issue_key}"
    document_title: str
    chunk_index: int
    total_chunks: int
    category: str              # one of CATEGORIES
    ingestion_timestamp: str   # ISO 8601
    content_hash: str          # sha256 of the SOURCE file/content (change detection)
    embedding_model: str       # e.g. "BAAI/bge-small-en-v1.5"
    model_version: str         # model revision/dim signature; forces re-index on change
    metadata: dict = field(default_factory=dict)  # epic_key, issue_key, assignee, ...


@dataclass
class SearchResult:
    chunk: Chunk
    relevance_score: float     # 0..1, higher = more relevant (fused score)
    match_type: Literal["dense", "keyword", "hybrid"]


@dataclass
class AcceptanceCriterion:
    ac_id: str                 # stable within a request, e.g. "AC-1"
    text: str


@dataclass
class ContextBundle:
    story_description: str
    acceptance_criteria: list[AcceptanceCriterion]
    issue_key: str | None
    linked_issues: list[dict]
    similar_stories: list[SearchResult]   # past stories WITH their test cases
    ui_flows: list[SearchResult]
    error_codes: list[SearchResult]
    business_rules: list[SearchResult]
    meeting_notes: list[SearchResult]
    template_spec: dict                   # headers, defaults, naming convention
    instruction_block: str                # how the host LLM should structure output
    total_tokens: int
    token_budget: int


@dataclass
class TestStep:
    step_number: str
    description: str
    expected_result: str


@dataclass
class TestCase:
    name: str                  # "{ISSUE_KEY}_TC-{NN} : {short description}"
    description: str
    precondition: str
    steps: list[TestStep]
    assigned_to: str
    requirement_id: str        # team-template column (issue key)
    ac_ids: list[str]          # which acceptance criteria this case covers (traceability)
    case_type: CaseType        # positive | negative | edge  (reliable coverage check)
    citations: list[str]       # source chunk ids that grounded this case
    workplace_capability: str = ""
    priority: str = ""
    status: str = "Not Run"
    type: str = "Manual"
    application: str = "CORE"


@dataclass
class ValidationError:
    type: str                  # "uncovered_ac" | "no_negative" | "bad_name" | "missing_column"
    target: str                # ac_id or tc name
    message: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError]                 # hard failures -> host must retry
    coverage_matrix: dict[str, list[str]]         # ac_id -> [tc names]
    low_confidence_acs: list[str]
    citation_warnings: list[dict]                 # [{tc_name, unresolved_id}]  (soft)


@dataclass
class IngestionResult:
    success: bool
    source_path: str
    chunks_created: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    redactions: int = 0
    skipped_unchanged: bool = False
    model_changed: bool = False
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0
