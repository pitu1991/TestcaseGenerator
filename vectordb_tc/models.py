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
# Business_Resolution (Phase C) is the authoritative artifact a human approves to
# settle a conflict; it outranks every source category during retrieval.
# Failure (Phase E) is a diagnostic test-failure artifact (error + stack trace),
# stored for similarity-based recurring-failure analysis — deliberately low
# authority so it never grounds test-case generation.
Category = Literal[
    "MOM", "UI_Flow", "Error_Code", "Business_Rule", "Q_and_A", "Story", "Test_Case",
    "Business_Resolution", "Failure",
]
CATEGORIES: tuple[str, ...] = (
    "MOM", "UI_Flow", "Error_Code", "Business_Rule", "Q_and_A", "Story", "Test_Case",
    "Business_Resolution", "Failure",
)

CaseType = Literal["positive", "negative", "edge"]

# Default per-category authority used by governance ranking. Higher = more trusted.
# Stamped on every chunk so the metadata schema is stable.
AUTHORITY_DEFAULTS: dict[str, int] = {
    "Business_Resolution": 100,
    "Story": 80, "Business_Rule": 80,
    "UI_Flow": 70, "Error_Code": 70,
    "Test_Case": 60,
    "MOM": 40, "Q_and_A": 40,
    "Failure": 10,   # diagnostic only; must never outrank real knowledge
}


def authority_for(category: str) -> int:
    return AUTHORITY_DEFAULTS.get(category, 50)


@dataclass
class Chunk:
    id: str                    # "{content_hash[:12]}_{chunk_index}_v{version}"
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
    # --- versioning (Phase A) ------------------------------------------------
    document_id: str = ""      # stable logical identity across versions (the anchor)
    version: int = 1           # monotonic per document_id, starts at 1
    is_latest: bool = True     # default retrieval filter; old versions flip to False
    module: str = "default"    # coarse grouping (e.g. "Authentication")
    section: str = ""          # nearest heading the chunk came from (best-effort)
    chunk_hash: str = ""       # sha256 of THIS chunk's text (Phase B delta detection)
    superseded_at: str = ""    # ISO 8601 when this chunk lost is_latest (audit)
    authority_score: int = 0   # governance ranking weight (Phase C)


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
    resolution_ids: list[str] = field(default_factory=list)  # governance traceability (Phase C)
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
class DeltaSection:
    section: str               # heading the change belongs to
    change: str                # "added" | "removed" | "changed"
    old_text: str = ""
    new_text: str = ""


@dataclass
class DocumentDelta:
    document_id: str
    from_version: int          # 0 when there is no prior version (all added)
    to_version: int
    sections: list[DeltaSection]      # only the sections that differ
    unchanged_sections: list[str]


@dataclass
class ConflictRecord:
    conflict_id: str
    project_id: str
    module: str
    status: str                # suspected | confirmed | dismissed | resolved
    priority: str              # low | medium | high
    created_at: str            # ISO 8601
    chunk_a_id: str
    chunk_b_id: str
    source_a: str
    source_b: str
    similarity: float
    assigned_reviewer: str = ""
    rationale: str = ""        # filled by the host LLM's adjudication
    resolution_id: str = ""


@dataclass
class ResolutionArtifact:
    resolution_id: str
    conflict_id: str
    decision: str              # source_a | source_b | merge | new_rule
    text: str                  # the authoritative statement humans approved
    approver: str
    authority_level: int
    effective_date: str
    created_at: str
    resolution_chunk_id: str = ""   # document_id of the re-ingested resolution chunk


@dataclass
class FailureArtifact:
    """A stored test-failure record (Phase E). The embedding is built from the
    error signature (test_name + error_message + stack_trace); root_cause / fix /
    assignee are human annotations attached after triage, kept as metadata so the
    error signature stays the search key. Identical failures collapse onto one
    failure_id and bump `occurrences` rather than creating duplicates."""
    failure_id: str            # "fail_{sig_hash[:12]}" — stable across recurrences
    test_name: str
    error_message: str
    stack_trace: str = ""
    run_id: str = ""           # CI run / build id of the most recent occurrence
    status: str = "failed"     # failed | error | flaky
    screenshot_path: str = ""  # local path or object-store URL (blob lives elsewhere)
    project: str = "default"
    root_cause: str = ""       # filled after triage
    fix_commit: str = ""       # commit that resolved it
    assignee: str = ""
    occurrences: int = 1
    first_seen: str = ""       # ISO 8601
    last_seen: str = ""        # ISO 8601


@dataclass
class IngestionResult:
    success: bool
    source_path: str
    document_id: str = ""
    version: int = 0
    chunks_created: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    redactions: int = 0
    skipped_unchanged: bool = False
    model_changed: bool = False
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0
