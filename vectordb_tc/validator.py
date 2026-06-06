"""Export validator - the piece that was missing from the design.

Runs BEFORE Excel is written. Hard failures return correctable errors so the host
LLM can retry (Req 6.2). Citation problems are soft warnings, not rejections
(Req 6.3 - the v3 judgement call). Produces the coverage matrix (Req 6.10 / 7)."""
from __future__ import annotations

import re

from models import (
    AcceptanceCriterion, TestCase, ValidationError, ValidationResult,
)

REQUIRED_NONEMPTY = ("name", "description", "precondition", "requirement_id")


def _name_pattern(issue_key: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(issue_key)}_TC-\d+\s*:\s*.+")


class ExportValidator:
    def validate(self, cases: list[TestCase], acceptance_criteria: list[AcceptanceCriterion],
                 issue_key: str, context_chunk_ids: set[str]) -> ValidationResult:
        errors: list[ValidationError] = []

        # coverage matrix: ac_id -> [tc names]
        matrix: dict[str, list[str]] = {ac.ac_id: [] for ac in acceptance_criteria}
        for tc in cases:
            for ac_id in tc.ac_ids:
                matrix.setdefault(ac_id, []).append(tc.name)

        # HARD: every AC covered
        for ac in acceptance_criteria:
            if not matrix.get(ac.ac_id):
                errors.append(ValidationError(
                    "uncovered_ac", ac.ac_id,
                    f"No test case maps to {ac.ac_id}. Add one and set its ac_ids."))

        # HARD: at least one negative case
        if not any(tc.case_type == "negative" for tc in cases):
            errors.append(ValidationError(
                "no_negative", "*", "At least one negative/error test case is required."))

        pat = _name_pattern(issue_key)
        for tc in cases:
            # HARD: naming convention
            if not pat.match(tc.name):
                errors.append(ValidationError(
                    "bad_name", tc.name,
                    f"Name must match '{issue_key}_TC-NN : <desc>'."))
            # HARD: required columns non-empty
            for fld in REQUIRED_NONEMPTY:
                if not str(getattr(tc, fld, "")).strip():
                    errors.append(ValidationError(
                        "missing_column", tc.name, f"Required field '{fld}' is empty."))
            if not tc.steps:
                errors.append(ValidationError("missing_column", tc.name, "No test steps."))

        # SOFT: citations must resolve to context shown this session -> warnings only
        warnings: list[dict] = []
        for tc in cases:
            for cid in tc.citations:
                if cid not in context_chunk_ids:
                    warnings.append({"tc_name": tc.name, "unresolved_id": cid})

        # low confidence: AC covered only by cases with no resolvable citation
        low_conf = []
        for ac in acceptance_criteria:
            covering = [tc for tc in cases if ac.ac_id in tc.ac_ids]
            if covering and all(
                not (set(tc.citations) & context_chunk_ids) for tc in covering
            ):
                low_conf.append(ac.ac_id)

        return ValidationResult(
            valid=not errors, errors=errors, coverage_matrix=matrix,
            low_confidence_acs=low_conf, citation_warnings=warnings,
        )
