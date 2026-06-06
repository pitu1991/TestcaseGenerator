"""Jira integration client.

Kiro's note: we already have a working Jira MCP server, so instead of speaking
MCP-over-the-wire we import that server module directly and call its functions.
Two things you must map to YOUR server's real function names (marked TODO below);
everything else here is complete, including the ADF -> text flattener (Req 3.2),
which is the part with real hidden complexity (nested tables, lists, panels)."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Jira server not found at {path}")
    spec = importlib.util.spec_from_file_location("jira_mcp_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __main__-guarded mcp.run() won't fire on import
    return mod


class JiraClient:
    def __init__(self, jira_server_path: str):
        # Raising here lets server.py return a clean 'jira_unavailable' (Req 3.7).
        self._mod = _load_module(jira_server_path)

    def get_issue(self, issue_key: str) -> dict:
        # TODO: map to your server's function, e.g. self._mod.get_issue(issue_key)
        raw = self._mod.get_issue(issue_key)  # adjust name if different
        return self._normalize_issue(raw)

    def get_epic_stories(self, epic_key: str) -> list[dict]:
        # TODO: map to your server's function, e.g. self._mod.get_epic_issues(epic_key)
        raws = self._mod.get_epic_issues(epic_key)  # adjust name if different
        return [self._normalize_issue(r) for r in raws]

    def get_all_epics_summary(self) -> list[dict]:
        return self._mod.get_all_epics()  # adjust name if different

    # --- normalization -------------------------------------------------------
    def _normalize_issue(self, raw: dict) -> dict:
        """Flatten a Jira issue into the fields ingestion expects, converting any
        ADF bodies to clean text."""
        fields = raw.get("fields", raw)
        return {
            "issue_key": raw.get("key") or raw.get("issue_key", ""),
            "summary": fields.get("summary", ""),
            "description": adf_to_text(fields.get("description")),
            "acceptance_criteria": adf_to_text(fields.get("acceptance_criteria")),
            "definition_of_done": adf_to_text(fields.get("definition_of_done")),
            "comments": [adf_to_text(c.get("body")) for c in
                         (fields.get("comment", {}).get("comments", []) or [])],
            "epic_key": fields.get("epic_key") or fields.get("parent", {}).get("key", ""),
            "assignee": (fields.get("assignee") or {}).get("displayName", ""),
            "status": (fields.get("status") or {}).get("name", ""),
            "story_points": fields.get("story_points") or fields.get("customfield_10016"),
        }


# --- ADF (Atlassian Document Format) -> plain text ---------------------------
def adf_to_text(node) -> str:
    """Flatten ADF JSON to readable text. Accepts a dict, a list, plain str, or None.
    Preserves headings (#), bullet/ordered lists, code blocks (```), and tables
    rendered as ' | '-joined rows so identifier/meaning pairs stay together."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(p for p in (adf_to_text(n) for n in node) if p)

    ntype = node.get("type", "")
    content = node.get("content", [])

    if ntype == "text":
        return node.get("text", "")
    if ntype == "hardBreak":
        return "\n"
    if ntype in ("mention", "emoji"):
        attrs = node.get("attrs", {})
        return attrs.get("text") or attrs.get("shortName") or ""
    if ntype == "paragraph":
        return "".join(adf_to_text(c) for c in content)
    if ntype == "heading":
        level = node.get("attrs", {}).get("level", 1)
        return "#" * int(level) + " " + "".join(adf_to_text(c) for c in content)
    if ntype == "codeBlock":
        return "```\n" + "".join(adf_to_text(c) for c in content) + "\n```"
    if ntype in ("blockquote", "panel"):
        return "\n".join(adf_to_text(c) for c in content)
    if ntype == "bulletList":
        return "\n".join("- " + adf_to_text(li) for li in content)
    if ntype == "orderedList":
        return "\n".join(f"{i}. " + adf_to_text(li)
                         for i, li in enumerate(content, 1))
    if ntype == "listItem":
        return " ".join(adf_to_text(c) for c in content).strip()
    if ntype in ("table",):
        return "\n".join(adf_to_text(row) for row in content)
    if ntype in ("tableRow",):
        cells = [adf_to_text(c).replace("\n", " ").strip() for c in content]
        return "| " + " | ".join(cells) + " |"
    if ntype in ("tableCell", "tableHeader"):
        return " ".join(adf_to_text(c) for c in content).strip()
    if ntype in ("doc", "rule"):
        return "\n".join(adf_to_text(c) for c in content)
    # Fallback: recurse into unknown containers so no text is silently dropped.
    return "".join(adf_to_text(c) for c in content) if content else ""
