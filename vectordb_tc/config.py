"""Configuration. Paths derive from a configurable project_root rather than being
hardcoded to a specific user's Desktop (Req 10.2). Defaults can still resolve to the
current setup via the project_root, but nothing is pinned to "snaraya4"."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_root() -> str:
    # Order: env var -> ./Project under cwd. Override in the config file.
    return os.environ.get("VECTORDB_PROJECT_ROOT", str(Path.cwd() / "Project"))


@dataclass
class AppConfig:
    project_root: str = field(default_factory=_default_root)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # 400, NOT 750. bge-small-en-v1.5 truncates at 512 tokens; 750 silently loses
    # the back third of every long chunk. The chunker also clamps to the model max.
    chunk_size: int = 400
    chunk_overlap: int = 80

    default_top_k: int = 10
    token_budget: int = 35000          # host-LLM context budget (approx tokens)
    rerank: bool = False               # Phase 1 default off (Req 4.2 / 10.1)

    pii_guard: bool = True             # lightweight SSN/bank guard, default on (Req 12.2)

    log_level: str = "INFO"
    log_file: str = "vectordb-server.log"

    # Derived paths (relative to project_root) -------------------------------
    @property
    def chromadb_path(self) -> str:
        return str(Path(self.project_root) / "vectordb-data")

    @property
    def output_dir(self) -> str:
        return str(Path(self.project_root) / "User Stories")

    @property
    def knowledge_dir(self) -> str:
        return str(Path(self.project_root) / "KnowledgeBase")

    @property
    def jira_server_path(self) -> str:
        return str(Path(self.project_root) / "jira-mcp" / "server.py")

    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        raw = Path(path).read_text(encoding="utf-8")
        if path.endswith((".yaml", ".yml")):
            import yaml  # lazy import so JSON-only setups don't need pyyaml
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
