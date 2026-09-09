"""Serializable schemas used by the run trace recorder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    sequence: int
    timestamp: str
    event_type: str
    node: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTrace:
    schema_version: str
    run_id: str
    session_id: str
    task: str
    started_at: str
    status: str
    version_manifest: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    ended_at: str | None = None
    duration_s: float | None = None
    final_state: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
