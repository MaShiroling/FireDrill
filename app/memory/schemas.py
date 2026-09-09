"""Serializable domain models for verified, cross-thread operational memories."""

from __future__ import annotations

import json
import unicodedata
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from app.memory.fingerprint import build_memory_fingerprint

MEMORY_SCHEMA_VERSION = "1.0"


class MemoryType(StrEnum):
    """Memory categories supported by the first implementation."""

    FAILURE_LESSON = "failure_lesson"
    SUCCESSFUL_CASE = "successful_case"


class MemoryStatus(StrEnum):
    """Review lifecycle for a memory."""

    CANDIDATE = "candidate"
    APPROVED = "approved"
    RETIRED = "retired"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _parse_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _namespace_part(value: str, field_name: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = "-".join(normalized.split())
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _required_text(value: str, field_name: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _ordered_unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


@dataclass(frozen=True)
class MemoryNamespace:
    """Isolation boundary for memories shared across conversation threads."""

    tenant_id: str
    device_type: str
    memory_type: MemoryType

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _namespace_part(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "device_type", _namespace_part(self.device_type, "device_type"))
        if not isinstance(self.memory_type, MemoryType):
            object.__setattr__(self, "memory_type", MemoryType(self.memory_type))

    def as_tuple(self) -> tuple[str, str, str]:
        return self.tenant_id, self.device_type, self.memory_type.value

    def to_dict(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "device_type": self.device_type,
            "memory_type": self.memory_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryNamespace:
        return cls(
            tenant_id=str(data["tenant_id"]),
            device_type=str(data["device_type"]),
            memory_type=MemoryType(str(data["memory_type"])),
        )


@dataclass(frozen=True)
class MemoryRecord:
    """A reviewable operational memory with evidence and lifecycle metadata."""

    memory_id: str
    namespace: MemoryNamespace
    scenario: str
    lesson: str
    fingerprint: str
    schema_version: str = MEMORY_SCHEMA_VERSION
    status: MemoryStatus = MemoryStatus.CANDIDATE
    preconditions: dict[str, Any] = field(default_factory=dict)
    recommended_actions: tuple[str, ...] = ()
    failure_codes: tuple[str, ...] = ()
    evidence_run_ids: tuple[str, ...] = ()
    evidence_case_ids: tuple[str, ...] = ()
    source_revision: str = "unknown"
    confidence: float = 0.5
    success_count: int = 0
    contradiction_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    expires_at: str | None = None
    embedding_model: str = "text-embedding-v4"
    embedding_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _required_text(self.memory_id, "memory_id"))
        object.__setattr__(self, "scenario", _required_text(self.scenario, "scenario"))
        object.__setattr__(self, "lesson", _required_text(self.lesson, "lesson"))
        object.__setattr__(self, "fingerprint", _required_text(self.fingerprint, "fingerprint"))
        object.__setattr__(self, "source_revision", self.source_revision.strip() or "unknown")
        object.__setattr__(self, "embedding_model", self.embedding_model.strip())
        object.__setattr__(self, "embedding_version", self.embedding_version.strip())
        if not isinstance(self.namespace, MemoryNamespace):
            raise TypeError("namespace must be a MemoryNamespace")
        if not isinstance(self.status, MemoryStatus):
            object.__setattr__(self, "status", MemoryStatus(self.status))
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.success_count < 0 or self.contradiction_count < 0:
            raise ValueError("memory counters must not be negative")
        _parse_timestamp(self.created_at, "created_at")
        _parse_timestamp(self.updated_at, "updated_at")
        if self.expires_at is not None:
            _parse_timestamp(self.expires_at, "expires_at")

        try:
            clean_preconditions = json.loads(
                json.dumps(self.preconditions, ensure_ascii=False, sort_keys=True)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("preconditions must be JSON serializable") from exc
        if not isinstance(clean_preconditions, dict):
            raise ValueError("preconditions must be a JSON object")
        object.__setattr__(self, "preconditions", clean_preconditions)
        object.__setattr__(self, "recommended_actions", _ordered_unique(self.recommended_actions))
        object.__setattr__(
            self, "failure_codes", tuple(sorted(_ordered_unique(self.failure_codes)))
        )
        object.__setattr__(self, "evidence_run_ids", _ordered_unique(self.evidence_run_ids))
        object.__setattr__(self, "evidence_case_ids", _ordered_unique(self.evidence_case_ids))
        if (
            self.status is MemoryStatus.APPROVED
            and not self.evidence_run_ids
            and not self.evidence_case_ids
        ):
            raise ValueError("approved memory must include at least one evidence ID")

    @property
    def memory_type(self) -> MemoryType:
        return self.namespace.memory_type

    @classmethod
    def create(
        cls,
        *,
        namespace: MemoryNamespace,
        scenario: str,
        lesson: str,
        preconditions: dict[str, Any] | None = None,
        recommended_actions: tuple[str, ...] | list[str] = (),
        failure_codes: tuple[str, ...] | list[str] = (),
        evidence_run_ids: tuple[str, ...] | list[str] = (),
        evidence_case_ids: tuple[str, ...] | list[str] = (),
        source_revision: str = "unknown",
        confidence: float = 0.5,
        success_count: int = 0,
        contradiction_count: int = 0,
        status: MemoryStatus = MemoryStatus.CANDIDATE,
        expires_at: str | None = None,
        embedding_model: str = "text-embedding-v4",
        embedding_version: str = "1",
    ) -> MemoryRecord:
        actions = _ordered_unique(recommended_actions)
        codes = tuple(sorted(_ordered_unique(failure_codes)))
        conditions = dict(preconditions or {})
        now = _now_iso()
        return cls(
            memory_id=f"mem-{uuid.uuid4().hex}",
            namespace=namespace,
            scenario=scenario,
            lesson=lesson,
            fingerprint=build_memory_fingerprint(
                namespace=namespace.as_tuple(),
                scenario=scenario,
                lesson=lesson,
                preconditions=conditions,
                recommended_actions=actions,
                failure_codes=codes,
            ),
            status=status,
            preconditions=conditions,
            recommended_actions=actions,
            failure_codes=codes,
            evidence_run_ids=_ordered_unique(evidence_run_ids),
            evidence_case_ids=_ordered_unique(evidence_case_ids),
            source_revision=source_revision,
            confidence=confidence,
            success_count=success_count,
            contradiction_count=contradiction_count,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
        )

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("now must include a timezone")
        return _parse_timestamp(self.expires_at, "expires_at") <= current

    def is_searchable(self, now: datetime | None = None) -> bool:
        return self.status is MemoryStatus.APPROVED and not self.is_expired(now)

    def with_status(self, status: MemoryStatus, *, updated_at: str | None = None) -> MemoryRecord:
        allowed = {
            MemoryStatus.CANDIDATE: {MemoryStatus.APPROVED, MemoryStatus.RETIRED},
            MemoryStatus.APPROVED: {MemoryStatus.RETIRED},
            MemoryStatus.RETIRED: set(),
        }
        target = MemoryStatus(status)
        if target is self.status:
            return self
        if target not in allowed[self.status]:
            raise ValueError(f"invalid memory status transition: {self.status} -> {target}")
        return replace(self, status=target, updated_at=updated_at or _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "namespace": self.namespace.to_dict(),
            "memory_type": self.memory_type.value,
            "scenario": self.scenario,
            "lesson": self.lesson,
            "preconditions": json.loads(json.dumps(self.preconditions, ensure_ascii=False)),
            "recommended_actions": list(self.recommended_actions),
            "failure_codes": list(self.failure_codes),
            "evidence_run_ids": list(self.evidence_run_ids),
            "evidence_case_ids": list(self.evidence_case_ids),
            "fingerprint": self.fingerprint,
            "source_revision": self.source_revision,
            "confidence": self.confidence,
            "status": self.status.value,
            "success_count": self.success_count,
            "contradiction_count": self.contradiction_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        namespace = MemoryNamespace.from_dict(dict(data["namespace"]))
        top_level_type = data.get("memory_type")
        if (
            top_level_type is not None
            and MemoryType(str(top_level_type)) is not namespace.memory_type
        ):
            raise ValueError("memory_type must match namespace.memory_type")
        return cls(
            schema_version=str(data.get("schema_version", MEMORY_SCHEMA_VERSION)),
            memory_id=str(data["memory_id"]),
            namespace=namespace,
            scenario=str(data["scenario"]),
            lesson=str(data["lesson"]),
            preconditions=dict(data.get("preconditions") or {}),
            recommended_actions=tuple(data.get("recommended_actions") or ()),
            failure_codes=tuple(data.get("failure_codes") or ()),
            evidence_run_ids=tuple(data.get("evidence_run_ids") or ()),
            evidence_case_ids=tuple(data.get("evidence_case_ids") or ()),
            fingerprint=str(data["fingerprint"]),
            source_revision=str(data.get("source_revision", "unknown")),
            confidence=float(data.get("confidence", 0.5)),
            status=MemoryStatus(str(data.get("status", MemoryStatus.CANDIDATE.value))),
            success_count=int(data.get("success_count", 0)),
            contradiction_count=int(data.get("contradiction_count", 0)),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
            expires_at=str(data["expires_at"]) if data.get("expires_at") is not None else None,
            embedding_model=str(data.get("embedding_model", "text-embedding-v4")),
            embedding_version=str(data.get("embedding_version", "1")),
        )
