"""SQLite source-of-truth repository for long-term operational memories.

The repository deliberately stores review state and evidence separately from the
future vector index.  Milvus can therefore be rebuilt without losing the audit
trail that decides whether a memory is safe to expose to the agent.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.memory.fingerprint import build_memory_fingerprint
from app.memory.schemas import MemoryNamespace, MemoryRecord, MemoryStatus

MEMORY_DB_SCHEMA_VERSION = 2


class MemoryConflictError(RuntimeError):
    """A persisted identity conflicts with the supplied memory."""


@dataclass(frozen=True)
class MemoryReviewEvent:
    """Immutable audit event for one human review transition."""

    review_id: int
    memory_id: str
    from_status: MemoryStatus
    to_status: MemoryStatus
    reviewer: str
    reason: str
    reviewed_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "memory_id": self.memory_id,
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "reviewed_at": self.reviewed_at,
        }


class MemoryRepository(Protocol):
    """Persistence operations needed by memory extraction and retrieval."""

    def upsert(self, record: MemoryRecord) -> MemoryRecord: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def get_by_fingerprint(
        self, namespace: MemoryNamespace, fingerprint: str
    ) -> MemoryRecord | None: ...

    def list_memories(
        self,
        namespace: MemoryNamespace,
        *,
        status: MemoryStatus | None = None,
        searchable_only: bool = False,
        limit: int = 50,
        offset: int = 0,
        now: datetime | None = None,
    ) -> list[MemoryRecord]: ...

    def set_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        *,
        reviewer: str = "system",
        reason: str = "status transition",
    ) -> MemoryRecord: ...

    def list_reviews(self, memory_id: str) -> list[MemoryReviewEvent]: ...


_MEMORY_COLUMNS = (
    "memory_id",
    "tenant_id",
    "device_type",
    "memory_type",
    "fingerprint",
    "schema_version",
    "status",
    "scenario",
    "lesson",
    "preconditions_json",
    "recommended_actions_json",
    "failure_codes_json",
    "evidence_run_ids_json",
    "evidence_case_ids_json",
    "source_revision",
    "confidence",
    "success_count",
    "contradiction_count",
    "created_at",
    "updated_at",
    "expires_at",
    "embedding_model",
    "embedding_version",
)

_MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
    CREATE TABLE IF NOT EXISTS agent_memories (
        memory_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        device_type TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('candidate', 'approved', 'retired')),
        scenario TEXT NOT NULL,
        lesson TEXT NOT NULL,
        preconditions_json TEXT NOT NULL,
        recommended_actions_json TEXT NOT NULL,
        failure_codes_json TEXT NOT NULL,
        evidence_run_ids_json TEXT NOT NULL,
        evidence_case_ids_json TEXT NOT NULL,
        source_revision TEXT NOT NULL,
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        success_count INTEGER NOT NULL CHECK (success_count >= 0),
        contradiction_count INTEGER NOT NULL CHECK (contradiction_count >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT,
        embedding_model TEXT NOT NULL,
        embedding_version TEXT NOT NULL,
        UNIQUE (tenant_id, device_type, memory_type, fingerprint)
    )
    """,
        """
    CREATE INDEX IF NOT EXISTS idx_agent_memories_namespace_status
    ON agent_memories (tenant_id, device_type, memory_type, status, updated_at DESC)
    """,
        """
    CREATE INDEX IF NOT EXISTS idx_agent_memories_fingerprint
    ON agent_memories (fingerprint)
    """,
    ),
    2: (
        """
        CREATE TABLE IF NOT EXISTS memory_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL REFERENCES agent_memories(memory_id),
            from_status TEXT NOT NULL,
            to_status TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            reason TEXT NOT NULL,
            reviewed_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_reviews_memory_time
        ON memory_reviews (memory_id, reviewed_at, review_id)
        """,
    ),
}


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordered_union(first: Iterable[str], second: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*first, *second)))


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _later_record(first: MemoryRecord, second: MemoryRecord) -> MemoryRecord:
    return second if _timestamp(second.updated_at) >= _timestamp(first.updated_at) else first


def _merge_status(first: MemoryStatus, second: MemoryStatus) -> MemoryStatus:
    """Never regress review state or resurrect retired knowledge implicitly."""
    if MemoryStatus.RETIRED in {first, second}:
        return MemoryStatus.RETIRED
    if MemoryStatus.APPROVED in {first, second}:
        return MemoryStatus.APPROVED
    return MemoryStatus.CANDIDATE


class SQLiteMemoryRepository:
    """Transactional SQLite repository with deterministic fingerprint upserts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(_MIGRATION_TABLE_SQL)
            row = connection.execute(
                "SELECT MAX(version) AS version FROM memory_schema_migrations"
            ).fetchone()
            current_version = int(row["version"] or 0)
            if current_version > MEMORY_DB_SCHEMA_VERSION:
                raise RuntimeError(
                    "memory database schema is newer than this application: "
                    f"{current_version} > {MEMORY_DB_SCHEMA_VERSION}"
                )
            for version in range(current_version + 1, MEMORY_DB_SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS[version]:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO memory_schema_migrations(version, applied_at) VALUES (?, ?)",
                    (
                        version,
                        datetime.now(UTC).isoformat(timespec="milliseconds"),
                    ),
                )

    @staticmethod
    def _validate_fingerprint(record: MemoryRecord) -> None:
        expected = build_memory_fingerprint(
            namespace=record.namespace.as_tuple(),
            scenario=record.scenario,
            lesson=record.lesson,
            preconditions=record.preconditions,
            recommended_actions=record.recommended_actions,
            failure_codes=record.failure_codes,
        )
        if record.fingerprint != expected:
            raise MemoryConflictError("memory fingerprint does not match its semantic content")

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord.from_dict(
            {
                "memory_id": row["memory_id"],
                "namespace": {
                    "tenant_id": row["tenant_id"],
                    "device_type": row["device_type"],
                    "memory_type": row["memory_type"],
                },
                "fingerprint": row["fingerprint"],
                "schema_version": row["schema_version"],
                "status": row["status"],
                "scenario": row["scenario"],
                "lesson": row["lesson"],
                "preconditions": json.loads(row["preconditions_json"]),
                "recommended_actions": json.loads(row["recommended_actions_json"]),
                "failure_codes": json.loads(row["failure_codes_json"]),
                "evidence_run_ids": json.loads(row["evidence_run_ids_json"]),
                "evidence_case_ids": json.loads(row["evidence_case_ids_json"]),
                "source_revision": row["source_revision"],
                "confidence": row["confidence"],
                "success_count": row["success_count"],
                "contradiction_count": row["contradiction_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "expires_at": row["expires_at"],
                "embedding_model": row["embedding_model"],
                "embedding_version": row["embedding_version"],
            }
        )

    @staticmethod
    def _params(record: MemoryRecord) -> dict[str, object]:
        return {
            "memory_id": record.memory_id,
            "tenant_id": record.namespace.tenant_id,
            "device_type": record.namespace.device_type,
            "memory_type": record.memory_type.value,
            "fingerprint": record.fingerprint,
            "schema_version": record.schema_version,
            "status": record.status.value,
            "scenario": record.scenario,
            "lesson": record.lesson,
            "preconditions_json": _json_dump(record.preconditions),
            "recommended_actions_json": _json_dump(record.recommended_actions),
            "failure_codes_json": _json_dump(record.failure_codes),
            "evidence_run_ids_json": _json_dump(record.evidence_run_ids),
            "evidence_case_ids_json": _json_dump(record.evidence_case_ids),
            "source_revision": record.source_revision,
            "confidence": record.confidence,
            "success_count": record.success_count,
            "contradiction_count": record.contradiction_count,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "expires_at": record.expires_at,
            "embedding_model": record.embedding_model,
            "embedding_version": record.embedding_version,
        }

    @staticmethod
    def _merge(existing: MemoryRecord, incoming: MemoryRecord) -> MemoryRecord:
        preferred = _later_record(existing, incoming)
        return replace(
            preferred,
            memory_id=existing.memory_id,
            fingerprint=existing.fingerprint,
            status=_merge_status(existing.status, incoming.status),
            evidence_run_ids=_ordered_union(existing.evidence_run_ids, incoming.evidence_run_ids),
            evidence_case_ids=_ordered_union(
                existing.evidence_case_ids, incoming.evidence_case_ids
            ),
            success_count=max(existing.success_count, incoming.success_count),
            contradiction_count=max(existing.contradiction_count, incoming.contradiction_count),
            created_at=existing.created_at,
            updated_at=max((existing.updated_at, incoming.updated_at), key=_timestamp),
        )

    @staticmethod
    def _find_identity(
        connection: sqlite3.Connection,
        namespace: MemoryNamespace,
        fingerprint: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT * FROM agent_memories
            WHERE tenant_id = ? AND device_type = ? AND memory_type = ? AND fingerprint = ?
            """,
            (*namespace.as_tuple(), fingerprint),
        ).fetchone()

    @staticmethod
    def _write(connection: sqlite3.Connection, record: MemoryRecord, *, insert: bool) -> None:
        params = SQLiteMemoryRepository._params(record)
        if insert:
            columns = ", ".join(_MEMORY_COLUMNS)
            values = ", ".join(f":{column}" for column in _MEMORY_COLUMNS)
            connection.execute(
                f"INSERT INTO agent_memories ({columns}) VALUES ({values})",  # noqa: S608
                params,
            )
            return
        assignments = ", ".join(
            f"{column} = :{column}" for column in _MEMORY_COLUMNS if column != "memory_id"
        )
        connection.execute(
            f"UPDATE agent_memories SET {assignments} WHERE memory_id = :memory_id",  # noqa: S608
            params,
        )

    def upsert(self, record: MemoryRecord) -> MemoryRecord:
        """Insert or merge a memory; replaying the same input is idempotent."""
        self._validate_fingerprint(record)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = self._find_identity(connection, record.namespace, record.fingerprint)
                stored = record if row is None else self._merge(self._row_to_record(row), record)
                self._write(connection, stored, insert=row is None)
                return stored
        except sqlite3.IntegrityError as exc:
            raise MemoryConflictError(
                f"memory identity conflicts with persisted data: {exc}"
            ) from exc

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def get_by_fingerprint(
        self, namespace: MemoryNamespace, fingerprint: str
    ) -> MemoryRecord | None:
        with self._connect() as connection:
            row = self._find_identity(connection, namespace, fingerprint)
        return self._row_to_record(row) if row is not None else None

    def list_memories(
        self,
        namespace: MemoryNamespace,
        *,
        status: MemoryStatus | None = None,
        searchable_only: bool = False,
        limit: int = 50,
        offset: int = 0,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must not be negative")
        if searchable_only and status not in (None, MemoryStatus.APPROVED):
            raise ValueError("searchable_only cannot be combined with a non-approved status")

        target_status = MemoryStatus.APPROVED if searchable_only else status
        clauses = ["tenant_id = ?", "device_type = ?", "memory_type = ?"]
        params: list[object] = list(namespace.as_tuple())
        if target_status is not None:
            clauses.append("status = ?")
            params.append(MemoryStatus(target_status).value)
        query = (
            "SELECT * FROM agent_memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC, memory_id ASC"
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        records = [self._row_to_record(row) for row in rows]
        if searchable_only:
            records = [record for record in records if record.is_searchable(now)]
        return records[offset : offset + limit]

    def set_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        *,
        reviewer: str = "system",
        reason: str = "status transition",
    ) -> MemoryRecord:
        """Apply the domain lifecycle rules and persist the transition atomically."""
        clean_reviewer = reviewer.strip()
        clean_reason = reason.strip()
        if not clean_reviewer:
            raise ValueError("reviewer must not be empty")
        if not clean_reason:
            raise ValueError("review reason must not be empty")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM agent_memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"memory not found: {memory_id}")
            current = self._row_to_record(row)
            target = MemoryStatus(status)
            if target is current.status:
                return current
            reviewed_at = datetime.now(UTC).isoformat(timespec="milliseconds")
            updated = current.with_status(target, updated_at=reviewed_at)
            self._write(connection, updated, insert=False)
            connection.execute(
                """
                INSERT INTO memory_reviews (
                    memory_id, from_status, to_status, reviewer, reason, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    current.status.value,
                    target.value,
                    clean_reviewer,
                    clean_reason,
                    reviewed_at,
                ),
            )
            return updated

    def list_reviews(self, memory_id: str) -> list[MemoryReviewEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT review_id, memory_id, from_status, to_status, reviewer, reason, reviewed_at
                FROM memory_reviews
                WHERE memory_id = ?
                ORDER BY reviewed_at, review_id
                """,
                (memory_id,),
            ).fetchall()
        return [
            MemoryReviewEvent(
                review_id=int(row["review_id"]),
                memory_id=str(row["memory_id"]),
                from_status=MemoryStatus(str(row["from_status"])),
                to_status=MemoryStatus(str(row["to_status"])),
                reviewer=str(row["reviewer"]),
                reason=str(row["reason"]),
                reviewed_at=str(row["reviewed_at"]),
            )
            for row in rows
        ]
