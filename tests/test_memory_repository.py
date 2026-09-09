"""SQLite long-term memory persistence and lifecycle tests."""

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.memory import (
    MemoryConflictError,
    MemoryNamespace,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    SQLiteMemoryRepository,
)


def _namespace(tenant_id: str = "tenant-a") -> MemoryNamespace:
    return MemoryNamespace(
        tenant_id=tenant_id,
        device_type="firewall",
        memory_type=MemoryType.FAILURE_LESSON,
    )


def _record(
    *,
    namespace: MemoryNamespace | None = None,
    run_ids: tuple[str, ...] = ("run-001",),
    status: MemoryStatus = MemoryStatus.CANDIDATE,
    expires_at: str | None = None,
) -> MemoryRecord:
    return MemoryRecord.create(
        namespace=namespace or _namespace(),
        scenario="Commit 返回超时且最终状态未知",
        lesson="禁止盲目重复提交，应先读取 Running Revision 和配置 Diff",
        preconditions={"tool": "commit_config"},
        recommended_actions=("get_firewall_overview", "get_config_diff"),
        failure_codes=("commit_state_unknown",),
        evidence_run_ids=run_ids,
        status=status,
        expires_at=expires_at,
        source_revision="abc123",
        confidence=0.8,
    )


def test_repository_initializes_versioned_schema(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "nested" / "agent_memory.db")

    with sqlite3.connect(repository.db_path) as connection:
        versions = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM memory_schema_migrations ORDER BY version"
            ).fetchall()
        ]
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list('agent_memories')").fetchall()
        }

    assert versions == [1, 2]
    assert "idx_agent_memories_namespace_status" in indexes
    with sqlite3.connect(repository.db_path) as connection:
        review_indexes = {
            row[1] for row in connection.execute("PRAGMA index_list('memory_reviews')").fetchall()
        }
    assert "idx_memory_reviews_memory_time" in review_indexes


def test_v1_database_migrates_without_losing_memories(tmp_path) -> None:
    db_path = tmp_path / "agent_memory.db"
    first = SQLiteMemoryRepository(db_path)
    original = first.upsert(_record())
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE memory_reviews")
        connection.execute("DELETE FROM memory_schema_migrations WHERE version = 2")

    migrated = SQLiteMemoryRepository(db_path)

    assert migrated.get(original.memory_id) == original
    with sqlite3.connect(db_path) as connection:
        versions = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM memory_schema_migrations ORDER BY version"
            ).fetchall()
        ]
    assert versions == [1, 2]


def test_memory_survives_repository_restart(tmp_path) -> None:
    db_path = tmp_path / "agent_memory.db"
    first = SQLiteMemoryRepository(db_path)
    original = _record()
    first.upsert(original)

    restarted = SQLiteMemoryRepository(db_path)

    assert restarted.get(original.memory_id) == original
    assert restarted.get_by_fingerprint(original.namespace, original.fingerprint) == original


def test_upsert_merges_evidence_and_is_idempotent(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    first = _record(run_ids=("run-001",))
    stored = repository.upsert(first)
    incoming = replace(
        _record(run_ids=("run-001", "run-002")),
        status=MemoryStatus.APPROVED,
        success_count=2,
        confidence=0.9,
        updated_at=(datetime.now(UTC) + timedelta(seconds=1)).isoformat(),
    )

    merged = repository.upsert(incoming)
    replayed = repository.upsert(incoming)

    assert merged.memory_id == stored.memory_id
    assert merged.status is MemoryStatus.APPROVED
    assert merged.evidence_run_ids == ("run-001", "run-002")
    assert merged.success_count == 2
    assert merged.confidence == 0.9
    assert replayed == merged
    assert repository.list_memories(_namespace()) == [merged]


def test_namespace_isolation_and_filters(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    tenant_a = repository.upsert(_record())
    tenant_b = repository.upsert(_record(namespace=_namespace("tenant-b")))

    assert repository.list_memories(_namespace()) == [tenant_a]
    assert repository.list_memories(_namespace("tenant-b")) == [tenant_b]
    assert repository.get_by_fingerprint(_namespace("tenant-b"), tenant_a.fingerprint) is None


def test_only_approved_unexpired_memories_are_searchable(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    candidate = repository.upsert(_record())
    approved = repository.set_status(candidate.memory_id, MemoryStatus.APPROVED)
    assert repository.list_memories(_namespace(), searchable_only=True) == [approved]

    expired = replace(
        _record(run_ids=("run-expired",)),
        memory_id="mem-expired",
        scenario="旧固件提交失败",
        lesson="仅适用于已经下线的旧固件",
        expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    )
    expired = replace(
        expired,
        fingerprint=MemoryRecord.create(
            namespace=expired.namespace,
            scenario=expired.scenario,
            lesson=expired.lesson,
            preconditions=expired.preconditions,
            recommended_actions=expired.recommended_actions,
            failure_codes=expired.failure_codes,
            evidence_run_ids=expired.evidence_run_ids,
            expires_at=expired.expires_at,
        ).fingerprint,
    )
    repository.upsert(expired)
    repository.set_status(expired.memory_id, MemoryStatus.APPROVED)

    assert repository.list_memories(_namespace(), searchable_only=True) == [approved]


def test_status_rules_and_missing_memory_are_enforced(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    stored = repository.upsert(_record())
    approved = repository.set_status(stored.memory_id, MemoryStatus.APPROVED)
    retired = repository.set_status(approved.memory_id, MemoryStatus.RETIRED)

    with pytest.raises(ValueError, match="invalid memory status transition"):
        repository.set_status(retired.memory_id, MemoryStatus.APPROVED)
    with pytest.raises(KeyError, match="memory not found"):
        repository.set_status("mem-missing", MemoryStatus.RETIRED)


def test_status_transitions_write_an_idempotent_review_audit(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    stored = repository.upsert(_record())

    approved = repository.set_status(
        stored.memory_id,
        MemoryStatus.APPROVED,
        reviewer="alice",
        reason="三次回放终态断言均通过",
    )
    repository.set_status(
        stored.memory_id,
        MemoryStatus.APPROVED,
        reviewer="alice",
        reason="重复请求",
    )
    history = repository.list_reviews(stored.memory_id)

    assert approved.status is MemoryStatus.APPROVED
    assert len(history) == 1
    assert history[0].from_status is MemoryStatus.CANDIDATE
    assert history[0].to_status is MemoryStatus.APPROVED
    assert history[0].reviewer == "alice"
    assert history[0].reason == "三次回放终态断言均通过"
    assert history[0].to_dict()["memory_id"] == stored.memory_id


def test_review_metadata_must_not_be_empty(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    stored = repository.upsert(_record())

    with pytest.raises(ValueError, match="reviewer"):
        repository.set_status(
            stored.memory_id,
            MemoryStatus.APPROVED,
            reviewer=" ",
            reason="valid",
        )
    with pytest.raises(ValueError, match="reason"):
        repository.set_status(
            stored.memory_id,
            MemoryStatus.APPROVED,
            reviewer="alice",
            reason=" ",
        )


def test_upsert_does_not_resurrect_retired_memory(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    stored = repository.upsert(_record())
    repository.set_status(stored.memory_id, MemoryStatus.APPROVED)
    retired = repository.set_status(stored.memory_id, MemoryStatus.RETIRED)

    replayed = repository.upsert(_record(run_ids=("run-001", "run-002")))

    assert replayed.memory_id == retired.memory_id
    assert replayed.status is MemoryStatus.RETIRED
    assert replayed.evidence_run_ids == ("run-001", "run-002")
    assert repository.list_memories(_namespace(), searchable_only=True) == []


def test_repository_rejects_tampered_fingerprint_and_memory_id_collision(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    original = repository.upsert(_record())

    with pytest.raises(MemoryConflictError, match="fingerprint"):
        repository.upsert(replace(_record(), fingerprint="not-a-valid-fingerprint"))

    other = MemoryRecord.create(
        namespace=_namespace(),
        scenario="规则命中异常",
        lesson="先检查规则顺序和命中计数",
        evidence_run_ids=("run-other",),
    )
    with pytest.raises(MemoryConflictError, match="identity conflicts"):
        repository.upsert(replace(other, memory_id=original.memory_id))


def test_list_argument_validation(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")

    with pytest.raises(ValueError, match="limit"):
        repository.list_memories(_namespace(), limit=0)
    with pytest.raises(ValueError, match="offset"):
        repository.list_memories(_namespace(), offset=-1)
    with pytest.raises(ValueError, match="non-approved"):
        repository.list_memories(_namespace(), status=MemoryStatus.CANDIDATE, searchable_only=True)


def test_repository_rejects_newer_database_schema(tmp_path) -> None:
    db_path = tmp_path / "future.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE memory_schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO memory_schema_migrations(version, applied_at) VALUES (?, ?)",
            (999, datetime.now(UTC).isoformat()),
        )

    with pytest.raises(RuntimeError, match="newer than this application"):
        SQLiteMemoryRepository(db_path)
