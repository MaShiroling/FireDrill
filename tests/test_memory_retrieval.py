"""Approved-only memory index and retrieval behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.memory import (
    MemoryNamespace,
    MemoryRecord,
    MemoryRetrievalService,
    MemorySearchSource,
    MemoryStatus,
    MemoryType,
    MemoryVectorEntry,
    MemoryVectorMatch,
    MemoryVectorSyncStats,
    MilvusMemoryVectorIndex,
    SQLiteMemoryRepository,
)


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(text.count("提交")), float(text.count("验证"))]


class FakeVectorIndex:
    def __init__(self) -> None:
        self.entries: dict[MemoryNamespace, dict[str, MemoryVectorEntry]] = {}
        self.fail_search = False

    def replace_namespace(
        self,
        namespace: MemoryNamespace,
        entries: list[MemoryVectorEntry],
    ) -> MemoryVectorSyncStats:
        previous = self.entries.get(namespace, {})
        current = {entry.memory_id: entry for entry in entries}
        self.entries[namespace] = current
        return MemoryVectorSyncStats(
            indexed=len(current),
            removed=len(set(previous) - set(current)),
        )

    def search(
        self,
        namespace: MemoryNamespace,
        query_vector: list[float],
        *,
        limit: int,
    ) -> list[MemoryVectorMatch]:
        if self.fail_search:
            raise RuntimeError("milvus unavailable")
        matches = [
            MemoryVectorMatch(
                memory_id=entry.memory_id,
                score=sum(a * b for a, b in zip(entry.vector, query_vector, strict=True)),
            )
            for entry in self.entries.get(namespace, {}).values()
        ]
        return sorted(matches, key=lambda match: -match.score)[:limit]


class FakeMilvusClient:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None
        self.upserted: list[dict[str, object]] = []
        self.deleted: list[str] = []

    def has_collection(self, **_kwargs) -> bool:
        return False

    def create_collection(self, **kwargs) -> None:
        self.created = kwargs

    def query(self, **_kwargs) -> list[dict[str, str]]:
        return [{"memory_id": "mem-stale"}]

    def upsert(self, *, data, **_kwargs) -> None:
        self.upserted = data

    def delete(self, *, ids, **_kwargs) -> None:
        self.deleted = ids

    def search(self, **_kwargs):
        return [
            [
                {
                    "id": "mem-current",
                    "distance": 0.91,
                    "entity": {"memory_id": "mem-current"},
                }
            ]
        ]


def _namespace(
    *,
    tenant_id: str = "tenant-a",
    memory_type: MemoryType = MemoryType.FAILURE_LESSON,
) -> MemoryNamespace:
    return MemoryNamespace(
        tenant_id=tenant_id,
        device_type="firewall",
        memory_type=memory_type,
    )


def _record(
    scenario: str,
    lesson: str,
    *,
    namespace: MemoryNamespace | None = None,
    expires_at: str | None = None,
) -> MemoryRecord:
    return MemoryRecord.create(
        namespace=namespace or _namespace(),
        scenario=scenario,
        lesson=lesson,
        recommended_actions=("get_config_diff",),
        evidence_run_ids=("run-001",),
        expires_at=expires_at,
        confidence=0.8,
    )


def _service(tmp_path):
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    index = FakeVectorIndex()
    return (
        repository,
        index,
        MemoryRetrievalService(
            repository,
            index,
            FakeEmbedder(),
            default_top_k=3,
        ),
    )


def test_sync_indexes_only_approved_unexpired_memories(tmp_path) -> None:
    repository, index, service = _service(tmp_path)
    candidate = repository.upsert(_record("候选", "提交后验证"))
    approved = repository.upsert(_record("提交超时", "提交前后都要验证"))
    repository.set_status(approved.memory_id, MemoryStatus.APPROVED)
    expired = repository.upsert(
        _record(
            "旧固件",
            "旧提交策略",
            expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        )
    )
    repository.set_status(expired.memory_id, MemoryStatus.APPROVED)

    result = service.sync_namespace(_namespace())

    assert not result.degraded
    assert result.indexed == 1
    assert set(index.entries[_namespace()]) == {approved.memory_id}
    assert candidate.memory_id not in index.entries[_namespace()]
    assert expired.memory_id not in index.entries[_namespace()]


def test_stale_vector_hit_is_rechecked_against_sqlite_status(tmp_path) -> None:
    repository, _index, service = _service(tmp_path)
    stored = repository.upsert(_record("提交超时", "先查询状态再决定是否重试提交"))
    repository.set_status(stored.memory_id, MemoryStatus.APPROVED)
    service.sync_namespace(_namespace())
    repository.set_status(
        stored.memory_id,
        MemoryStatus.RETIRED,
        reviewer="alice",
        reason="经验已失效",
    )

    result = service.search(
        "提交超时",
        tenant_id="tenant-a",
        device_type="firewall",
    )

    assert result.hits == ()


def test_namespace_isolation_survives_cross_scope_vector_candidate(tmp_path) -> None:
    repository, index, service = _service(tmp_path)
    other = repository.upsert(
        _record("提交超时", "租户 B 的私有经验", namespace=_namespace(tenant_id="tenant-b"))
    )
    repository.set_status(other.memory_id, MemoryStatus.APPROVED)
    index.entries[_namespace()] = {
        other.memory_id: MemoryVectorEntry(
            memory_id=other.memory_id,
            namespace=_namespace(),
            vector=(2.0, 0.0),
            embedding_model=other.embedding_model,
            embedding_version=other.embedding_version,
            updated_at=other.updated_at,
        )
    }

    result = service.search(
        "提交超时",
        tenant_id="tenant-a",
        device_type="firewall",
    )

    assert result.hits == ()


def test_vector_failure_falls_back_to_approved_lexical_search(tmp_path) -> None:
    repository, index, service = _service(tmp_path)
    approved = repository.upsert(_record("Commit 提交超时", "应先查询运行配置再验证"))
    repository.set_status(approved.memory_id, MemoryStatus.APPROVED)
    repository.upsert(_record("Commit 提交超时", "未经审核的候选经验"))
    index.fail_search = True

    result = service.search(
        "提交超时怎么验证",
        tenant_id="tenant-a",
        device_type="firewall",
    )

    assert result.mode == MemorySearchSource.LEXICAL.value
    assert result.degraded_reason == "milvus unavailable"
    assert [hit.record.memory_id for hit in result.hits] == [approved.memory_id]
    assert result.hits[0].source is MemorySearchSource.LEXICAL


def test_explicit_lexical_search_uses_only_approved_sqlite_records(tmp_path) -> None:
    repository, index, service = _service(tmp_path)
    approved = repository.upsert(_record("Commit 提交超时", "应先核对运行状态"))
    repository.set_status(approved.memory_id, MemoryStatus.APPROVED)
    repository.upsert(_record("Commit 提交超时", "未经审核的候选经验"))
    index.fail_search = True

    result = service.search_lexical(
        "提交超时",
        tenant_id="tenant-a",
        device_type="firewall",
        degraded_reason="semantic timeout",
    )

    assert result.mode == MemorySearchSource.LEXICAL.value
    assert result.degraded_reason == "semantic timeout"
    assert [hit.record.memory_id for hit in result.hits] == [approved.memory_id]


def test_sync_failure_is_reported_without_changing_sqlite(tmp_path) -> None:
    repository, index, _service_instance = _service(tmp_path)
    approved = repository.upsert(_record("提交超时", "提交后验证"))
    approved = repository.set_status(approved.memory_id, MemoryStatus.APPROVED)

    class BrokenEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding quota exceeded")

    broken = MemoryRetrievalService(repository, index, BrokenEmbedder())
    result = broken.sync_namespace(_namespace())

    assert result.degraded
    assert result.error == "embedding quota exceeded"
    assert repository.get(approved.memory_id) == approved
    assert index.entries == {}


def test_sync_removes_retired_projection(tmp_path) -> None:
    repository, index, service = _service(tmp_path)
    stored = repository.upsert(_record("提交超时", "提交后验证"))
    repository.set_status(stored.memory_id, MemoryStatus.APPROVED)
    service.sync_namespace(_namespace())
    repository.set_status(stored.memory_id, MemoryStatus.RETIRED)

    result = service.sync_namespace(_namespace())

    assert result.indexed == 0
    assert result.removed == 1
    assert index.entries[_namespace()] == {}


def test_top_k_must_be_positive(tmp_path) -> None:
    _repository, _index, service = _service(tmp_path)

    try:
        service.search(
            "提交",
            tenant_id="tenant-a",
            device_type="firewall",
            top_k=0,
        )
    except ValueError as exc:
        assert str(exc) == "top_k must be positive"
    else:
        raise AssertionError("expected top_k validation")


def test_irrelevant_low_score_vector_candidate_is_not_returned(tmp_path) -> None:
    repository, _index, service = _service(tmp_path)
    approved = repository.upsert(_record("提交超时", "先验证运行状态"))
    repository.set_status(approved.memory_id, MemoryStatus.APPROVED)
    service.sync_namespace(_namespace())

    result = service.search(
        "完全无关",
        tenant_id="tenant-a",
        device_type="firewall",
    )

    assert result.hits == ()


def test_milvus_projection_uses_separate_collection_and_removes_stale_rows() -> None:
    client = FakeMilvusClient()
    index = MilvusMemoryVectorIndex(
        client,
        collection_name="agent_memory_v1",
        vector_dim=2,
    )
    entry = MemoryVectorEntry(
        memory_id="mem-current",
        namespace=_namespace(),
        vector=(1.0, 0.0),
        embedding_model="test-embedding",
        embedding_version="1",
        updated_at=datetime.now(UTC).isoformat(),
    )

    stats = index.replace_namespace(_namespace(), [entry])
    matches = index.search(_namespace(), [1.0, 0.0], limit=3)

    assert client.created is not None
    assert client.created["collection_name"] == "agent_memory_v1"
    assert client.created["collection_name"] != "biz"
    assert client.upserted[0]["memory_id"] == "mem-current"
    assert client.deleted == ["mem-stale"]
    assert stats == MemoryVectorSyncStats(indexed=1, removed=1)
    assert matches == [MemoryVectorMatch(memory_id="mem-current", score=0.91)]
