"""Scope-aware human review service tests."""

import pytest

from app.memory import (
    MemoryNamespace,
    MemoryRecord,
    MemoryReviewService,
    MemoryStatus,
    MemoryType,
    SQLiteMemoryRepository,
)


def _record(
    memory_type: MemoryType,
    *,
    tenant_id: str = "tenant-a",
    scenario: str = "防火墙变更",
) -> MemoryRecord:
    return MemoryRecord.create(
        namespace=MemoryNamespace(
            tenant_id=tenant_id,
            device_type="firewall",
            memory_type=memory_type,
        ),
        scenario=scenario,
        lesson="执行后必须核对终态",
        recommended_actions=("get_firewall_overview",),
        evidence_case_ids=("FW-C01",),
    )


def test_review_list_combines_types_and_filters_status(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    service = MemoryReviewService(repository)
    failure = repository.upsert(_record(MemoryType.FAILURE_LESSON, scenario="失败教训"))
    success = repository.upsert(_record(MemoryType.SUCCESSFUL_CASE, scenario="成功案例"))
    repository.set_status(
        success.memory_id,
        MemoryStatus.APPROVED,
        reviewer="alice",
        reason="证据充分",
    )

    candidates = service.list_for_review(tenant_id="tenant-a", device_type="firewall")
    approved = service.list_for_review(
        tenant_id="tenant-a",
        device_type="firewall",
        status=MemoryStatus.APPROVED,
    )

    assert candidates == [failure]
    assert [item.memory_id for item in approved] == [success.memory_id]


def test_approve_and_retire_keep_auditable_history(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    service = MemoryReviewService(repository)
    stored = repository.upsert(_record(MemoryType.FAILURE_LESSON))

    approved = service.approve(
        stored.memory_id,
        tenant_id="tenant-a",
        device_type="firewall",
        reviewer="alice",
        reason="回放证据与教训一致",
    )
    retired = service.retire(
        stored.memory_id,
        tenant_id="tenant-a",
        device_type="firewall",
        reviewer="bob",
        reason="新固件下该经验已失效",
    )
    history = service.review_history(
        stored.memory_id,
        tenant_id="tenant-a",
        device_type="firewall",
    )

    assert approved.status is MemoryStatus.APPROVED
    assert retired.status is MemoryStatus.RETIRED
    assert [(item.reviewer, item.to_status) for item in history] == [
        ("alice", MemoryStatus.APPROVED),
        ("bob", MemoryStatus.RETIRED),
    ]


def test_review_scope_hides_other_tenant_memory(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    service = MemoryReviewService(repository)
    stored = repository.upsert(_record(MemoryType.FAILURE_LESSON, tenant_id="tenant-secret"))

    with pytest.raises(KeyError, match="memory not found"):
        service.get_detail(
            stored.memory_id,
            tenant_id="tenant-a",
            device_type="firewall",
        )
    with pytest.raises(KeyError, match="memory not found"):
        service.approve(
            stored.memory_id,
            tenant_id="tenant-a",
            device_type="firewall",
            reviewer="mallory",
            reason="越权审核",
        )


def test_review_list_supports_type_and_pagination(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    service = MemoryReviewService(repository)
    repository.upsert(_record(MemoryType.FAILURE_LESSON, scenario="failure-a"))
    repository.upsert(_record(MemoryType.FAILURE_LESSON, scenario="failure-b"))
    repository.upsert(_record(MemoryType.SUCCESSFUL_CASE, scenario="success-a"))

    failures = service.list_for_review(
        tenant_id="tenant-a",
        device_type="firewall",
        memory_type=MemoryType.FAILURE_LESSON,
        limit=1,
        offset=1,
    )

    assert len(failures) == 1
    assert failures[0].memory_type is MemoryType.FAILURE_LESSON


def test_retired_memory_cannot_be_approved_again(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    service = MemoryReviewService(repository)
    stored = repository.upsert(_record(MemoryType.FAILURE_LESSON))
    service.retire(
        stored.memory_id,
        tenant_id="tenant-a",
        device_type="firewall",
        reviewer="alice",
        reason="候选内容不适用",
    )

    with pytest.raises(ValueError, match="invalid memory status transition"):
        service.approve(
            stored.memory_id,
            tenant_id="tenant-a",
            device_type="firewall",
            reviewer="alice",
            reason="尝试恢复",
        )
