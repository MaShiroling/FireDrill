"""Long-term memory schema, namespace and fingerprint tests."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.memory import (
    MemoryNamespace,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    build_memory_fingerprint,
)


def _namespace(tenant_id: str = "default") -> MemoryNamespace:
    return MemoryNamespace(
        tenant_id=tenant_id,
        device_type="Firewall",
        memory_type=MemoryType.FAILURE_LESSON,
    )


def test_fingerprint_ignores_harmless_formatting_and_failure_code_order() -> None:
    first = build_memory_fingerprint(
        namespace=_namespace().as_tuple(),
        scenario=" Commit   状态未知 ",
        lesson="先查询 Running，再决定是否重试",
        recommended_actions=["get_firewall_overview", "get_config_diff"],
        failure_codes=["verification_missing", "commit_state_unknown"],
    )
    second = build_memory_fingerprint(
        namespace=("DEFAULT", "FIREWALL", "FAILURE_LESSON"),
        scenario="commit 状态未知",
        lesson="先查询 running，再决定是否重试",
        recommended_actions=["GET_FIREWALL_OVERVIEW", "GET_CONFIG_DIFF"],
        failure_codes=["commit_state_unknown", "verification_missing"],
    )

    assert first == second


def test_fingerprint_keeps_namespace_and_action_order_semantics() -> None:
    common = {
        "scenario": "提交状态未知",
        "lesson": "先读真实状态",
        "recommended_actions": ["get_firewall_overview", "get_config_diff"],
    }

    base = build_memory_fingerprint(namespace=_namespace().as_tuple(), **common)
    other_tenant = build_memory_fingerprint(namespace=_namespace("tenant-b").as_tuple(), **common)
    reversed_actions = build_memory_fingerprint(
        namespace=_namespace().as_tuple(),
        **{**common, "recommended_actions": list(reversed(common["recommended_actions"]))},
    )

    assert base != other_tenant
    assert base != reversed_actions


def test_memory_record_round_trips_through_json() -> None:
    record = MemoryRecord.create(
        namespace=_namespace(),
        scenario="提交超时且状态未知",
        lesson="禁止盲目重复提交，应先查询真实运行状态",
        preconditions={"tool": "commit_config"},
        recommended_actions=["get_firewall_overview", "get_config_diff"],
        failure_codes=["commit_state_unknown"],
        evidence_run_ids=["run-001", "run-001", "run-002"],
        evidence_case_ids=["FW-F04"],
        source_revision="abc123",
        confidence=0.9,
    )

    payload = json.loads(json.dumps(record.to_dict(), ensure_ascii=False))
    restored = MemoryRecord.from_dict(payload)

    assert restored == record
    assert restored.memory_type is MemoryType.FAILURE_LESSON
    assert restored.evidence_run_ids == ("run-001", "run-002")


def test_only_approved_unexpired_memory_is_searchable() -> None:
    candidate = MemoryRecord.create(
        namespace=_namespace(),
        scenario="规则定位",
        lesson="先查询真实 rule_id",
        evidence_run_ids=["run-rule-id"],
    )
    approved = candidate.with_status(MemoryStatus.APPROVED)
    retired = approved.with_status(MemoryStatus.RETIRED)

    assert candidate.is_searchable() is False
    assert approved.is_searchable() is True
    assert retired.is_searchable() is False


def test_expired_approved_memory_is_not_searchable() -> None:
    now = datetime.now(UTC)
    record = MemoryRecord.create(
        namespace=_namespace(),
        scenario="旧固件处理经验",
        lesson="只适用于旧版本",
        status=MemoryStatus.APPROVED,
        evidence_run_ids=["run-expired"],
        expires_at=(now - timedelta(seconds=1)).isoformat(),
    )

    assert record.is_expired(now) is True
    assert record.is_searchable(now) is False


def test_schema_rejects_invalid_confidence_and_status_transition() -> None:
    with pytest.raises(ValueError, match="confidence"):
        MemoryRecord.create(
            namespace=_namespace(),
            scenario="提交失败",
            lesson="查询状态",
            confidence=1.1,
        )

    approved = MemoryRecord.create(
        namespace=_namespace(),
        scenario="提交失败",
        lesson="查询状态",
        status=MemoryStatus.APPROVED,
        evidence_case_ids=["FW-F04"],
    )
    with pytest.raises(ValueError, match="invalid memory status transition"):
        approved.with_status(MemoryStatus.CANDIDATE)


def test_approved_memory_requires_evidence_and_preconditions_are_json_safe() -> None:
    with pytest.raises(ValueError, match="evidence ID"):
        MemoryRecord.create(
            namespace=_namespace(),
            scenario="无来源经验",
            lesson="不能进入规划上下文",
            status=MemoryStatus.APPROVED,
        )

    with pytest.raises(ValueError, match="JSON serializable"):
        MemoryRecord.create(
            namespace=_namespace(),
            scenario="非法前置条件",
            lesson="不能落库",
            preconditions={"value": object()},
        )
