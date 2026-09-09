"""Flywheel-to-memory candidate extraction tests."""

from app.memory import (
    MemoryStatus,
    MemoryType,
    SQLiteMemoryRepository,
    extract_failure_memories,
    extract_failure_memory,
    extract_recovered_memories,
    persist_memory_candidates,
)


def _failure_sample(**overrides):
    sample = {
        "sample_id": "failure-001",
        "source_case_id": "FW-C01",
        "category": "change",
        "task": "放通 TCP 443，提交并验证",
        "scenario": {"fault": "none"},
        "expect_success": True,
        "failure_codes": [
            "assertion_failed",
            "verification_missing",
            "false_completion",
        ],
        "failed_assertion_types": ["traffic", "revision"],
        "occurrence_count": 2,
        "priority_score": 90,
        "occurrences": [
            {"source_tag": "legacy", "run": 1, "trace_id": None},
            {"source_tag": "current", "run": 2, "trace_id": "trace-002"},
        ],
    }
    sample.update(overrides)
    return sample


def _replay_case():
    return {
        "id": "REPLAY-FW-F02-12345678",
        "category": "fault",
        "task": "请放通 trust 到 untrust 的 TCP 465，提交生效并验证",
        "scenario": {"fault": "commit_flaky", "fail_times": 2},
        "assert": [
            {"type": "rule_present"},
            {"type": "revision", "op": ">", "value": 1},
            {"type": "traffic", "expect": "allow"},
        ],
        "flywheel": {"source_case_id": "FW-F02"},
    }


def _regression_case(status: str = "recovered"):
    return {
        "case_id": "REPLAY-FW-F02-12345678",
        "source_case_id": "FW-F02",
        "runs": 3,
        "healthy_runs": 3 if status == "recovered" else 1,
        "target_failure_codes": ["retry_exhausted"],
        "target_recurrence_runs": 0 if status == "recovered" else 2,
        "status": status,
    }


def test_failure_sample_becomes_reviewable_candidate() -> None:
    memory = extract_failure_memory(
        _failure_sample(),
        tenant_id="Tenant A",
        device_type="Firewall",
        source_revision="flywheel-test",
    )

    assert memory is not None
    assert memory.memory_type is MemoryType.FAILURE_LESSON
    assert memory.status is MemoryStatus.CANDIDATE
    assert memory.namespace.tenant_id == "tenant-a"
    assert "终态验证" in memory.lesson
    assert memory.recommended_actions[:2] == ("get_config_diff", "commit_config")
    assert memory.evidence_run_ids == ("legacy:FW-C01:run-1", "trace-002")
    assert memory.evidence_case_ids == ("FW-C01",)
    assert memory.confidence == 0.7


def test_non_actionable_failure_is_filtered() -> None:
    memory = extract_failure_memory(
        _failure_sample(failure_codes=["assertion_failed", "evaluator_error"]),
        tenant_id="local",
        device_type="firewall",
    )

    assert memory is None


def test_failure_batch_persists_idempotently_and_accumulates_evidence(tmp_path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "agent_memory.db")
    first = extract_failure_memories(
        [_failure_sample(occurrences=[{"source_tag": "v1", "run": 1}])],
        tenant_id="local",
        device_type="firewall",
    )
    second = extract_failure_memories(
        [
            _failure_sample(
                occurrences=[
                    {"source_tag": "v1", "run": 1},
                    {"source_tag": "v2", "run": 2},
                ]
            )
        ],
        tenant_id="local",
        device_type="firewall",
    )

    first_stored = persist_memory_candidates(repository, first)
    second_stored = persist_memory_candidates(repository, second)

    assert second_stored[0].memory_id == first_stored[0].memory_id
    assert second_stored[0].evidence_run_ids == ("v1:FW-C01:run-1", "v2:FW-C01:run-2")
    assert len(repository.list_memories(first[0].namespace)) == 1


def test_only_recovered_replay_becomes_success_candidate() -> None:
    report = {"cases": [_regression_case(), _regression_case("unstable")]}
    catalog = {"REPLAY-FW-F02-12345678": _replay_case()}

    memories = extract_recovered_memories(
        report,
        catalog,
        tenant_id="local",
        device_type="firewall",
        source_revision="regression-v2",
    )

    assert len(memories) == 1
    memory = memories[0]
    assert memory.memory_type is MemoryType.SUCCESSFUL_CASE
    assert memory.status is MemoryStatus.CANDIDATE
    assert memory.success_count == 3
    assert memory.contradiction_count == 0
    assert memory.confidence == 0.8
    assert "有限重试" in memory.lesson
    assert memory.evidence_case_ids == ("REPLAY-FW-F02-12345678", "FW-F02")
    assert memory.recommended_actions == (
        "get_firewall_overview",
        "list_firewall_rules",
        "add_firewall_rule",
        "get_config_diff",
        "commit_config",
        "test_traffic",
    )


def test_recovered_case_without_catalog_or_task_is_skipped() -> None:
    report = {"cases": [_regression_case()]}

    assert (
        extract_recovered_memories(
            report,
            {},
            tenant_id="local",
            device_type="firewall",
        )
        == []
    )
