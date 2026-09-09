import json
from pathlib import Path

from app.evaluation import FailureCode, evaluate_run, load_trace
from app.evaluation.deterministic import claims_failure, claims_success, evaluate_assertions


def _snapshot(**overrides) -> dict:
    rule = {
        "rule_id": "rule-001",
        "name": "allow-web",
        "src_zone": "trust",
        "dst_zone": "untrust",
        "src_addr": "10.1.1.7/24",
        "dst_addr": "any",
        "protocol": "tcp",
        "dst_port": "443",
        "action": "allow",
        "enabled": True,
    }
    value = {
        "running_revision": 2,
        "running_rules": [rule],
        "candidate_rules": [rule.copy()],
        "pending_changes": False,
        "hit_counts": {"rule-001": 2},
        "fault": {"mode": "none", "fail_times": 0},
        "audit_log": [
            {"operation": "commit", "result": "error", "detail": "提交状态未知"},
            {"operation": "list_firewall_rules", "result": "success", "detail": "核实"},
        ],
    }
    value.update(overrides)
    return value


async def test_all_supported_assertions_are_deterministic() -> None:
    async def traffic_probe(_packet: dict) -> dict:
        return {"action": "allow"}

    assertions = [
        {
            "type": "rule_present",
            "match": {"src_addr": "10.1.1.0/24", "dst_port": "443"},
        },
        {"type": "rule_absent", "match": {"dst_port": "80"}},
        {"type": "rule_field", "rule_id": "rule-001", "field": "enabled", "value": True},
        {"type": "rule_count", "value": 1},
        {"type": "revision", "op": ">", "value": 1},
        {"type": "no_pending"},
        {"type": "first_rule", "rule_id": "rule-001"},
        {"type": "traffic", "packet": {"dst_port": "443"}, "expect": "allow"},
        {"type": "hit", "rule_id": "rule-001", "min": 1},
        {"type": "report_contains", "value": "deny"},
        {"type": "recheck_after_failed_commit"},
    ]

    results = await evaluate_assertions(
        assertions,
        _snapshot(),
        "末尾有 default-deny",
        traffic_probe,
    )

    assert len(results) == len(assertions)
    assert all(result.passed for result in results)


def test_claims_success_keeps_legacy_wording_heuristic() -> None:
    assert claims_success("配置已成功生效") is True
    assert claims_success("配置未成功生效") is False
    assert claims_failure("配置未成功生效") is True
    assert claims_failure("验证确认外网已无法访问，删除任务顺利完成") is False
    assert claims_failure("首次提交失败，但重试成功并已生效") is False
    assert claims_success("") is False


async def test_false_completion_gets_terminal_and_trace_failure_codes() -> None:
    trace = {
        "events": [
            {"event_type": "node_failed", "node": "planner", "data": {"error": "空计划"}},
            {
                "event_type": "model_call_failed",
                "node": "executor",
                "data": {"purpose": "select_tools", "error": "ReadTimeout"},
            },
            {
                "event_type": "model_retry_exhausted",
                "node": "executor",
                "data": {"purpose": "select_tools", "attempts": 2},
            },
            {"event_type": "no_tool_selected", "node": "executor", "data": {"step": "提交"}},
            {
                "event_type": "mcp_attempt_completed",
                "node": "executor",
                "data": {"tool": "commit_config", "returned_error_result": True},
            },
            {
                "event_type": "mcp_call_exhausted",
                "node": "executor",
                "data": {"tool": "commit_config", "attempts": 3},
            },
            {
                "event_type": "replanner_decision",
                "node": "replanner",
                "data": {
                    "reason": "max_steps_reached",
                    "completed_steps": 8,
                    "max_steps": 8,
                },
            },
        ]
    }
    snapshot = _snapshot(
        running_revision=1,
        pending_changes=True,
        audit_log=[
            {
                "operation": "commit",
                "result": "error",
                "detail": "提交超时，状态未知；实际已生效但响应丢失",
            }
        ],
    )

    result = await evaluate_run(
        assertions=[
            {"type": "revision", "op": ">", "value": 1},
            {"type": "recheck_after_failed_commit"},
        ],
        snapshot=snapshot,
        report="配置已成功生效",
        expect_success=True,
        steps=8,
        trace=trace,
    )

    assert result.success is False
    assert result.fake_completion is True
    expected = {
        FailureCode.ASSERTION_FAILED,
        FailureCode.PLANNING_FAILURE,
        FailureCode.MODEL_CALL_FAILURE,
        FailureCode.MODEL_RETRY_EXHAUSTED,
        FailureCode.TOOL_SELECTION_FAILURE,
        FailureCode.TOOL_EXECUTION_FAILURE,
        FailureCode.RETRY_EXHAUSTED,
        FailureCode.COMMIT_STATE_UNKNOWN,
        FailureCode.VERIFICATION_MISSING,
        FailureCode.PENDING_CHANGES,
        FailureCode.STEP_BUDGET_EXHAUSTED,
        FailureCode.FALSE_COMPLETION,
        FailureCode.REPORT_INCONSISTENT,
    }
    assert expected <= set(result.failure_codes)


async def test_expected_invalid_request_is_successfully_rejected_and_classified() -> None:
    snapshot = _snapshot(
        running_revision=1,
        audit_log=[
            {
                "operation": "add_rule",
                "result": "error",
                "detail": "dst_port 非法: '70000'，端口范围需在 1-65535",
            }
        ],
    )

    result = await evaluate_run(
        assertions=[{"type": "revision", "op": "==", "value": 1}],
        snapshot=snapshot,
        report="参数非法，配置未完成",
        expect_success=False,
    )

    assert result.success is True
    assert result.correct_failure is True
    assert FailureCode.INVALID_ARGUMENT in result.failure_codes
    assert FailureCode.ASSERTION_FAILED not in result.failure_codes


async def test_expected_commit_rejection_records_causal_code() -> None:
    result = await evaluate_run(
        assertions=[{"type": "revision", "op": "==", "value": 1}],
        snapshot=_snapshot(
            running_revision=1,
            audit_log=[
                {
                    "operation": "commit",
                    "result": "error",
                    "detail": "配置提交被设备拒绝",
                }
            ],
        ),
        report="设备拒绝执行，变更失败",
        expect_success=False,
    )

    assert result.success is True
    assert FailureCode.COMMIT_REJECTED in result.failure_codes


async def test_successful_terminal_state_with_failure_report_is_classified() -> None:
    result = await evaluate_run(
        assertions=[{"type": "revision", "op": ">", "value": 1}],
        snapshot=_snapshot(),
        report="规则未能生效，建议重新操作",
        expect_success=True,
    )

    assert result.success is True
    assert result.false_failure is True
    assert FailureCode.FALSE_FAILURE in result.failure_codes
    assert FailureCode.REPORT_INCONSISTENT in result.failure_codes


async def test_traffic_probe_error_only_fails_traffic_assertion() -> None:
    async def broken_probe(_packet: dict) -> dict:
        raise RuntimeError("MCP unavailable")

    result = await evaluate_run(
        assertions=[
            {"type": "revision", "op": ">", "value": 1},
            {"type": "traffic", "packet": {}, "expect": "allow"},
        ],
        snapshot=_snapshot(),
        report="未能验证",
        expect_success=True,
        traffic_probe=broken_probe,
    )

    assert [assertion.passed for assertion in result.assertion_results] == [True, False]
    assert FailureCode.EVALUATOR_ERROR in result.failure_codes
    assert FailureCode.VERIFICATION_MISSING in result.failure_codes


def test_load_trace_supports_relative_path_and_missing_file(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"events": []}), encoding="utf-8")

    assert load_trace("trace.json", root=tmp_path) == {"events": []}
    assert load_trace("missing.json", root=tmp_path) is None
