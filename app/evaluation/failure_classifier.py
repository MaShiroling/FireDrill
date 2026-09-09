"""Rule-based failure classification using trace and out-of-band evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.evaluation.schemas import AssertionResult, FailureCode

_INVALID_ARGUMENT_MARKERS = (
    "非法",
    "无效",
    "格式",
    "端口范围",
    "不存在",
    "重复",
    "没有任何候选改动",
)
_VERIFY_ASSERTIONS = {"traffic", "hit", "recheck_after_failed_commit"}


def _add(evidence: defaultdict[FailureCode, list[str]], code: FailureCode, detail: str) -> None:
    if detail not in evidence[code]:
        evidence[code].append(detail)


def classify_failures(
    *,
    success: bool,
    expect_success: bool,
    fake_completion: bool,
    false_failure: bool,
    assertion_results: list[AssertionResult],
    snapshot: dict[str, Any],
    trace: dict[str, Any] | None,
    run_error: str,
    steps: int,
) -> tuple[list[FailureCode], dict[str, list[str]]]:
    """Return stable labels plus compact, inspectable evidence for each label."""
    evidence: defaultdict[FailureCode, list[str]] = defaultdict(list)
    failed_assertions = [result for result in assertion_results if not result.passed]

    if run_error:
        _add(evidence, FailureCode.RUN_ERROR, run_error[:200])
    if failed_assertions:
        for result in failed_assertions:
            assertion_type = str(result.assertion.get("type", "unknown"))
            _add(
                evidence,
                FailureCode.ASSERTION_FAILED,
                f"{assertion_type}: {result.detail}",
            )
            if assertion_type in _VERIFY_ASSERTIONS:
                _add(
                    evidence,
                    FailureCode.VERIFICATION_MISSING,
                    f"{assertion_type}: {result.detail}",
                )
            if assertion_type == "traffic" and "验证器" in result.detail:
                _add(evidence, FailureCode.EVALUATOR_ERROR, result.detail)
            if assertion_type == "report_contains":
                _add(evidence, FailureCode.REPORT_INCONSISTENT, result.detail)

    if fake_completion:
        _add(evidence, FailureCode.FALSE_COMPLETION, "报告声称成功，但终态断言未通过")
        _add(evidence, FailureCode.REPORT_INCONSISTENT, "报告结论与带外终态不一致")

    if false_failure:
        _add(evidence, FailureCode.FALSE_FAILURE, "带外终态成功，但报告声称执行失败")
        _add(evidence, FailureCode.REPORT_INCONSISTENT, "报告结论与带外终态不一致")

    if snapshot.get("pending_changes") and not success:
        _add(evidence, FailureCode.PENDING_CHANGES, "运行结束时仍有候选配置未提交")

    audit_errors = [
        event for event in snapshot.get("audit_log", []) if event.get("result") == "error"
    ]
    inspect_audit_errors = not success or not expect_success
    if inspect_audit_errors:
        for event in audit_errors:
            operation = str(event.get("operation", "unknown"))
            detail = str(event.get("detail", ""))
            if any(marker in detail for marker in _INVALID_ARGUMENT_MARKERS):
                _add(evidence, FailureCode.INVALID_ARGUMENT, f"{operation}: {detail}")
            if operation == "commit" and ("拒绝" in detail or "commit_reject" in detail):
                _add(evidence, FailureCode.COMMIT_REJECTED, detail)

    failed_recheck = any(
        not result.passed and result.assertion.get("type") == "recheck_after_failed_commit"
        for result in assertion_results
    )
    commit_unknown = any(
        event.get("operation") == "commit"
        and event.get("result") == "error"
        and (
            "状态未知" in str(event.get("detail", ""))
            or "实际已生效" in str(event.get("detail", ""))
        )
        for event in snapshot.get("audit_log", [])
    )
    if commit_unknown and failed_recheck:
        _add(evidence, FailureCode.COMMIT_STATE_UNKNOWN, "提交返回状态未知，且 Agent 未带外核实")

    events = trace.get("events", []) if trace else []
    if not success:
        for event in events:
            event_type = event.get("event_type")
            node = event.get("node")
            data = event.get("data", {})
            if event_type == "node_failed" and node == "planner":
                _add(
                    evidence,
                    FailureCode.PLANNING_FAILURE,
                    str(data.get("error", "Planner 失败并使用兜底计划"))[:200],
                )
            elif event_type == "model_call_failed":
                _add(
                    evidence,
                    FailureCode.MODEL_CALL_FAILURE,
                    f"{node or 'unknown'}/{data.get('purpose', 'unknown')}: "
                    f"{str(data.get('error', '模型调用失败'))[:160]}",
                )
            elif event_type == "model_retry_exhausted":
                _add(
                    evidence,
                    FailureCode.MODEL_RETRY_EXHAUSTED,
                    f"{node or 'unknown'}/{data.get('purpose', 'unknown')}: "
                    f"{data.get('attempts', 0)} 次尝试耗尽",
                )
            elif event_type == "no_tool_selected":
                _add(
                    evidence,
                    FailureCode.TOOL_SELECTION_FAILURE,
                    str(data.get("step", "执行步骤未选择工具"))[:200],
                )
            elif event_type == "tool_call_completed" and str(data.get("status", "")).lower() in {
                "error",
                "failed",
                "failure",
            }:
                _add(
                    evidence,
                    FailureCode.TOOL_EXECUTION_FAILURE,
                    f"{data.get('name', 'unknown')}: {str(data.get('content', ''))[:160]}",
                )
            elif event_type == "mcp_attempt_completed" and data.get("returned_error_result"):
                _add(
                    evidence,
                    FailureCode.TOOL_EXECUTION_FAILURE,
                    f"{data.get('tool', 'unknown')}: MCP 返回错误结果",
                )
            elif event_type == "mcp_call_exhausted":
                _add(
                    evidence,
                    FailureCode.RETRY_EXHAUSTED,
                    f"{data.get('tool', 'unknown')}: {data.get('attempts', 0)} 次尝试耗尽",
                )
            elif (
                event_type == "mcp_retry_skipped"
                and data.get("reason") == "ambiguous"
                and data.get("tool") == "commit_config"
            ):
                _add(
                    evidence,
                    FailureCode.COMMIT_STATE_UNKNOWN,
                    f"{data.get('tool', 'unknown')}: 结果不确定，已停止盲目重试",
                )
            elif event_type == "replanner_decision" and data.get("reason") == "max_steps_reached":
                _add(
                    evidence,
                    FailureCode.STEP_BUDGET_EXHAUSTED,
                    f"已执行 {data.get('completed_steps', steps)} 步，达到上限 {data.get('max_steps', 8)}",
                )

        if steps >= 8 and FailureCode.STEP_BUDGET_EXHAUSTED not in evidence:
            _add(evidence, FailureCode.STEP_BUDGET_EXHAUSTED, f"已执行 {steps} 步")

    ordered_codes = [code for code in FailureCode if code in evidence]
    serialized_evidence = {code.value: evidence[code] for code in ordered_codes}
    return ordered_codes, serialized_evidence
