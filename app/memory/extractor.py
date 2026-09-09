"""Deterministic extraction of reviewable memories from flywheel artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.memory.repository import MemoryRepository
from app.memory.schemas import MemoryNamespace, MemoryRecord, MemoryStatus, MemoryType


@dataclass(frozen=True)
class FailureLessonRule:
    lesson: str
    actions: tuple[str, ...]


# Ordered by operational risk.  The first matched rule supplies the concise lesson;
# actions from every matched rule are retained in priority order.
_FAILURE_LESSON_RULES: tuple[tuple[str, FailureLessonRule], ...] = (
    (
        "commit_state_unknown",
        FailureLessonRule(
            lesson=(
                "提交结果未知时不得更换幂等键盲目重试；先读取 Running Revision 和配置差异，"
                "确认真实状态后再决定继续提交或结束任务。"
            ),
            actions=("get_firewall_overview", "get_config_diff"),
        ),
    ),
    (
        "false_completion",
        FailureLessonRule(
            lesson=(
                "只有候选配置已下发、提交成功且终态验证通过后才能宣布任务完成；"
                "任一环节缺失都应明确报告未完成。"
            ),
            actions=(
                "get_config_diff",
                "commit_config",
                "get_firewall_overview",
                "test_traffic",
            ),
        ),
    ),
    (
        "verification_missing",
        FailureLessonRule(
            lesson=(
                "配置变更后必须核对 Running Revision、待提交差异和业务流量，"
                "不能仅凭工具返回或候选配置判断已经生效。"
            ),
            actions=("get_firewall_overview", "get_config_diff", "test_traffic"),
        ),
    ),
    (
        "pending_changes",
        FailureLessonRule(
            lesson="任务结束前必须检查 pending_changes；仍有候选差异时不得报告变更已完成。",
            actions=("get_config_diff", "get_firewall_overview"),
        ),
    ),
    (
        "invalid_argument",
        FailureLessonRule(
            lesson=(
                "变更参数必须来自工具返回的真实设备状态，尤其不能把规则名称当作 rule_id，"
                "参数校验失败后应重新查询再修正。"
            ),
            actions=("list_firewall_rules", "get_firewall_rule"),
        ),
    ),
    (
        "commit_rejected",
        FailureLessonRule(
            lesson=(
                "设备明确拒绝提交属于永久失败，不应自动重试；应保留拒绝原因，"
                "并根据任务意图保留或放弃候选配置。"
            ),
            actions=("get_config_diff", "discard_candidate"),
        ),
    ),
    (
        "retry_exhausted",
        FailureLessonRule(
            lesson=("临时错误只能进行有限重试；重试耗尽后应停止变更，保留最后一次错误和设备状态。"),
            actions=("get_firewall_overview", "get_config_diff"),
        ),
    ),
    (
        "tool_execution_failure",
        FailureLessonRule(
            lesson=(
                "工具失败后先区分临时错误与永久错误；涉及副作用的调用状态不明确时，"
                "必须读取真实状态后再决策。"
            ),
            actions=("get_firewall_overview", "get_config_diff"),
        ),
    ),
    (
        "tool_selection_failure",
        FailureLessonRule(
            lesson="执行计划时必须选择能够返回真实设备状态的工具，不能猜测配置或对象标识。",
            actions=("get_firewall_overview", "list_firewall_rules"),
        ),
    ),
    (
        "step_budget_exhausted",
        FailureLessonRule(
            lesson=(
                "接近步骤预算时应优先完成提交和终态验证；无法闭环时应如实报告未完成，"
                "不能用总结替代剩余操作。"
            ),
            actions=("get_config_diff", "commit_config", "get_firewall_overview"),
        ),
    ),
    (
        "false_failure",
        FailureLessonRule(
            lesson=(
                "最终报告必须以终态硬断言为准；如果配置和业务验证已经通过，"
                "应报告成功并将中途的临时错误说明为已恢复。"
            ),
            actions=("get_firewall_overview", "get_config_diff", "test_traffic"),
        ),
    ),
    (
        "report_inconsistent",
        FailureLessonRule(
            lesson="最终报告必须与 Running 配置、待提交差异和业务验证结果保持一致。",
            actions=("get_firewall_overview", "get_config_diff"),
        ),
    ),
    (
        "planning_failure",
        FailureLessonRule(
            lesson="变更计划应明确包含读取现状、修改候选配置、提交和终态验证四个阶段。",
            actions=("get_firewall_overview", "get_config_diff", "commit_config"),
        ),
    ),
)

_ASSERTION_ACTIONS: dict[str, tuple[str, ...]] = {
    "rule_present": ("list_firewall_rules",),
    "rule_absent": ("list_firewall_rules",),
    "revision": ("get_firewall_overview",),
    "no_pending": ("get_config_diff",),
    "traffic": ("test_traffic",),
    "hit": ("get_rule_hit_count",),
    "recheck_after_failed_commit": ("get_firewall_overview", "get_config_diff"),
}


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _failure_run_ids(sample: Mapping[str, Any]) -> tuple[str, ...]:
    case_id = str(sample.get("source_case_id") or "unknown-case")
    identifiers: list[str] = []
    for occurrence_value in _as_list(sample.get("occurrences")):
        occurrence = _as_dict(occurrence_value)
        trace_id = str(occurrence.get("trace_id") or "").strip()
        if trace_id:
            identifiers.append(trace_id)
            continue
        source_tag = str(occurrence.get("source_tag") or "evaluation").strip()
        run = str(occurrence.get("run") or "unknown").strip()
        identifiers.append(f"{source_tag}:{case_id}:run-{run}")
    return _ordered_unique(identifiers)


def _failure_confidence(sample: Mapping[str, Any]) -> float:
    occurrences = max(1, int(sample.get("occurrence_count", 1) or 1))
    priority = min(100.0, max(0.0, float(sample.get("priority_score", 0) or 0)))
    return round(min(0.95, 0.45 + min(occurrences, 5) * 0.08 + priority * 0.001), 2)


def _failure_preconditions(sample: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "category": str(sample.get("category") or "unknown"),
        "expect_success": bool(sample.get("expect_success", True)),
        "failed_assertion_types": sorted(
            str(value) for value in _as_list(sample.get("failed_assertion_types"))
        ),
    }
    scenario = sample.get("scenario")
    if isinstance(scenario, (dict, list, str, int, float, bool)):
        result["fault_scenario"] = scenario
    return result


def extract_failure_memory(
    sample: Mapping[str, Any],
    *,
    tenant_id: str,
    device_type: str,
    source_revision: str = "unknown",
) -> MemoryRecord | None:
    """Convert one failure-pool sample to a candidate, or skip noisy evidence."""
    codes = _ordered_unique(str(code) for code in _as_list(sample.get("failure_codes")))
    matched = [rule for code, rule in _FAILURE_LESSON_RULES if code in codes]
    if not matched:
        return None

    scenario = str(sample.get("task") or sample.get("category") or "未命名运维任务").strip()
    actions = _ordered_unique(action for rule in matched for action in rule.actions)
    source_case_id = str(sample.get("source_case_id") or "").strip()
    return MemoryRecord.create(
        namespace=MemoryNamespace(
            tenant_id=tenant_id,
            device_type=device_type,
            memory_type=MemoryType.FAILURE_LESSON,
        ),
        scenario=scenario,
        lesson=matched[0].lesson,
        preconditions=_failure_preconditions(sample),
        recommended_actions=actions,
        failure_codes=codes,
        evidence_run_ids=_failure_run_ids(sample),
        evidence_case_ids=(source_case_id,) if source_case_id else (),
        source_revision=source_revision,
        confidence=_failure_confidence(sample),
        status=MemoryStatus.CANDIDATE,
    )


def extract_failure_memories(
    samples: Iterable[Mapping[str, Any]],
    *,
    tenant_id: str,
    device_type: str,
    source_revision: str = "unknown",
) -> list[MemoryRecord]:
    candidates = (
        extract_failure_memory(
            sample,
            tenant_id=tenant_id,
            device_type=device_type,
            source_revision=source_revision,
        )
        for sample in samples
    )
    return [candidate for candidate in candidates if candidate is not None]


def _case_actions(case: Mapping[str, Any]) -> tuple[str, ...]:
    task = str(case.get("task") or "")
    actions: list[str] = ["get_firewall_overview", "list_firewall_rules"]
    if any(word in task for word in ("删除", "移除")):
        actions.append("delete_firewall_rule")
    elif any(word in task for word in ("修改", "更新", "禁用", "启用")):
        actions.append("update_firewall_rule")
    elif any(word in task for word in ("移动", "顺序", "优先级")):
        actions.append("move_firewall_rule")
    elif any(word in task for word in ("放通", "阻断", "新增", "添加", "创建")):
        actions.append("add_firewall_rule")

    assertion_types: list[str] = []
    for assertion_value in _as_list(case.get("assert")):
        assertion_type = str(_as_dict(assertion_value).get("type") or "")
        assertion_types.append(assertion_type)
    if "revision" in assertion_types or any(action.endswith("firewall_rule") for action in actions):
        actions.extend(("get_config_diff", "commit_config"))
    for assertion_type in assertion_types:
        actions.extend(_ASSERTION_ACTIONS.get(assertion_type, ()))
    return _ordered_unique(actions)


def _successful_lesson(case: Mapping[str, Any], report_case: Mapping[str, Any]) -> str:
    scenario = _as_dict(case.get("scenario"))
    fault = str(scenario.get("fault") or "")
    prefix = "该任务的回放已经通过终态断言。"
    if fault == "commit_flaky":
        prefix = "在提交暂时失败的场景中，该任务通过有限重试恢复并通过终态断言。"
    elif fault == "commit_lose":
        prefix = "在提交结果不确定的场景中，该任务通过读取真实状态完成判定并通过终态断言。"
    return (
        f"{prefix}复用时仍需遵循“读取现状—修改候选配置—检查差异—提交—终态验证”的闭环，"
        f"不能因为历史 {int(report_case.get('healthy_runs', 0) or 0)} 次成功而跳过验证。"
    )


def extract_recovered_memories(
    regression_report: Mapping[str, Any],
    replay_catalog: Mapping[str, Mapping[str, Any]],
    *,
    tenant_id: str,
    device_type: str,
    source_revision: str = "unknown",
) -> list[MemoryRecord]:
    """Create successful-case candidates only from fully recovered replay cases."""
    candidates: list[MemoryRecord] = []
    for value in _as_list(regression_report.get("cases")):
        report_case = _as_dict(value)
        if report_case.get("status") != "recovered":
            continue
        case_id = str(report_case.get("case_id") or "").strip()
        case = replay_catalog.get(case_id)
        if not case or not str(case.get("task") or "").strip():
            continue

        assertions = [_as_dict(item) for item in _as_list(case.get("assert"))]
        assertion_types = sorted({str(item.get("type")) for item in assertions if item.get("type")})
        flywheel = _as_dict(case.get("flywheel"))
        source_case_id = str(
            report_case.get("source_case_id") or flywheel.get("source_case_id") or ""
        ).strip()
        healthy_runs = int(report_case.get("healthy_runs", 0) or 0)
        runs = max(1, int(report_case.get("runs", 1) or 1))
        healthy_rate = min(1.0, max(0.0, healthy_runs / runs))
        confidence = round(min(0.95, 0.5 + min(healthy_runs, 4) * 0.1) * healthy_rate, 2)
        evidence_case_ids = _ordered_unique((case_id, source_case_id))
        candidates.append(
            MemoryRecord.create(
                namespace=MemoryNamespace(
                    tenant_id=tenant_id,
                    device_type=device_type,
                    memory_type=MemoryType.SUCCESSFUL_CASE,
                ),
                scenario=str(case["task"]),
                lesson=_successful_lesson(case, report_case),
                preconditions={
                    "category": str(case.get("category") or "unknown"),
                    "fault_scenario": case.get("scenario"),
                    "expected_assertion_types": assertion_types,
                    "recovered_from": sorted(
                        str(code) for code in _as_list(report_case.get("target_failure_codes"))
                    ),
                },
                recommended_actions=_case_actions(case),
                failure_codes=tuple(
                    str(code) for code in _as_list(report_case.get("target_failure_codes"))
                ),
                evidence_case_ids=evidence_case_ids,
                source_revision=source_revision,
                confidence=confidence,
                success_count=healthy_runs,
                contradiction_count=int(report_case.get("target_recurrence_runs", 0) or 0),
                status=MemoryStatus.CANDIDATE,
            )
        )
    return candidates


def persist_memory_candidates(
    repository: MemoryRepository, candidates: Iterable[MemoryRecord]
) -> list[MemoryRecord]:
    """Persist a batch and return canonical records after repository merging."""
    return [repository.upsert(candidate) for candidate in candidates]
