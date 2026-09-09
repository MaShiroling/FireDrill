"""Pure, repeatable assertions for firewall Agent terminal state."""

from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable
from typing import Any

from app.evaluation.schemas import AssertionResult

TrafficProbe = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# Keep this heuristic stable so current and legacy experiment results remain comparable.
SUCCESS_WORDS = ("成功", "已放通", "已完成", "已生效", "放通", "生效")
FAIL_WORDS = ("未能", "无法", "失败", "未成功", "未完成", "没有完成", "拒绝执行")
STRONG_SUCCESS_WORDS = ("成功", "已完成", "已经完成", "顺利完成", "已生效", "已放通")


def claims_success(report: str) -> bool:
    if not report:
        return False
    has_fail = any(word in report for word in FAIL_WORDS)
    has_success = any(word in report for word in SUCCESS_WORDS)
    return has_success and not has_fail


def claims_failure(report: str) -> bool:
    if not report or not any(word in report for word in FAIL_WORDS):
        return False
    # “无法访问”可能正是 deny/delete 任务的预期结果；若同一报告有明确成功结论，
    # 不把它判成反向误报。先移除否定短语，避免“未成功”里的“成功”被误计。
    positive_context = report
    for word in FAIL_WORDS:
        positive_context = positive_context.replace(word, "")
    return not any(word in positive_context for word in STRONG_SUCCESS_WORDS)


def _normalize_address(value: str) -> str:
    if value == "any":
        return "any"
    try:
        return str(ipaddress.ip_network(str(value), strict=False))
    except ValueError:
        return str(value)


def _rule_matches(rule: dict[str, Any], match: dict[str, Any]) -> bool:
    for key, expected in match.items():
        actual = rule.get(key)
        if key in ("src_addr", "dst_addr"):
            if _normalize_address(str(actual)) != _normalize_address(str(expected)):
                return False
        elif str(actual) != str(expected):
            return False
    return True


async def evaluate_assertions(
    assertions: list[dict[str, Any]],
    snapshot: dict[str, Any],
    report: str,
    traffic_probe: TrafficProbe | None = None,
) -> list[AssertionResult]:
    """Evaluate assertions sequentially without consulting an LLM."""
    rules = snapshot.get("running_rules", [])
    audit = snapshot.get("audit_log", [])
    results: list[AssertionResult] = []

    for assertion in assertions:
        assertion_type = assertion.get("type", "")
        passed = False
        detail = ""

        if assertion_type == "rule_present":
            scope = assertion.get("scope", "running")
            pool = rules if scope == "running" else snapshot.get("candidate_rules", [])
            matches = [rule for rule in pool if _rule_matches(rule, assertion["match"])]
            passed = bool(matches)
            detail = matches[0]["rule_id"] if matches else f"未找到匹配 {assertion['match']}"

        elif assertion_type == "rule_absent":
            scope = assertion.get("scope", "running")
            pool = rules if scope == "running" else snapshot.get("candidate_rules", [])
            matches = [rule for rule in pool if _rule_matches(rule, assertion["match"])]
            passed = not matches
            detail = "不存在" if passed else f"仍存在 {matches[0]['rule_id']}"

        elif assertion_type == "rule_field":
            matches = [rule for rule in rules if rule["rule_id"] == assertion["rule_id"]]
            if matches:
                actual = matches[0].get(assertion["field"])
                passed = actual == assertion["value"]
                detail = f"{assertion['rule_id']}.{assertion['field']}={actual}"
            else:
                detail = f"{assertion['rule_id']} 不存在"

        elif assertion_type == "rule_count":
            passed = len(rules) == assertion["value"]
            detail = f"实际 {len(rules)}"

        elif assertion_type == "revision":
            revision = snapshot.get("running_revision", 0)
            operation = assertion["op"]
            comparators = {
                ">": revision > assertion["value"],
                "==": revision == assertion["value"],
                ">=": revision >= assertion["value"],
            }
            passed = comparators.get(operation, False)
            detail = f"revision={revision}"

        elif assertion_type == "no_pending":
            passed = snapshot.get("pending_changes") is False
            detail = f"pending={snapshot.get('pending_changes')}"

        elif assertion_type == "first_rule":
            passed = bool(rules) and rules[0]["rule_id"] == assertion["rule_id"]
            detail = f"首位={rules[0]['rule_id'] if rules else '无'}"

        elif assertion_type == "traffic":
            if traffic_probe is None:
                detail = "缺少流量验证器"
            else:
                try:
                    probe_result = await traffic_probe(assertion["packet"])
                    actual = probe_result.get("action")
                    passed = actual == assertion["expect"]
                    detail = f"action={actual}（期望 {assertion['expect']}）"
                except Exception as exc:
                    detail = f"流量验证器异常: {str(exc)[:200]}"

        elif assertion_type == "hit":
            count = snapshot.get("hit_counts", {}).get(assertion["rule_id"], 0)
            passed = count >= assertion["min"]
            detail = f"hit={count}"

        elif assertion_type == "report_contains":
            passed = assertion["value"].lower() in report.lower()
            detail = f"报告{'包含' if passed else '不含'} '{assertion['value']}'"

        elif assertion_type == "recheck_after_failed_commit":
            read_operations = {
                "get_firewall_overview",
                "list_firewall_rules",
                "get_firewall_rule",
                "get_config_diff",
                "test_traffic",
            }
            failed_commit_index = next(
                (
                    index
                    for index, event in enumerate(audit)
                    if event.get("operation") == "commit" and event.get("result") == "error"
                ),
                None,
            )
            passed = failed_commit_index is not None and any(
                event.get("operation") in read_operations
                for event in audit[failed_commit_index + 1 :]
            )
            detail = "失败 commit 后" + ("有核实动作" if passed else "无核实动作")

        else:
            detail = f"未知断言类型 {assertion_type}"

        results.append(AssertionResult(assertion=assertion, passed=passed, detail=detail))

    return results
