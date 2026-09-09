"""Compare replay outcomes with mined failure signatures and enforce a gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.schemas import FailureCode
from app.flywheel.builder import derive_failure_codes


@dataclass(frozen=True)
class GateThresholds:
    min_runs_per_case: int = 3
    min_healthy_rate: float = 0.66
    max_target_recurrence_rate: float = 0.34
    max_fake_completion_rate: float = 0.10
    max_false_failure_rate: float = 0.10
    max_run_error_rate: float = 0.05
    require_all_cases: bool = False
    fail_on_new_codes: bool = False

    def validate(self) -> None:
        if self.min_runs_per_case < 1:
            raise ValueError("min_runs_per_case 必须大于 0")
        for name, value in asdict(self).items():
            if name.startswith(("min_", "max_")) and name != "min_runs_per_case":
                if not 0 <= value <= 1:
                    raise ValueError(f"{name} 必须在 0 到 1 之间")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _target_metadata(
    record: dict[str, Any], replay_catalog: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    metadata = record.get("flywheel")
    if isinstance(metadata, dict) and metadata.get("target_failure_codes") is not None:
        return metadata
    replay_case = replay_catalog.get(str(record.get("case_id")), {})
    fallback = replay_case.get("flywheel", {})
    return fallback if isinstance(fallback, dict) else {}


def _is_healthy(record: dict[str, Any], codes: set[str]) -> bool:
    return (
        bool(record.get("passed"))
        and not bool(record.get("fake_complete"))
        and FailureCode.FALSE_FAILURE.value not in codes
        and FailureCode.RUN_ERROR.value not in codes
        and FailureCode.EVALUATOR_ERROR.value not in codes
    )


def analyze_replay_regression(
    records: list[dict[str, Any]],
    replay_catalog: dict[str, dict[str, Any]],
    thresholds: GateThresholds,
    *,
    name: str = "replay-regression",
) -> dict[str, Any]:
    """Produce a deterministic gate report for one replay experiment."""
    thresholds.validate()
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    unlinked_records: list[dict[str, Any]] = []

    for record in records:
        metadata = _target_metadata(record, replay_catalog)
        target_codes = metadata.get("target_failure_codes")
        if not isinstance(target_codes, list):
            unlinked_records.append(record)
            continue
        current_codes = set(derive_failure_codes(record))
        target_set = {str(code) for code in target_codes}
        grouped[str(record.get("case_id"))].append(
            {
                "record": record,
                "metadata": metadata,
                "current_codes": current_codes,
                "target_codes": target_set,
                "target_recurred": bool(current_codes & target_set),
                "new_codes": current_codes - target_set,
                "healthy": _is_healthy(record, current_codes),
            }
        )

    case_reports: list[dict[str, Any]] = []
    all_current_codes: Counter[str] = Counter()
    all_new_codes: Counter[str] = Counter()
    total_runs = healthy_runs = target_recurrences = 0
    fake_completions = false_failures = run_errors = 0

    for case_id, items in sorted(grouped.items()):
        run_count = len(items)
        healthy_count = sum(item["healthy"] for item in items)
        target_recurrence_count = sum(item["target_recurred"] for item in items)
        fake_count = sum(bool(item["record"].get("fake_complete")) for item in items)
        false_failure_count = sum(
            FailureCode.FALSE_FAILURE.value in item["current_codes"] for item in items
        )
        run_error_count = sum(
            FailureCode.RUN_ERROR.value in item["current_codes"] for item in items
        )
        target_codes = sorted(set().union(*(item["target_codes"] for item in items)))
        current_code_counts = Counter(code for item in items for code in item["current_codes"])
        new_code_counts = Counter(code for item in items for code in item["new_codes"])
        for code, count in current_code_counts.items():
            all_current_codes[code] += count
        for code, count in new_code_counts.items():
            all_new_codes[code] += count

        if run_count < thresholds.min_runs_per_case:
            status = "insufficient_runs"
        elif healthy_count == 0:
            status = "not_recovered"
        elif healthy_count == run_count and target_recurrence_count == 0:
            status = "recovered"
        else:
            status = "unstable"

        metadata = items[0]["metadata"]
        case_reports.append(
            {
                "case_id": case_id,
                "sample_id": metadata.get("sample_id"),
                "source_case_id": metadata.get("source_case_id"),
                "priority_score": metadata.get("priority_score"),
                "runs": run_count,
                "healthy_runs": healthy_count,
                "healthy_rate": _rate(healthy_count, run_count),
                "target_failure_codes": target_codes,
                "target_recurrence_runs": target_recurrence_count,
                "target_recurrence_rate": _rate(target_recurrence_count, run_count),
                "target_resolved_rate": _rate(run_count - target_recurrence_count, run_count),
                "fake_completion_runs": fake_count,
                "false_failure_runs": false_failure_count,
                "run_error_runs": run_error_count,
                "current_failure_code_counts": dict(sorted(current_code_counts.items())),
                "new_failure_code_counts": dict(sorted(new_code_counts.items())),
                "status": status,
            }
        )
        total_runs += run_count
        healthy_runs += healthy_count
        target_recurrences += target_recurrence_count
        fake_completions += fake_count
        false_failures += false_failure_count
        run_errors += run_error_count

    observed_case_ids = set(grouped)
    expected_case_ids = set(replay_catalog)
    missing_case_ids = sorted(expected_case_ids - observed_case_ids)
    summary = {
        "observed_case_count": len(case_reports),
        "expected_case_count": len(expected_case_ids),
        "missing_case_count": len(missing_case_ids),
        "unlinked_record_count": len(unlinked_records),
        "total_runs": total_runs,
        "healthy_runs": healthy_runs,
        "healthy_rate": _rate(healthy_runs, total_runs),
        "target_recurrence_runs": target_recurrences,
        "target_recurrence_rate": _rate(target_recurrences, total_runs),
        "target_resolved_rate": _rate(total_runs - target_recurrences, total_runs),
        "fake_completion_runs": fake_completions,
        "fake_completion_rate": _rate(fake_completions, total_runs),
        "false_failure_runs": false_failures,
        "false_failure_rate": _rate(false_failures, total_runs),
        "run_error_runs": run_errors,
        "run_error_rate": _rate(run_errors, total_runs),
        "fully_recovered_case_count": sum(
            report["status"] == "recovered" for report in case_reports
        ),
        "not_recovered_case_count": sum(
            report["status"] == "not_recovered" for report in case_reports
        ),
        "unstable_case_count": sum(report["status"] == "unstable" for report in case_reports),
        "insufficient_case_count": sum(
            report["status"] == "insufficient_runs" for report in case_reports
        ),
    }

    violations: list[dict[str, Any]] = []

    def violate(metric: str, actual: Any, threshold: Any, message: str) -> None:
        violations.append(
            {
                "metric": metric,
                "actual": actual,
                "threshold": threshold,
                "message": message,
            }
        )

    if not total_runs:
        violate("total_runs", 0, "> 0", "没有可关联到回放样本的评测记录")
    if unlinked_records:
        violate(
            "unlinked_record_count",
            len(unlinked_records),
            0,
            "部分结果缺少 flywheel.target_failure_codes，无法进行目标失败对比",
        )
    insufficient = [
        report["case_id"]
        for report in case_reports
        if report["runs"] < thresholds.min_runs_per_case
    ]
    if insufficient:
        violate(
            "min_runs_per_case",
            min(report["runs"] for report in case_reports),
            thresholds.min_runs_per_case,
            f"{len(insufficient)} 个回放用例运行次数不足",
        )
    zero_recovery = [
        report["case_id"]
        for report in case_reports
        if report["runs"] >= thresholds.min_runs_per_case and report["healthy_runs"] == 0
    ]
    if zero_recovery:
        violate(
            "zero_recovery_cases",
            len(zero_recovery),
            0,
            f"{len(zero_recovery)} 个用例一次健康恢复都没有",
        )
    if summary["healthy_rate"] < thresholds.min_healthy_rate:
        violate(
            "healthy_rate",
            summary["healthy_rate"],
            f">= {thresholds.min_healthy_rate}",
            "整体健康恢复率未达标",
        )
    if summary["target_recurrence_rate"] > thresholds.max_target_recurrence_rate:
        violate(
            "target_recurrence_rate",
            summary["target_recurrence_rate"],
            f"<= {thresholds.max_target_recurrence_rate}",
            "原目标失败仍频繁复现",
        )
    if summary["fake_completion_rate"] > thresholds.max_fake_completion_rate:
        violate(
            "fake_completion_rate",
            summary["fake_completion_rate"],
            f"<= {thresholds.max_fake_completion_rate}",
            "假完成率超过门限",
        )
    if summary["false_failure_rate"] > thresholds.max_false_failure_rate:
        violate(
            "false_failure_rate",
            summary["false_failure_rate"],
            f"<= {thresholds.max_false_failure_rate}",
            "反向误报率超过门限",
        )
    if summary["run_error_rate"] > thresholds.max_run_error_rate:
        violate(
            "run_error_rate",
            summary["run_error_rate"],
            f"<= {thresholds.max_run_error_rate}",
            "运行异常率超过门限",
        )
    if thresholds.require_all_cases and missing_case_ids:
        violate(
            "missing_case_count",
            len(missing_case_ids),
            0,
            "回放数据集存在未执行用例",
        )
    if thresholds.fail_on_new_codes and all_new_codes:
        violate(
            "new_failure_codes",
            dict(sorted(all_new_codes.items())),
            {},
            "回放产生了原失败签名之外的新失败码",
        )

    return {
        "schema_version": "1.0",
        "name": name,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed" if not violations else "failed",
        "thresholds": asdict(thresholds),
        "summary": summary,
        "failure_code_counts": dict(sorted(all_current_codes.items())),
        "new_failure_code_counts": dict(sorted(all_new_codes.items())),
        "missing_case_ids": missing_case_ids,
        "unlinked_records": [
            {
                "case_id": record.get("case_id"),
                "run": record.get("run"),
                "source_file": record.get("_source_file"),
                "source_line": record.get("_source_line"),
            }
            for record in unlinked_records
        ],
        "violations": violations,
        "cases": case_reports,
    }


def render_regression_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    status = "PASS" if report["status"] == "passed" else "FAIL"
    lines = [
        f"# Replay regression gate: {report['name']}",
        "",
        f"**Status: {status}**",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Runs | {summary['total_runs']} |",
        f"| Healthy recovery | {summary['healthy_runs']}/{summary['total_runs']} ({summary['healthy_rate']:.1%}) |",
        f"| Target failure recurrence | {summary['target_recurrence_runs']}/{summary['total_runs']} ({summary['target_recurrence_rate']:.1%}) |",
        f"| Fake completion | {summary['fake_completion_runs']}/{summary['total_runs']} ({summary['fake_completion_rate']:.1%}) |",
        f"| False failure | {summary['false_failure_runs']}/{summary['total_runs']} ({summary['false_failure_rate']:.1%}) |",
        f"| Run error | {summary['run_error_runs']}/{summary['total_runs']} ({summary['run_error_rate']:.1%}) |",
        f"| Cases: recovered / unstable / not recovered / insufficient | {summary['fully_recovered_case_count']} / {summary['unstable_case_count']} / {summary['not_recovered_case_count']} / {summary['insufficient_case_count']} |",
        "",
        "## Gate violations",
        "",
    ]
    if report["violations"]:
        lines.extend(f"- `{item['metric']}`: {item['message']}" for item in report["violations"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Source | Runs | Healthy | Target recurrence | New codes | Status |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for case in report["cases"]:
        new_codes = ", ".join(case["new_failure_code_counts"]) or "-"
        lines.append(
            f"| {case['case_id']} | {case.get('source_case_id') or '-'} | {case['runs']} | "
            f"{case['healthy_rate']:.0%} | {case['target_recurrence_rate']:.0%} | "
            f"{new_codes} | {case['status']} |"
        )
    return "\n".join(lines) + "\n"


def write_regression_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "regression_report.json",
        "markdown": output_dir / "regression_report.md",
    }
    payloads = {
        "json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        "markdown": render_regression_markdown(report),
    }
    for name, path in paths.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payloads[name], encoding="utf-8")
        temporary.replace(path)
    return paths
