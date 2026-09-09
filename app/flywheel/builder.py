"""Build a deduplicated failure pool and executable replay cases."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.deterministic import claims_failure
from app.evaluation.schemas import FailureCode

_TRACE_EVENTS = {
    "node_failed",
    "model_call_failed",
    "model_retry_scheduled",
    "model_retry_skipped",
    "model_retry_exhausted",
    "no_tool_selected",
    "tool_call_requested",
    "tool_call_completed",
    "mcp_retry_scheduled",
    "mcp_retry_skipped",
    "mcp_call_exhausted",
    "replanner_decision",
    "run_finished",
}
_HANDLED_ONLY_CODES = {
    FailureCode.INVALID_ARGUMENT.value,
    FailureCode.COMMIT_REJECTED.value,
}


def load_case_catalog(path: Path) -> dict[str, dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"用例文件必须是 JSON 数组: {path}")
    return {str(case["id"]): case for case in cases}


def load_result_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"结果文件 JSON 无效: {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"结果记录必须是对象: {path}:{line_number}")
            records.append(
                {
                    **record,
                    "_source_file": str(path),
                    "_source_tag": path.stem,
                    "_source_line": line_number,
                }
            )
    return records


def derive_failure_codes(record: dict[str, Any]) -> list[str]:
    """Backfill useful labels for historical results created before schema 1.0."""
    valid_codes = {code.value for code in FailureCode}
    codes = {str(code) for code in record.get("failure_codes", []) if str(code) in valid_codes}
    failed_assertions = [item for item in record.get("asserts", []) if not item.get("pass")]

    if record.get("error"):
        codes.add(FailureCode.RUN_ERROR.value)
    if failed_assertions:
        codes.add(FailureCode.ASSERTION_FAILED.value)
    if any(
        item.get("type") in {"traffic", "hit", "recheck_after_failed_commit"}
        for item in failed_assertions
    ):
        codes.add(FailureCode.VERIFICATION_MISSING.value)
    if any(
        item.get("type") == "no_pending" and "True" in str(item.get("detail", ""))
        for item in failed_assertions
    ):
        codes.add(FailureCode.PENDING_CHANGES.value)

    fake_completion = bool(record.get("fake_complete")) or (
        bool(record.get("claims_success")) and not bool(record.get("passed"))
    )
    if fake_completion:
        codes.add(FailureCode.FALSE_COMPLETION.value)
        codes.add(FailureCode.REPORT_INCONSISTENT.value)

    report = str(record.get("report_tail", ""))
    false_failure = bool(record.get("false_failure")) or (
        bool(record.get("expect_success")) and bool(record.get("passed")) and claims_failure(report)
    )
    if false_failure:
        codes.add(FailureCode.FALSE_FAILURE.value)
        codes.add(FailureCode.REPORT_INCONSISTENT.value)

    if not record.get("passed") and int(record.get("steps", 0) or 0) >= 8:
        codes.add(FailureCode.STEP_BUDGET_EXHAUSTED.value)

    return [code.value for code in FailureCode if code.value in codes]


def _is_candidate(record: dict[str, Any], codes: list[str], include_handled: bool) -> bool:
    model_failure = (
        not bool(record.get("passed"))
        or bool(record.get("fake_complete"))
        or FailureCode.FALSE_FAILURE.value in codes
    )
    if model_failure:
        return True
    return include_handled and bool(set(codes) & _HANDLED_ONLY_CODES)


def _load_trace(record: dict[str, Any], root: Path) -> dict[str, Any] | None:
    raw_path = record.get("trace_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = root / path
    try:
        trace = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return trace if isinstance(trace, dict) else None


def _compact_trajectory(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace:
        return []
    return [
        {
            "sequence": event.get("sequence"),
            "event_type": event.get("event_type"),
            "node": event.get("node"),
            "data": event.get("data", {}),
        }
        for event in trace.get("events", [])
        if event.get("event_type") in _TRACE_EVENTS
    ]


def _priority(record: dict[str, Any], codes: list[str], trace_available: bool) -> int:
    weights = {
        FailureCode.FALSE_COMPLETION.value: 35,
        FailureCode.FALSE_FAILURE.value: 30,
        FailureCode.RUN_ERROR.value: 25,
        FailureCode.MODEL_CALL_FAILURE.value: 20,
        FailureCode.MODEL_RETRY_EXHAUSTED.value: 25,
        FailureCode.RETRY_EXHAUSTED.value: 22,
        FailureCode.COMMIT_STATE_UNKNOWN.value: 22,
        FailureCode.PLANNING_FAILURE.value: 18,
        FailureCode.TOOL_SELECTION_FAILURE.value: 18,
        FailureCode.REPORT_INCONSISTENT.value: 15,
        FailureCode.VERIFICATION_MISSING.value: 15,
        FailureCode.PENDING_CHANGES.value: 12,
        FailureCode.STEP_BUDGET_EXHAUSTED.value: 12,
        FailureCode.TOOL_EXECUTION_FAILURE.value: 10,
        FailureCode.ASSERTION_FAILED.value: 8,
        FailureCode.EVALUATOR_ERROR.value: -15,
    }
    score = 10 + sum(weights.get(code, 0) for code in set(codes))
    if record.get("expect_success", True) and not record.get("passed"):
        score += 10
    failed_count = sum(1 for item in record.get("asserts", []) if not item.get("pass"))
    score += min(failed_count * 3, 12)
    if trace_available:
        score += 5
    return max(0, min(score, 100))


def _severity(score: int) -> str:
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _fingerprint(record: dict[str, Any], codes: list[str]) -> str:
    failed_types = sorted(
        str(item.get("type", "unknown"))
        for item in record.get("asserts", [])
        if not item.get("pass")
    )
    signature = {
        "case_id": record.get("case_id"),
        "failure_codes": sorted(codes),
        "failed_assertion_types": failed_types,
    }
    raw = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _representative(
    record: dict[str, Any], trace: dict[str, Any] | None, codes: list[str], score: int
) -> dict[str, Any]:
    return {
        "source_file": record.get("_source_file"),
        "source_tag": record.get("_source_tag"),
        "source_line": record.get("_source_line"),
        "run": record.get("run"),
        "passed": bool(record.get("passed")),
        "claims_success": bool(record.get("claims_success")),
        "fake_complete": bool(record.get("fake_complete")),
        "false_failure": FailureCode.FALSE_FAILURE.value in codes,
        "correct_failure": bool(record.get("correct_failure")),
        "steps": record.get("steps", 0),
        "duration_s": record.get("duration_s"),
        "error": record.get("error", ""),
        "asserts": record.get("asserts", []),
        "report_tail": record.get("report_tail", ""),
        "trace_id": record.get("trace_id"),
        "trace_path": record.get("trace_path"),
        "trace_available": trace is not None,
        "trajectory": _compact_trajectory(trace),
        "priority_score": score,
    }


def build_failure_pool(
    records: list[dict[str, Any]],
    case_catalog: dict[str, dict[str, Any]],
    *,
    root: Path,
    include_handled: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    for record in records:
        codes = derive_failure_codes(record)
        if not _is_candidate(record, codes, include_handled):
            continue
        fingerprint = _fingerprint(record, codes)
        trace = _load_trace(record, root)
        score = _priority(record, codes, trace is not None)
        failed_types = sorted(
            {
                str(item.get("type", "unknown"))
                for item in record.get("asserts", [])
                if not item.get("pass")
            }
        )
        occurrence = {
            "source_file": record.get("_source_file"),
            "source_tag": record.get("_source_tag"),
            "source_line": record.get("_source_line"),
            "run": record.get("run"),
            "trace_id": record.get("trace_id"),
            "trace_path": record.get("trace_path"),
        }

        if fingerprint not in groups:
            case = case_catalog.get(str(record.get("case_id")), {})
            groups[fingerprint] = {
                "schema_version": "1.0",
                "sample_id": f"failure-{fingerprint[:12]}",
                "fingerprint": fingerprint,
                "source_case_id": record.get("case_id"),
                "category": record.get("category", case.get("category", "unknown")),
                "task": case.get("task", record.get("task", "")),
                "scenario": case.get("scenario"),
                "expect_success": record.get("expect_success", case.get("expect_success", True)),
                "expected_assertions": case.get("assert", []),
                "failure_codes": codes,
                "failed_assertion_types": failed_types,
                "occurrences": [occurrence],
                "representative": _representative(record, trace, codes, score),
            }
        else:
            group = groups[fingerprint]
            group["occurrences"].append(occurrence)
            if score > group["representative"]["priority_score"]:
                group["representative"] = _representative(record, trace, codes, score)

    pool = []
    for group in groups.values():
        occurrence_count = len(group["occurrences"])
        frequency_bonus = min(max(occurrence_count - 1, 0) * 4, 16)
        priority_score = min(
            int(group["representative"]["priority_score"]) + frequency_bonus,
            100,
        )
        group["occurrence_count"] = occurrence_count
        group["priority_score"] = priority_score
        group["severity"] = _severity(priority_score)
        pool.append(group)

    pool.sort(
        key=lambda item: (-item["priority_score"], -item["occurrence_count"], item["sample_id"])
    )

    replay_cases = []
    for sample in pool:
        if not sample["task"] or not sample["expected_assertions"]:
            continue
        replay_case = {
            "id": f"REPLAY-{sample['source_case_id']}-{sample['fingerprint'][:8]}",
            "category": sample["category"],
            "expect_success": sample["expect_success"],
            "task": sample["task"],
            "assert": sample["expected_assertions"],
            "flywheel": {
                "sample_id": sample["sample_id"],
                "source_case_id": sample["source_case_id"],
                "target_failure_codes": sample["failure_codes"],
                "priority_score": sample["priority_score"],
                "occurrence_count": sample["occurrence_count"],
            },
        }
        if sample.get("scenario"):
            replay_case["scenario"] = sample["scenario"]
        replay_cases.append(replay_case)

    code_counts = Counter(code for sample in pool for code in sample["failure_codes"])
    category_counts = Counter(str(sample["category"]) for sample in pool)
    severity_counts = Counter(str(sample["severity"]) for sample in pool)
    source_record_counts = Counter(str(record.get("_source_file", "unknown")) for record in records)
    trace_backed_samples = sum(bool(sample["representative"]["trace_available"]) for sample in pool)
    failure_occurrences = sum(item["occurrence_count"] for item in pool)
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "input_record_count": len(records),
        "source_record_counts": dict(sorted(source_record_counts.items())),
        "failure_occurrence_count": failure_occurrences,
        "deduplicated_sample_count": len(pool),
        "deduplication_rate": (
            round(1 - len(pool) / failure_occurrences, 4) if failure_occurrences else 0.0
        ),
        "trace_backed_sample_count": trace_backed_samples,
        "replay_case_count": len(replay_cases),
        "include_handled": include_handled,
        "failure_code_counts": dict(sorted(code_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
    }
    return pool, replay_cases, manifest


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_failure_artifacts(
    output_dir: Path,
    pool: list[dict[str, Any]],
    replay_cases: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "failure_pool": output_dir / "failure_pool.jsonl",
        "replay_cases": output_dir / "replay_cases.json",
        "manifest": output_dir / "manifest.json",
    }
    pool_text = "".join(
        json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n" for sample in pool
    )
    _atomic_write(paths["failure_pool"], pool_text)
    _atomic_write(
        paths["replay_cases"],
        json.dumps(replay_cases, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(paths["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return paths
