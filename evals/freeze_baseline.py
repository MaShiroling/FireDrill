"""Freeze reproducible evaluation baselines from existing result artifacts.

This command does not run an Agent. It validates completed JSONL results, records
their hashes and source revision, and writes a machine-readable baseline manifest.

Example:
    python evals/freeze_baseline.py --tag current --comparison-tag legacy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "evals" / "cases_firewall.json"
DEFAULT_RESULTS_DIR = ROOT / "evals" / "results"
DEFAULT_BASELINES_DIR = ROOT / "evals" / "baselines"

METRIC_DEFINITIONS = {
    "task_success_rate": (
        "passed=true 的运行数 / 运行总数；passed 要求所有终态断言通过且 runner 无错误"
    ),
    "fake_completion_rate": (
        "fake_complete=true 的运行数 / 运行总数；表示报告声称成功但终态断言未通过"
    ),
    "run_error_rate": "runner 或 Agent 产生非空 error 的运行数 / 运行总数",
    "assertion_pass_rate": "通过的确定性终态断言数 / 全部终态断言数",
    "average_steps": "每次运行产生的 step_complete 事件数的算术平均值",
    "average_duration_s": "每次运行墙钟耗时（秒）的算术平均值",
}

REQUIRED_RESULT_FIELDS = {
    "case_id",
    "category",
    "run",
    "passed",
    "fake_complete",
    "steps",
    "duration_s",
    "error",
    "asserts",
}


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = REQUIRED_RESULT_FIELDS - record.keys()
        if missing:
            raise ValueError(f"{path}:{line_no} 缺少字段: {sorted(missing)}")
        key = (str(record["case_id"]), int(record["run"]))
        if key in seen:
            raise ValueError(f"{path}:{line_no} 存在重复运行: {key}")
        seen.add(key)
        records.append(record)
    if not records:
        raise ValueError(f"结果文件为空: {path}")
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(bool(r["passed"]) for r in records)
    fake_complete = sum(bool(r["fake_complete"]) for r in records)
    run_errors = sum(bool(r.get("error")) for r in records)
    assertion_total = sum(len(r.get("asserts", [])) for r in records)
    assertion_passed = sum(
        bool(assertion.get("pass")) for record in records for assertion in record.get("asserts", [])
    )
    return {
        "runs": total,
        "passed": passed,
        "task_success_rate": _rate(passed, total),
        "fake_completions": fake_complete,
        "fake_completion_rate": _rate(fake_complete, total),
        "run_errors": run_errors,
        "run_error_rate": _rate(run_errors, total),
        "assertions": assertion_total,
        "assertions_passed": assertion_passed,
        "assertion_pass_rate": _rate(assertion_passed, assertion_total),
        "average_steps": round(sum(float(r["steps"]) for r in records) / total, 2),
        "average_duration_s": round(sum(float(r["duration_s"]) for r in records) / total, 2),
    }


def summarize_by_category(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    categories = sorted({str(record["category"]) for record in records})
    return {
        category: summarize([r for r in records if r["category"] == category])
        for category in categories
    }


def validate_primary_coverage(
    cases: list[dict[str, Any]], records: list[dict[str, Any]], expected_runs: int
) -> None:
    expected = {
        (str(case["id"]), run_idx) for case in cases for run_idx in range(1, expected_runs + 1)
    }
    actual = {(str(record["case_id"]), int(record["run"])) for record in records}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "主基线覆盖不完整: "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''}, "
            f"extra={extra[:10]}{'...' if len(extra) > 10 else ''}"
        )


def select_cases(cases: list[dict[str, Any]], raw_case_ids: str) -> list[dict[str, Any]]:
    """Select a stable case subset for targeted replay baselines."""
    if not raw_case_ids.strip():
        return cases

    requested = [case_id.strip() for case_id in raw_case_ids.split(",") if case_id.strip()]
    if not requested:
        raise ValueError("--case-ids 未包含有效用例 ID")
    if len(requested) != len(set(requested)):
        raise ValueError("--case-ids 包含重复用例 ID")

    case_by_id = {str(case["id"]): case for case in cases}
    missing = [case_id for case_id in requested if case_id not in case_by_id]
    if missing:
        raise ValueError(f"--case-ids 包含未知用例: {missing}")
    return [case_by_id[case_id] for case_id in requested]


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_metadata() -> dict[str, Any]:
    status = _git_output("status", "--short")
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "short_commit": _git_output("rev-parse", "--short", "HEAD"),
        "subject": _git_output("log", "-1", "--format=%s"),
        "worktree_dirty": bool(status),
    }


def artifact(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = sorted({str(record["case_id"]) for record in records})
    run_indices = sorted({int(record["run"]) for record in records})
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "case_count": len(case_ids),
        "run_indices": run_indices,
        "metrics": summarize(records),
        "by_category": summarize_by_category(records),
    }


def build_comparison(
    primary: list[dict[str, Any]], comparison: list[dict[str, Any]]
) -> dict[str, Any]:
    primary_by_key = {(str(r["case_id"]), int(r["run"])): r for r in primary}
    comparison_by_key = {(str(r["case_id"]), int(r["run"])): r for r in comparison}
    shared_keys = sorted(primary_by_key.keys() & comparison_by_key.keys())
    if not shared_keys:
        raise ValueError("主版本和对照版本没有可比较的 (case_id, run) 记录")

    primary_shared = [primary_by_key[key] for key in shared_keys]
    comparison_shared = [comparison_by_key[key] for key in shared_keys]
    primary_metrics = summarize(primary_shared)
    comparison_metrics = summarize(comparison_shared)
    return {
        "scope": "两个结果文件共有的 (case_id, run)；用于同口径 A/B",
        "case_count": len({key[0] for key in shared_keys}),
        "runs": len(shared_keys),
        "categories": sorted({str(r["category"]) for r in primary_shared}),
        "primary": primary_metrics,
        "comparison": comparison_metrics,
        "delta": {
            "task_success_rate_pp": round(
                (primary_metrics["task_success_rate"] - comparison_metrics["task_success_rate"])
                * 100,
                2,
            ),
            "fake_completion_rate_pp": round(
                (
                    primary_metrics["fake_completion_rate"]
                    - comparison_metrics["fake_completion_rate"]
                )
                * 100,
                2,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 FireDrill 评测基线")
    parser.add_argument("--tag", default="current", help="主结果标签")
    parser.add_argument("--comparison-tag", default="legacy", help="可选对照结果标签")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--case-ids",
        default="",
        help="逗号分隔的用例 ID；用于冻结定向回放子集，默认要求用例文件全量覆盖",
    )
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases_path = args.cases.resolve()
    primary_path = DEFAULT_RESULTS_DIR / f"{args.tag}.jsonl"
    comparison_path = DEFAULT_RESULTS_DIR / f"{args.comparison_tag}.jsonl"
    output_path = args.output or DEFAULT_BASELINES_DIR / f"firewall-{args.tag}.json"

    all_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = select_cases(all_cases, args.case_ids)
    primary = load_jsonl(primary_path)
    validate_primary_coverage(cases, primary, args.expected_runs)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "baseline_id": f"firewall-{args.tag}",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": git_metadata(),
        "configuration": {
            "provenance": (
                "从当前源码默认值补录；历史 JSONL 未逐次记录运行时配置，"
                "后续轨迹版本将改为每次运行直接采集"
            ),
            "primary_tag": args.tag,
            "comparison_tag": args.comparison_tag,
            "expected_runs_per_case": args.expected_runs,
            "selected_case_ids": [str(case["id"]) for case in cases],
            "model": "qwen-max",
            "planner_temperature": 0,
            "executor_temperature": 0,
            "replanner_temperature": 0,
            "max_agent_steps": 8,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "dataset": {
            "path": str(cases_path.relative_to(ROOT)),
            "sha256": sha256_file(cases_path),
            "case_count": len(cases),
            "category_counts": dict(sorted(Counter(c["category"] for c in cases).items())),
        },
        "artifacts": {args.tag: artifact(primary_path, primary)},
    }

    if comparison_path.exists():
        comparison = load_jsonl(comparison_path)
        manifest["artifacts"][args.comparison_tag] = artifact(comparison_path, comparison)
        manifest["ab_comparison"] = build_comparison(primary, comparison)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"基线已冻结: {output_path}")
    print(json.dumps(manifest["artifacts"][args.tag]["metrics"], ensure_ascii=False, indent=2))
    if "ab_comparison" in manifest:
        print("A/B 可比子集:")
        print(json.dumps(manifest["ab_comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
