"""Evaluate a replay run and return a CI-friendly regression gate status."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.flywheel import (  # noqa: E402
    GateThresholds,
    analyze_replay_regression,
    load_case_catalog,
    load_result_records,
    write_regression_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查回放结果是否通过回归门禁")
    parser.add_argument("--results", action="append", required=True, help="回放结果 JSONL，可重复")
    parser.add_argument(
        "--replay-cases",
        default=str(ROOT / "evals" / "artifacts" / "flywheel" / "replay_cases.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "evals" / "artifacts" / "flywheel" / "regression"),
    )
    parser.add_argument("--name", default="replay-regression")
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--min-healthy-rate", type=float, default=0.66)
    parser.add_argument("--max-target-recurrence-rate", type=float, default=0.34)
    parser.add_argument("--max-fake-completion-rate", type=float, default=0.10)
    parser.add_argument("--max-false-failure-rate", type=float, default=0.10)
    parser.add_argument("--max-run-error-rate", type=float, default=0.05)
    parser.add_argument("--require-all-cases", action="store_true")
    parser.add_argument("--fail-on-new-codes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds = GateThresholds(
        min_runs_per_case=args.min_runs,
        min_healthy_rate=args.min_healthy_rate,
        max_target_recurrence_rate=args.max_target_recurrence_rate,
        max_fake_completion_rate=args.max_fake_completion_rate,
        max_false_failure_rate=args.max_false_failure_rate,
        max_run_error_rate=args.max_run_error_rate,
        require_all_cases=args.require_all_cases,
        fail_on_new_codes=args.fail_on_new_codes,
    )
    records = load_result_records([Path(path) for path in args.results])
    replay_catalog = load_case_catalog(Path(args.replay_cases))
    report = analyze_replay_regression(records, replay_catalog, thresholds, name=args.name)
    paths = write_regression_report(Path(args.output_dir), report)
    summary = report["summary"]
    print(
        f"gate={report['status'].upper()} runs={summary['total_runs']} "
        f"healthy={summary['healthy_rate']:.1%} "
        f"target_recurrence={summary['target_recurrence_rate']:.1%} "
        f"fake={summary['fake_completion_rate']:.1%}"
    )
    for name, path in paths.items():
        print(f"report_{name}: {path}")
    if report["violations"]:
        for violation in report["violations"]:
            print(f"FAIL {violation['metric']}: {violation['message']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
