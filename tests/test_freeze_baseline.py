import json
from pathlib import Path

import pytest

from evals.freeze_baseline import (
    build_comparison,
    load_jsonl,
    select_cases,
    summarize,
    validate_primary_coverage,
)


def _record(case_id: str, run: int, *, passed: bool, fake: bool = False) -> dict:
    return {
        "case_id": case_id,
        "category": "change",
        "run": run,
        "passed": passed,
        "fake_complete": fake,
        "steps": 2,
        "duration_s": 4.0,
        "error": "",
        "asserts": [{"type": "revision", "pass": passed, "detail": "revision=2"}],
    }


def test_summarize_uses_run_count_as_fake_completion_denominator() -> None:
    metrics = summarize(
        [
            _record("A", 1, passed=True),
            _record("A", 2, passed=False, fake=True),
        ]
    )

    assert metrics["task_success_rate"] == 0.5
    assert metrics["fake_completion_rate"] == 0.5
    assert metrics["assertion_pass_rate"] == 0.5


def test_comparison_only_uses_shared_case_runs() -> None:
    primary = [
        _record("A", 1, passed=True),
        _record("B", 1, passed=True),
    ]
    comparison = [
        _record("A", 1, passed=False, fake=True),
    ]

    result = build_comparison(primary, comparison)

    assert result["runs"] == 1
    assert result["case_count"] == 1
    assert result["delta"]["task_success_rate_pp"] == 100.0
    assert result["delta"]["fake_completion_rate_pp"] == -100.0


def test_primary_coverage_rejects_missing_run() -> None:
    cases = [{"id": "A"}]
    with pytest.raises(ValueError, match="覆盖不完整"):
        validate_primary_coverage(cases, [_record("A", 1, passed=True)], expected_runs=2)


def test_select_cases_preserves_requested_order() -> None:
    cases = [{"id": "A"}, {"id": "B"}, {"id": "C"}]

    selected = select_cases(cases, "C,A")

    assert [case["id"] for case in selected] == ["C", "A"]


def test_select_cases_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="未知用例"):
        select_cases([{"id": "A"}], "A,missing")


def test_load_jsonl_rejects_duplicate_case_run(tmp_path: Path) -> None:
    result_file = tmp_path / "duplicate.jsonl"
    record = _record("A", 1, passed=True)
    result_file.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重复运行"):
        load_jsonl(result_file)
