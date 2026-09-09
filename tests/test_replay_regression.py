import json
from pathlib import Path

import pytest

from app.flywheel import (
    GateThresholds,
    analyze_replay_regression,
    render_regression_markdown,
    write_regression_report,
)


def _catalog() -> dict:
    return {
        "REPLAY-A": {
            "id": "REPLAY-A",
            "flywheel": {
                "sample_id": "failure-a",
                "source_case_id": "FW-C01",
                "target_failure_codes": ["assertion_failed", "false_completion"],
                "priority_score": 90,
            },
        }
    }


def _record(
    run: int,
    *,
    passed: bool = True,
    fake: bool = False,
    error: str = "",
    assertion_type: str = "revision",
) -> dict:
    return {
        "case_id": "REPLAY-A",
        "run": run,
        "expect_success": True,
        "passed": passed,
        "claims_success": fake,
        "fake_complete": fake,
        "false_failure": False,
        "correct_failure": False,
        "steps": 4,
        "error": error,
        "asserts": [
            {
                "type": assertion_type,
                "pass": passed,
                "detail": "revision=2" if passed else "revision=1",
            }
        ],
        "failure_codes": [],
    }


def test_fully_recovered_replay_passes_gate() -> None:
    report = analyze_replay_regression(
        [_record(1), _record(2), _record(3)],
        _catalog(),
        GateThresholds(),
    )

    assert report["status"] == "passed"
    assert report["summary"]["healthy_rate"] == 1.0
    assert report["summary"]["target_recurrence_rate"] == 0.0
    assert report["cases"][0]["status"] == "recovered"


def test_recurrent_target_failure_breaks_gate_and_reports_new_codes() -> None:
    records = [
        _record(1, passed=False, fake=True, assertion_type="traffic"),
        _record(2, passed=False, fake=True, assertion_type="traffic"),
        _record(3),
    ]

    report = analyze_replay_regression(records, _catalog(), GateThresholds())

    assert report["status"] == "failed"
    metrics = {item["metric"] for item in report["violations"]}
    assert {"healthy_rate", "target_recurrence_rate", "fake_completion_rate"} <= metrics
    assert report["new_failure_code_counts"]["verification_missing"] == 2
    assert report["cases"][0]["status"] == "unstable"


def test_result_embedded_flywheel_metadata_takes_precedence() -> None:
    record = _record(1)
    record["flywheel"] = {
        "sample_id": "embedded",
        "source_case_id": "FW-X01",
        "target_failure_codes": ["run_error"],
    }

    report = analyze_replay_regression(
        [record],
        {},
        GateThresholds(min_runs_per_case=1),
    )

    assert report["cases"][0]["sample_id"] == "embedded"
    assert report["cases"][0]["target_failure_codes"] == ["run_error"]


def test_gate_detects_incomplete_unlinked_and_missing_cases() -> None:
    catalog = {
        **_catalog(),
        "REPLAY-B": {
            "id": "REPLAY-B",
            "flywheel": {"target_failure_codes": ["assertion_failed"]},
        },
    }
    unlinked = {**_record(1), "case_id": "UNKNOWN"}

    report = analyze_replay_regression(
        [_record(1), unlinked],
        catalog,
        GateThresholds(require_all_cases=True),
    )

    metrics = {item["metric"] for item in report["violations"]}
    assert "unlinked_record_count" in metrics
    assert "min_runs_per_case" in metrics
    assert "missing_case_count" in metrics
    assert report["missing_case_ids"] == ["REPLAY-B"]


def test_fail_on_new_codes_is_optional() -> None:
    records = [_record(1), _record(2), _record(3)]
    records[0]["passed"] = False
    records[0]["error"] = "timeout"
    records[0]["asserts"][0]["pass"] = False

    permissive = analyze_replay_regression(
        records,
        _catalog(),
        GateThresholds(
            min_healthy_rate=0.5,
            max_target_recurrence_rate=1.0,
            max_run_error_rate=1.0,
        ),
    )
    strict = analyze_replay_regression(
        records,
        _catalog(),
        GateThresholds(
            min_healthy_rate=0.5,
            max_target_recurrence_rate=1.0,
            max_run_error_rate=1.0,
            fail_on_new_codes=True,
        ),
    )

    assert permissive["status"] == "passed"
    assert strict["status"] == "failed"
    assert any(item["metric"] == "new_failure_codes" for item in strict["violations"])


def test_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = analyze_replay_regression(
        [_record(1), _record(2), _record(3)],
        _catalog(),
        GateThresholds(),
        name="test-gate",
    )

    markdown = render_regression_markdown(report)
    paths = write_regression_report(tmp_path, report)

    assert "Status: PASS" in markdown
    assert "REPLAY-A" in markdown
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["status"] == "passed"
    assert paths["markdown"].read_text(encoding="utf-8") == markdown


def test_threshold_validation_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="max_run_error_rate"):
        GateThresholds(max_run_error_rate=1.1).validate()
