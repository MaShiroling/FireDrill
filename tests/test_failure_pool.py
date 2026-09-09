import json
from pathlib import Path

from app.evaluation import FailureCode
from app.flywheel import (
    build_failure_pool,
    derive_failure_codes,
    load_result_records,
    write_failure_artifacts,
)


def _record(run: int, *, passed: bool, fake: bool = False, report: str = "") -> dict:
    return {
        "case_id": "FW-C01",
        "category": "change",
        "run": run,
        "expect_success": True,
        "passed": passed,
        "claims_success": fake,
        "fake_complete": fake,
        "correct_failure": False,
        "steps": 4,
        "duration_s": 10.0,
        "error": "",
        "asserts": [
            {"type": "revision", "pass": passed, "detail": "revision=1"},
            {"type": "traffic", "pass": passed, "detail": "action=deny（期望 allow）"},
        ],
        "report_tail": report,
        "_source_file": "evals/results/current.jsonl",
        "_source_tag": "current",
        "_source_line": run,
    }


def _catalog() -> dict:
    return {
        "FW-C01": {
            "id": "FW-C01",
            "category": "change",
            "expect_success": True,
            "task": "放通 TCP 443 并验证",
            "assert": [{"type": "revision", "op": ">", "value": 1}],
        }
    }


def test_derive_failure_codes_backfills_legacy_result() -> None:
    codes = derive_failure_codes(_record(1, passed=False, fake=True))

    assert FailureCode.ASSERTION_FAILED.value in codes
    assert FailureCode.VERIFICATION_MISSING.value in codes
    assert FailureCode.FALSE_COMPLETION.value in codes
    assert FailureCode.REPORT_INCONSISTENT.value in codes


def test_pool_deduplicates_repeated_failure_and_builds_replay_case(tmp_path: Path) -> None:
    records = [_record(1, passed=False), _record(2, passed=False)]

    pool, replay, manifest = build_failure_pool(records, _catalog(), root=tmp_path)

    assert len(pool) == 1
    assert pool[0]["occurrence_count"] == 2
    assert len(pool[0]["occurrences"]) == 2
    assert replay[0]["task"] == "放通 TCP 443 并验证"
    assert replay[0]["flywheel"]["occurrence_count"] == 2
    assert manifest["failure_occurrence_count"] == 2
    assert manifest["deduplicated_sample_count"] == 1
    assert manifest["deduplication_rate"] == 0.5


def test_false_failure_is_mined_even_when_terminal_assertions_pass(tmp_path: Path) -> None:
    record = _record(1, passed=True, report="规则未能生效")

    pool, _replay, _manifest = build_failure_pool([record], _catalog(), root=tmp_path)

    assert len(pool) == 1
    assert FailureCode.FALSE_FAILURE.value in pool[0]["failure_codes"]


def test_trace_is_compacted_into_representative_trajectory(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "sequence": 1,
                        "event_type": "tool_inventory_loaded",
                        "node": "planner",
                        "data": {},
                    },
                    {
                        "sequence": 2,
                        "event_type": "no_tool_selected",
                        "node": "executor",
                        "data": {"step": "提交"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    record = _record(1, passed=False)
    record["trace_path"] = "trace.json"

    pool, _replay, _manifest = build_failure_pool([record], _catalog(), root=tmp_path)

    representative = pool[0]["representative"]
    assert representative["trace_available"] is True
    assert [event["event_type"] for event in representative["trajectory"]] == ["no_tool_selected"]


def test_handled_negative_is_optional(tmp_path: Path) -> None:
    record = {
        **_record(1, passed=True),
        "case_id": "FW-E01",
        "category": "error",
        "expect_success": False,
        "correct_failure": True,
        "failure_codes": [FailureCode.INVALID_ARGUMENT.value],
        "asserts": [{"type": "revision", "pass": True, "detail": "revision=1"}],
    }

    default_pool, _, _ = build_failure_pool([record], {}, root=tmp_path)
    handled_pool, _, _ = build_failure_pool([record], {}, root=tmp_path, include_handled=True)

    assert default_pool == []
    assert len(handled_pool) == 1


def test_legacy_report_heuristic_alone_does_not_create_failure_sample(tmp_path: Path) -> None:
    record = {
        **_record(1, passed=True),
        "case_id": "FW-E05",
        "category": "error",
        "expect_success": False,
        "claims_success": True,
        "correct_failure": False,
        "report_tail": "当前没有改动，无需提交；以后有改动再提交生效。",
    }

    pool, _, _ = build_failure_pool([record], {}, root=tmp_path)

    assert pool == []


def test_result_loading_and_atomic_artifact_output(tmp_path: Path) -> None:
    result_path = tmp_path / "current.jsonl"
    raw_record = _record(1, passed=False)
    raw_record = {key: value for key, value in raw_record.items() if not key.startswith("_")}
    result_path.write_text(json.dumps(raw_record) + "\n", encoding="utf-8")

    records = load_result_records([result_path])
    pool, replay, manifest = build_failure_pool(records, _catalog(), root=tmp_path)
    paths = write_failure_artifacts(tmp_path / "flywheel", pool, replay, manifest)

    assert records[0]["_source_tag"] == "current"
    assert len(paths["failure_pool"].read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads(paths["replay_cases"].read_text(encoding="utf-8"))[0]["id"].startswith(
        "REPLAY-FW-C01-"
    )
    assert json.loads(paths["manifest"].read_text(encoding="utf-8"))["replay_case_count"] == 1
