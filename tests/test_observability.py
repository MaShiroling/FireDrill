import json
from pathlib import Path
from types import SimpleNamespace

from app.agent.mcp_client import retry_interceptor
from app.observability import begin_run, finish_run, get_current_recorder, trace_event
from app.observability.redaction import REDACTED, sanitize


def test_sanitize_redacts_nested_secrets_and_truncates_strings() -> None:
    value = {
        "api_key": "secret-key",
        "nested": {"Authorization": "Bearer secret", "input_tokens": 42},
        "content": "abcdefgh",
    }

    cleaned = sanitize(value, max_chars=4)

    assert cleaned["api_key"] == REDACTED
    assert cleaned["nested"]["Authorization"] == REDACTED
    assert cleaned["nested"]["input_tokens"] == 42
    assert cleaned["content"].startswith("abcd...[TRUNCATED")


def test_run_recorder_persists_ordered_trace(tmp_path: Path) -> None:
    recorder, token = begin_run(
        enabled=True,
        root=tmp_path,
        trace_dir=tmp_path / "runs",
        session_id="session/unsafe",
        task="change firewall",
        runtime={"model": "test-model", "api_key": "must-not-leak"},
        metadata={"case_id": "FW-C01"},
        max_value_chars=100,
    )
    assert recorder is not None
    assert get_current_recorder() is recorder

    trace_event("node_started", node="planner", data={"attempt": 1})
    trace_event("tool_call_completed", node="executor", data={"result": "ok"})
    finish_run(
        recorder,
        token,
        status="completed",
        final_state={"response": "done"},
    )

    assert get_current_recorder() is None
    payload = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["metadata"]["case_id"] == "FW-C01"
    assert payload["version_manifest"]["runtime"]["api_key"] == REDACTED
    assert [event["sequence"] for event in payload["events"]] == [1, 2, 3, 4]
    assert payload["events"][-1]["event_type"] == "run_finished"
    assert payload["metrics"]["tool_calls"] == 1
    assert payload["metrics"]["model_calls"] == 0


async def test_mcp_retry_attempts_are_recorded(tmp_path: Path) -> None:
    recorder, token = begin_run(
        enabled=True,
        root=tmp_path,
        trace_dir=tmp_path / "runs",
        session_id="retry-test",
        task="call a tool",
        runtime={},
    )
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        return {"ok": True}

    request = SimpleNamespace(
        name="list_firewall_rules", server_name="firewall", args={"password": "hidden"}
    )
    result = await retry_interceptor(request, handler, max_retries=2, delay=0)
    finish_run(recorder, token, status="completed")

    assert result == {"ok": True}
    assert recorder is not None
    payload = json.loads(recorder.path.read_text(encoding="utf-8"))
    completed = [
        event for event in payload["events"] if event["event_type"] == "mcp_attempt_completed"
    ]
    assert [event["data"]["success"] for event in completed] == [False, True]
    scheduled = [
        event for event in payload["events"] if event["event_type"] == "mcp_retry_scheduled"
    ]
    assert len(scheduled) == 1
    assert scheduled[0]["data"]["error_kind"] == "transient"
    assert payload["metrics"]["mcp_retries_scheduled"] == 1
    assert payload["metrics"]["mcp_retries_skipped"] == 0
    assert payload["metrics"]["mcp_retry_exhausted"] == 0
    started = [event for event in payload["events"] if event["event_type"] == "mcp_attempt_started"]
    assert started[0]["data"]["args"]["password"] == REDACTED
