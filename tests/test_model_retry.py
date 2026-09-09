import json
from pathlib import Path

import pytest

from app.agent.aiops.model_retry import invoke_model_with_retry, is_transient_model_error
from app.agent.aiops.utils import format_execution_context
from app.observability import begin_run, finish_run


async def test_transient_exception_group_is_retried_and_traced(tmp_path: Path) -> None:
    recorder, token = begin_run(
        enabled=True,
        root=tmp_path,
        trace_dir=tmp_path / "runs",
        session_id="model-retry",
        task="select a tool",
        runtime={},
    )
    attempts = 0

    async def call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ExceptionGroup("TaskGroup", [TimeoutError("read timed out")])
        return "ok"

    result = await invoke_model_with_retry(
        call,
        node="executor",
        purpose="select_tools",
        max_attempts=2,
        delay_s=0,
    )
    finish_run(recorder, token, status="completed")

    assert result == "ok"
    assert attempts == 2
    assert recorder is not None
    payload = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert payload["metrics"]["model_retries_scheduled"] == 1
    assert payload["metrics"]["model_retry_exhausted"] == 0
    failures = [event for event in payload["events"] if event["event_type"] == "model_call_failed"]
    assert failures[0]["data"]["retryable"] is True
    assert failures[0]["data"]["will_retry"] is True


async def test_permanent_model_error_is_not_retried() -> None:
    attempts = 0

    async def call() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid structured output schema")

    with pytest.raises(ValueError, match="invalid structured output"):
        await invoke_model_with_retry(call, node="planner", purpose="create_plan", delay_s=0)

    assert attempts == 1


def test_transient_detection_inspects_exception_group_leaves() -> None:
    assert is_transient_model_error(
        ExceptionGroup("TaskGroup", [TimeoutError("timeout"), ConnectionError("reset")])
    )
    assert not is_transient_model_error(
        ExceptionGroup("TaskGroup", [TimeoutError("timeout"), ValueError("bad schema")])
    )


def test_execution_context_preserves_real_rule_ids() -> None:
    context = format_execution_context(
        "请删除 allow-dns（rule-003）",
        [("新增规则", "成功，rule_id=rule-007")],
    )

    assert "rule-003" in context
    assert "rule-007" in context
    assert "实际结果" in context
