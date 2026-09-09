import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp_servers"))

from firewall_server import FirewallState
from mcp.types import CallToolResult, TextContent

from app.agent.mcp_client import retry_interceptor


def _request(name: str):
    return SimpleNamespace(name=name, server_name="firewall", args={})


def _result(payload: dict, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        isError=is_error,
    )


def _payload(result: CallToolResult) -> dict:
    return json.loads(result.content[0].text)


def _prepare_change(firewall: FirewallState) -> None:
    result = firewall.add_rule(
        "allow-db",
        "trust",
        "dmz",
        "10.1.7.0/24",
        "172.16.1.40/32",
        "tcp",
        "5432",
        "allow",
        "test",
    )
    assert result["success"] is True


async def test_application_success_false_flaky_commit_retries_until_success() -> None:
    firewall = FirewallState()
    _prepare_change(firewall)
    firewall.fault = {"mode": "commit_flaky", "fail_times": 2}
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return _result(firewall.commit())

    result = await retry_interceptor(_request("commit_config"), handler, delay=0)

    assert _payload(result)["success"] is True
    assert attempts == 3
    assert firewall.running_revision == 2
    commits = [event for event in firewall.audit_log if event["operation"] == "commit"]
    assert [event["result"] for event in commits] == ["error", "error", "success"]


async def test_commit_state_unknown_is_not_retried() -> None:
    firewall = FirewallState()
    _prepare_change(firewall)
    firewall.fault = {"mode": "commit_lose", "fail_times": 0}
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return _result(firewall.commit())

    result = await retry_interceptor(_request("commit_config"), handler, delay=0)

    assert _payload(result)["success"] is False
    assert attempts == 1
    assert firewall.running_revision == 2


async def test_permanent_commit_rejection_is_not_retried() -> None:
    firewall = FirewallState()
    _prepare_change(firewall)
    firewall.fault = {"mode": "commit_reject", "fail_times": 0}
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return _result(firewall.commit())

    result = await retry_interceptor(_request("commit_config"), handler, delay=0)

    assert "拒绝" in _payload(result)["error"]
    assert attempts == 1
    assert firewall.running_revision == 1


async def test_mutating_tool_transport_timeout_is_not_blindly_retried() -> None:
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("network timeout")

    result = await retry_interceptor(_request("commit_config"), handler, delay=0)

    assert result.isError is True
    assert "结果不确定" in result.content[0].text
    assert attempts == 1


async def test_read_tool_transport_exception_is_retried() -> None:
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("connection reset")
        return _result({"success": True, "rules": []})

    result = await retry_interceptor(_request("list_firewall_rules"), handler, delay=0)

    assert _payload(result)["success"] is True
    assert attempts == 2


async def test_protocol_error_with_retry_hint_is_retried() -> None:
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return CallToolResult(
                content=[TextContent(type="text", text="服务繁忙，请稍后重试")],
                isError=True,
            )
        return _result({"success": True})

    result = await retry_interceptor(_request("get_firewall_overview"), handler, delay=0)

    assert _payload(result)["success"] is True
    assert attempts == 2


async def test_retry_exhaustion_preserves_last_application_error() -> None:
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return _result({"success": False, "error": "设备繁忙，请稍后重试"})

    result = await retry_interceptor(_request("commit_config"), handler, max_retries=2, delay=0)

    assert _payload(result) == {"success": False, "error": "设备繁忙，请稍后重试"}
    assert attempts == 2


async def test_structured_content_error_is_detected() -> None:
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return CallToolResult(
                content=[TextContent(type="text", text="busy")],
                structuredContent={"success": False, "error": "设备繁忙，请稍后重试"},
                isError=False,
            )
        return _result({"success": True})

    result = await retry_interceptor(_request("commit_config"), handler, delay=0)

    assert _payload(result)["success"] is True
    assert attempts == 2
