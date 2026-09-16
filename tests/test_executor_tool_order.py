from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.aiops.executor import (
    CandidateRevisionConflictError,
    RevisionUnavailableError,
    apply_commit_revision_guard,
    blocked_write_tools,
    execute_selected_tools,
    requires_sequential_tool_execution,
    update_revision_context,
)


def _call(name: str, call_id: str) -> dict[str, Any]:
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


def test_revision_guard_overrides_hallucinated_revision() -> None:
    call = {
        "name": "commit_config",
        "args": {"expected_revision": 123456},
        "id": "call-1",
        "type": "tool_call",
    }

    apply_commit_revision_guard(
        call,
        {"latest_running_revision": 1, "candidate_base_revision": 1},
    )

    assert call["args"]["expected_revision"] == 1


def test_revision_guard_injects_missing_revision() -> None:
    call = _call("commit_config", "call-1")

    apply_commit_revision_guard(
        call,
        {"latest_running_revision": 7, "candidate_base_revision": 7},
    )

    assert call["args"]["expected_revision"] == 7


def test_revision_guard_blocks_commit_without_observed_revision() -> None:
    with pytest.raises(RevisionUnavailableError):
        apply_commit_revision_guard(_call("commit_config", "call-1"), {})


def test_revision_guard_blocks_a_genuinely_stale_candidate() -> None:
    with pytest.raises(CandidateRevisionConflictError):
        apply_commit_revision_guard(
            _call("commit_config", "call-1"),
            {"latest_running_revision": 2, "candidate_base_revision": 1},
        )


def test_revision_context_is_updated_from_diff_result() -> None:
    context: dict[str, Any] = {}
    messages = [
        ToolMessage(
            content=(
                '{"success":true,"running_revision":3,'
                '"candidate_base_revision":3,"change_set_id":"cs-1"}'
            ),
            tool_call_id="call-1",
            name="get_config_diff",
        )
    ]

    update_revision_context(messages, context)

    assert context == {
        "latest_running_revision": 3,
        "candidate_base_revision": 3,
        "change_set_id": "cs-1",
    }


class RecordingToolNode:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def ainvoke(self, payload: dict[str, list[Any]]) -> dict[str, list[ToolMessage]]:
        calls = payload["messages"][-1].tool_calls
        self.batches.append([call["name"] for call in calls])
        return {
            "messages": [
                ToolMessage(
                    content=f"{call['name']}:ok",
                    tool_call_id=call["id"],
                    name=call["name"],
                )
                for call in calls
            ]
        }


def test_stateful_batch_requires_sequential_execution() -> None:
    calls = [
        _call("commit_config", "call-1"),
        _call("get_firewall_overview", "call-2"),
        _call("test_traffic", "call-3"),
    ]

    assert requires_sequential_tool_execution(calls) is True
    assert requires_sequential_tool_execution(calls[1:]) is False
    assert requires_sequential_tool_execution(calls[:1]) is False


def test_read_only_policy_blocks_stateful_tools() -> None:
    calls = [
        _call("list_firewall_rules", "call-1"),
        _call("add_firewall_rule", "call-2"),
        _call("commit_config", "call-3"),
    ]

    assert blocked_write_tools(calls, allow_write=False) == [
        "add_firewall_rule",
        "commit_config",
    ]
    assert blocked_write_tools(calls, allow_write=True) == []


@pytest.mark.asyncio
async def test_stateful_batch_is_executed_one_call_at_a_time_in_model_order() -> None:
    calls = [
        _call("commit_config", "call-1"),
        _call("get_firewall_overview", "call-2"),
        _call("test_traffic", "call-3"),
    ]
    response = AIMessage(content="", tool_calls=calls)
    node = RecordingToolNode()

    messages, mode = await execute_selected_tools(
        node,  # type: ignore[arg-type]
        [],
        response,
        revision_context={
            "latest_running_revision": 1,
            "candidate_base_revision": 1,
        },
    )

    assert mode == "sequential"
    assert node.batches == [
        ["commit_config"],
        ["get_firewall_overview"],
        ["test_traffic"],
    ]
    assert [message.tool_call_id for message in messages] == ["call-1", "call-2", "call-3"]


@pytest.mark.asyncio
async def test_read_only_batch_keeps_toolnode_parallel_path() -> None:
    calls = [
        _call("get_firewall_overview", "call-1"),
        _call("list_firewall_rules", "call-2"),
    ]
    response = AIMessage(content="", tool_calls=calls)
    node = RecordingToolNode()

    messages, mode = await execute_selected_tools(node, [], response)  # type: ignore[arg-type]

    assert mode == "parallel"
    assert node.batches == [["get_firewall_overview", "list_firewall_rules"]]
    assert [message.tool_call_id for message in messages] == ["call-1", "call-2"]


class RevisionAwareToolNode:
    def __init__(self) -> None:
        self.commit_args: dict[str, Any] | None = None

    async def ainvoke(self, payload: dict[str, list[Any]]) -> dict[str, list[ToolMessage]]:
        call = payload["messages"][-1].tool_calls[0]
        if call["name"] == "get_config_diff":
            content = (
                '{"success":true,"running_revision":1,'
                '"candidate_base_revision":1,"change_set_id":"cs-1"}'
            )
        else:
            self.commit_args = dict(call["args"])
            content = '{"success":true,"running_revision":2}'
        return {
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            ]
        }


@pytest.mark.asyncio
async def test_diff_then_commit_batch_uses_freshly_observed_revision() -> None:
    calls = [
        _call("get_config_diff", "call-1"),
        {
            "name": "commit_config",
            "args": {"expected_revision": 123456},
            "id": "call-2",
            "type": "tool_call",
        },
    ]
    response = AIMessage(content="", tool_calls=calls)
    node = RevisionAwareToolNode()
    context: dict[str, Any] = {}

    _, mode = await execute_selected_tools(
        node,  # type: ignore[arg-type]
        [],
        response,
        revision_context=context,
    )

    assert mode == "sequential"
    assert node.commit_args == {"expected_revision": 1}
    assert context["latest_running_revision"] == 2
