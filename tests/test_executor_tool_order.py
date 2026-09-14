from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.aiops.executor import (
    blocked_write_tools,
    execute_selected_tools,
    requires_sequential_tool_execution,
)


def _call(name: str, call_id: str) -> dict[str, Any]:
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


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

    messages, mode = await execute_selected_tools(node, [], response)  # type: ignore[arg-type]

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
