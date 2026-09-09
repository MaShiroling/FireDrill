"""Safe Planner integration boundary for long-term memory."""

from __future__ import annotations

import importlib
from dataclasses import replace

from app.agent.aiops.memory_context import (
    PlannerMemoryContext,
    format_planner_memory_context,
    load_planner_memory_context,
)
from app.memory import (
    MemoryNamespace,
    MemoryRecord,
    MemorySearchHit,
    MemorySearchResult,
    MemorySearchSource,
    MemoryStatus,
    MemoryType,
)


def _hit(memory_id: str, lesson: str) -> MemorySearchHit:
    record = MemoryRecord.create(
        namespace=MemoryNamespace(
            tenant_id="local",
            device_type="firewall",
            memory_type=MemoryType.FAILURE_LESSON,
        ),
        scenario="Commit 返回超时",
        lesson=lesson,
        recommended_actions=("get_firewall_overview", "get_config_diff"),
        evidence_run_ids=("run-001",),
        status=MemoryStatus.APPROVED,
        confidence=0.9,
    )
    record = replace(record, memory_id=memory_id)
    return MemorySearchHit(record=record, score=0.91, source=MemorySearchSource.VECTOR)


def test_formatter_marks_memory_as_reference_and_respects_budget() -> None:
    first = _hit("mem-first", "先核对 Running Revision，再决定是否重试")
    second = _hit("mem-second", "完成提交后必须执行流量验证" * 30)
    result = MemorySearchResult(hits=(first, second), mode="vector")

    context = format_planner_memory_context(result, max_chars=500)

    assert len(context.text) <= 500
    assert "不是当前设备状态" in context.text
    assert "禁止仅凭历史记忆宣布任务完成" in context.text
    assert context.recalled_ids == ("mem-first", "mem-second")
    assert context.injected_ids == ("mem-first",)
    assert context.truncated


async def test_disabled_memory_does_not_initialize_retrieval_service() -> None:
    def forbidden_factory():
        raise AssertionError("disabled memory must not initialize dependencies")

    context = await load_planner_memory_context(
        "提交超时",
        enabled=False,
        tenant_id="local",
        device_type="firewall",
        top_k=3,
        max_chars=1000,
        service_factory=forbidden_factory,
    )

    assert context == PlannerMemoryContext()


async def test_retrieval_failure_degrades_to_empty_context() -> None:
    def broken_factory():
        raise RuntimeError("memory backend unavailable")

    context = await load_planner_memory_context(
        "提交超时",
        enabled=True,
        tenant_id="local",
        device_type="firewall",
        top_k=3,
        max_chars=1000,
        service_factory=broken_factory,
    )

    assert context.text == ""
    assert context.mode == "unavailable"
    assert context.degraded_reason == "memory backend unavailable"


async def test_loader_passes_scope_and_limits_to_retrieval() -> None:
    captured: dict[str, object] = {}
    hit = _hit("mem-approved", "先验证再重试")

    class FakeService:
        def search(self, query: str, **kwargs) -> MemorySearchResult:
            captured.update({"query": query, **kwargs})
            return MemorySearchResult(hits=(hit,), mode="vector")

    context = await load_planner_memory_context(
        "提交超时",
        enabled=True,
        tenant_id="tenant-a",
        device_type="firewall",
        top_k=2,
        max_chars=1000,
        service_factory=FakeService,
    )

    assert captured == {
        "query": "提交超时",
        "tenant_id": "tenant-a",
        "device_type": "firewall",
        "top_k": 2,
    }
    assert context.injected_ids == ("mem-approved",)


async def test_planner_injects_memory_context_and_records_ids(monkeypatch) -> None:
    planner_module = importlib.import_module("app.agent.aiops.planner")
    model_payload: dict[str, object] = {}
    memory_call: dict[str, object] = {}
    trace_events: list[tuple[str, dict[str, object]]] = []

    class FakeKnowledgeTool:
        async def ainvoke(self, _payload):
            return ""

    class FakeMcpClient:
        async def get_tools(self):
            return []

    class FakeChain:
        async def ainvoke(self, payload):
            model_payload.update(payload)
            return {
                "steps": ["读取当前配置", "完成变更后验证"],
                "memory_ids_used": ["mem-approved", "mem-not-injected"],
            }

    class FakePrompt:
        def __or__(self, _other):
            return FakeChain()

    class FakeLlm:
        def with_structured_output(self, _schema):
            return object()

    async def fake_mcp_client():
        return FakeMcpClient()

    async def fake_memory_context(query: str, **kwargs):
        memory_call.update({"query": query, **kwargs})
        return PlannerMemoryContext(
            text="## 已审核的历史长期记忆\n经验：提交后验证",
            recalled_ids=("mem-approved",),
            injected_ids=("mem-approved",),
            mode="vector",
        )

    monkeypatch.setattr(planner_module, "retrieve_knowledge", FakeKnowledgeTool())
    monkeypatch.setattr(planner_module, "get_mcp_client_with_retry", fake_mcp_client)
    monkeypatch.setattr(planner_module, "DEFAULT_LOCAL_AGENT_TOOLS", [])
    monkeypatch.setattr(planner_module, "format_tools_description", lambda _tools: "无")
    monkeypatch.setattr(planner_module, "planner_prompt", FakePrompt())
    monkeypatch.setattr(planner_module, "ChatQwen", lambda **_kwargs: FakeLlm())
    monkeypatch.setattr(planner_module, "load_planner_memory_context", fake_memory_context)
    monkeypatch.setattr(planner_module.config, "agent_memory_enabled", True)
    monkeypatch.setattr(
        planner_module,
        "trace_event",
        lambda event_type, **kwargs: trace_events.append((event_type, kwargs)),
    )

    result = await planner_module.planner(
        {
            "input": "提交 rule-003 并验证",
            "plan": [],
            "past_steps": [],
            "response": "",
            "memory_tenant_id": "tenant-a",
            "memory_device_type": "firewall",
        }
    )

    assert result == {"plan": ["读取当前配置", "完成变更后验证"]}
    assert model_payload["memory_context"] == "## 已审核的历史长期记忆\n经验：提交后验证"
    assert memory_call["tenant_id"] == "tenant-a"
    assert memory_call["device_type"] == "firewall"
    completed = [data for event, data in trace_events if event == "memory_retrieval_completed"]
    assert completed[0]["data"]["recalled_ids"] == ["mem-approved"]
    assert completed[0]["data"]["injected_ids"] == ["mem-approved"]
    usage = [data for event, data in trace_events if event == "memory_usage_reported"]
    assert usage[0]["data"]["used_ids"] == ["mem-approved"]
    assert usage[0]["data"]["invalid_ids"] == ["mem-not-injected"]
