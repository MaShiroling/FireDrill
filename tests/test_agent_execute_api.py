"""通用 Agent 执行接口测试。"""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import aiops


class FakeAIOpsService:
    """记录接口透传参数并返回固定事件。"""

    def __init__(self) -> None:
        self.execute_kwargs: dict | None = None

    async def execute(self, **kwargs):
        self.execute_kwargs = kwargs
        yield {"type": "plan", "plan": ["读取配置", "提交并验证"]}
        yield {"type": "complete", "response": "配置完成"}


def _parse_sse_data(response_text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: ")
    ]


def test_execute_agent_streams_graph_events(monkeypatch) -> None:
    fake_service = FakeAIOpsService()
    monkeypatch.setattr(aiops, "aiops_service", fake_service)

    app = FastAPI()
    app.include_router(aiops.router, prefix="/api")

    with TestClient(app) as client:
        response = client.post(
            "/api/agent/execute",
            json={
                "session_id": "demo-001",
                "task": "提交并验证防火墙规则",
                "tenant_id": "tenant-a",
                "device_type": "firewall-v2",
                "allow_write": False,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _parse_sse_data(response.text) == [
        {"type": "plan", "plan": ["读取配置", "提交并验证"]},
        {"type": "complete", "response": "配置完成"},
    ]
    assert fake_service.execute_kwargs == {
        "user_input": "提交并验证防火墙规则",
        "session_id": "demo-001",
        "trace_metadata": {"entrypoint": "api_agent_execute"},
        "memory_tenant_id": "tenant-a",
        "memory_device_type": "firewall-v2",
        "allow_write": False,
    }


def test_execute_agent_rejects_empty_task() -> None:
    app = FastAPI()
    app.include_router(aiops.router, prefix="/api")

    with TestClient(app) as client:
        response = client.post(
            "/api/agent/execute",
            json={"session_id": "demo-001", "task": ""},
        )

    assert response.status_code == 422
