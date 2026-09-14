"""
AIOps 请求模型
"""

from pydantic import BaseModel, ConfigDict, Field


class AIOpsRequest(BaseModel):
    """AIOps 诊断请求"""

    model_config = ConfigDict(
        json_schema_extra={"example": {"session_id": "session-123"}},
    )

    session_id: str | None = Field(
        default="default",
        description="会话ID，用于追踪诊断历史",
    )


class AgentExecuteRequest(BaseModel):
    """通用 Plan-Execute-Replan Agent 执行请求。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "demo-001",
                "task": "请放通 trust 区访问 dmz 区的 TCP 22 流量，提交并验证。",
                "tenant_id": "local",
                "device_type": "firewall",
                "allow_write": True,
            }
        },
    )

    session_id: str = Field(
        ...,
        min_length=1,
        description="会话ID，用作 LangGraph thread_id 和执行轨迹标识",
    )
    task: str = Field(
        ...,
        min_length=1,
        description="需要 Agent 规划并执行的完整任务描述",
    )
    tenant_id: str = Field(
        default="local",
        min_length=1,
        description="长期记忆检索使用的租户隔离标识",
    )
    device_type: str = Field(
        default="firewall",
        min_length=1,
        description="长期记忆检索使用的设备类型",
    )
    allow_write: bool = Field(
        default=True,
        description="是否允许调用配置写工具；自动路由的只读任务必须传 false",
    )
