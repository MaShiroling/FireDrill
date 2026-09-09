"""响应数据模型

定义 API 响应的 Pydantic 模型
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Optional


class ChatData(BaseModel):
    """对话响应数据载荷"""

    success: bool = Field(..., description="是否成功")
    answer: Optional[str] = Field(None, description="AI 回答")
    error_message: Optional[str] = Field(None, description="错误信息", alias="errorMessage")

    model_config = ConfigDict(populate_by_name=True)


class ChatResponse(BaseModel):
    """对话响应（与 /api/chat 实际返回结构一致）"""

    code: int = Field(..., description="业务状态码")
    message: str = Field(..., description="状态描述")
    data: ChatData = Field(..., description="响应数据")


class SessionInfoResponse(BaseModel):
    """会话信息响应"""

    session_id: str = Field(..., description="会话 ID")
    message_count: int = Field(..., description="消息数量")
    history: List[Dict[str, str]] = Field(..., description="历史消息列表")


class ApiResponse(BaseModel):
    """通用 API 响应"""

    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    data: Optional[Any] = Field(None, description="数据")


class HealthResponse(BaseModel):
    """健康检查响应（与 /health 实际返回结构一致）"""

    code: int = Field(..., description="HTTP 状态码")
    message: str = Field(..., description="状态描述")
    data: Dict[str, Any] = Field(..., description="健康检查明细（服务信息、Milvus 状态等）")
