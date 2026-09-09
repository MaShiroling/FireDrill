"""
AIOps 请求模型
"""

from typing import Optional
from pydantic import BaseModel, Field


class AIOpsRequest(BaseModel):
    """AIOps 诊断请求"""

    session_id: Optional[str] = Field(
        default="default",
        description="会话ID，用于追踪诊断历史"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-123"
            }
        }
