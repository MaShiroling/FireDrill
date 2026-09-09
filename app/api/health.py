"""健康检查接口"""

from typing import Any
from fastapi import APIRouter, Response
from app.config import config
from app.core.milvus_client import milvus_manager
from app.models.response import HealthResponse
from loguru import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(response: Response):
    """健康检查接口
    检查服务状态和数据库连接状态

    Returns:
        HealthResponse: 健康检查结果（Milvus 不可用时 HTTP 状态码为 503）
    """
    # 检查服务基本状态
    health_data: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy"
    }

    # 检查 Milvus 连接状态
    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status: str = "connected" if milvus_healthy else "disconnected"
        milvus_message: str = "Milvus 连接正常" if milvus_healthy else "Milvus 连接异常"
        health_data["milvus"] = {
            "status": milvus_status,
            "message": milvus_message
        }
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus 检查失败: {str(e)}"
        }

    # 判断整体健康状态：Milvus 不可用则服务不可用（503）
    overall_status = "healthy"
    status_code = 200

    if health_data["milvus"]["status"] != "connected":
        overall_status = "unhealthy"
        status_code = 503
        health_data["error"] = "数据库不可用"

    health_data["status"] = overall_status

    response.status_code = status_code
    return HealthResponse(
        code=status_code,
        message="服务运行正常" if overall_status == "healthy" else "服务不可用",
        data=health_data
    )
