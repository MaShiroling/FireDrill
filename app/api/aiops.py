"""
AIOps 智能运维接口
"""

import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models.aiops import AIOpsRequest
from app.services.aiops_service import aiops_service

router = APIRouter()


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    """
    AIOps 故障诊断接口（流式 SSE）

    **功能说明：**
    - 基于 Plan-Execute-Replan 模式诊断（Planner 定计划 → Executor 逐步执行 → Replanner 评估决策）
    - 诊断所需数据由 Agent 通过工具自行获取（Prometheus 告警、监控指标、日志、知识库）
    - 返回 SSE 格式，event 恒为 message，data 字段为 JSON 字符串

    **SSE 事件类型：**

    1. `status` - 节点运行状态
       ```json
       {"type": "status", "stage": "planner|executor|replanner", "message": "..."}
       ```

    2. `plan` - 诊断计划制定完成
       ```json
       {"type": "plan", "stage": "plan_created", "message": "执行计划已制定，共 N 个步骤", "plan": ["步骤1", "步骤2"]}
       ```

    3. `step_complete` - 步骤执行完成（每个步骤一条）
       ```json
       {"type": "step_complete", "stage": "step_executed", "message": "步骤执行完成 (2/6)", "current_step": "...", "remaining_steps": 4}
       ```

    4. `report` - 最终诊断报告（Markdown 文本）
       ```json
       {"type": "report", "stage": "final_report", "message": "最终报告已生成", "report": "# 告警分析报告\\n..."}
       ```

    5. `complete` - 诊断完成（收到后应关闭连接）
       ```json
       {"type": "complete", "stage": "diagnosis_complete", "message": "诊断流程完成", "diagnosis": {"status": "completed", "report": "..."}}
       ```

    6. `error` - 错误信息
       ```json
       {"type": "error", "stage": "error", "message": "..."}
       ```

    **使用示例：**
    ```bash
    curl -X POST "http://localhost:9900/api/aiops" \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "session-123"}' \\
      --no-buffer
    ```

    注意：本接口为 POST，浏览器原生 EventSource 只支持 GET，前端请使用
    fetch + ReadableStream 手动解析 SSE 帧（参考 static/app.js）。

    Args:
        request: AIOps 诊断请求

    Returns:
        SSE 事件流
    """
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式）")

    async def event_generator():
        try:
            async for event in aiops_service.diagnose(session_id=session_id):
                # 发送事件
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False)
                }

                # 如果是完成或错误事件，结束流
                if event.get("type") in ["complete", "error"]:
                    break

            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"诊断异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())
