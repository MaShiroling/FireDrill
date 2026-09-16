"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

import operator
from typing import Annotated, NotRequired, TypedDict


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 状态"""

    # 用户输入（任务描述）
    input: str

    # 执行计划（步骤列表）
    plan: list[str]

    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[list[tuple], operator.add]

    # 最终响应/报告
    response: str

    # 长期记忆隔离域；由服务入口设置，Planner 只检索该租户和设备类型。
    memory_tenant_id: NotRequired[str]
    memory_device_type: NotRequired[str]

    # 工具执行安全策略；False 时 Executor 硬拦截所有配置写工具。
    allow_write: NotRequired[bool]

    # 防火墙提交守卫使用的、由查询工具真实观测到的状态。
    # 这些值只能从工具响应中更新，不能由模型自由生成。
    latest_running_revision: NotRequired[int]
    candidate_base_revision: NotRequired[int]
    change_set_id: NotRequired[str]
