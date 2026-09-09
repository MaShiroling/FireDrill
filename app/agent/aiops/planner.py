"""
Planner 节点：制定执行计划
基于 LangGraph 官方教程实现
"""

from textwrap import dedent
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.observability import trace_event
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS, retrieve_knowledge

from .memory_context import load_planner_memory_context
from .state import PlanExecuteState
from .utils import format_tools_description


class Plan(BaseModel):
    """计划的输出格式"""

    steps: list[str] = Field(
        description="完成任务所需的不同步骤。这些步骤应该按顺序执行，每一步都建立在前一步的基础上。"
    )
    memory_ids_used: list[str] = Field(
        default_factory=list,
        description="本计划实际参考的长期记忆 memory_id；未使用长期记忆时返回空列表。",
    )


# Planner 提示词
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定计划，实际的工具调用由 Executor 负责执行。

                {experience_context}

                {memory_context}

                对于给定的任务，请创建一个简单的、逐步的计划来完成它。计划应该：
                - 将任务分解为逻辑上独立的步骤
                - 每个步骤应该明确使用哪些工具(如果需要工具的话)来获取信息, 最好能同时提供工具执行所需要的参数
                - 步骤之间应该有清晰的依赖关系
                - 步骤描述要具体、可操作
                - 正常配置变更控制在 4-6 步；工具层会处理明确的临时错误，不要把每次重试拆成独立计划步骤
                - 用户给出 rule-003 形式的规则 ID 时，所有查询、修改、删除步骤必须原样携带该 ID
                - 只有规则名称但没有规则 ID 时，先用 list_firewall_rules 查询真实 ID；不得把名称当作 rule_id
                - 新增规则的 ID 由 add_firewall_rule 返回，后续步骤必须复用实际返回值，禁止预先猜测 ID
                - **如果有相关经验文档，请参考其中的方法和步骤制定计划**
                - 如果实际参考了长期记忆，只能在 memory_ids_used 中填写上下文提供的 memory_id；未参考则返回空列表

                示例输入："分析当前系统的性能问题"
                示例输出（假设有对应工具）：
                步骤1: 使用 get_metrics 工具收集系统的 CPU 和内存使用情况
                步骤2: 使用 query_logs 工具检查最近的错误日志
                步骤3: 使用 query_database 工具分析慢查询日志
                步骤4: 综合以上信息生成性能分析报告
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def planner(state: PlanExecuteState) -> dict[str, Any]:
    """
    规划节点：根据用户输入生成执行计划

    流程：
    1. 先查询内部文档，获取相关经验和最佳实践
    2. 基于经验文档和可用工具制定执行计划
    """
    logger.info("=== Planner：制定执行计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")
    trace_event("node_started", node="planner", data={"input": input_text})

    try:
        # 步骤1: 查询内部文档获取相关经验
        logger.info("查询内部文档，寻找相关经验...")
        experience_docs = ""
        try:
            # retrieve_knowledge 使用 response_format="content_and_artifact"
            # ainvoke() 只返回 content（字符串），不是元组
            context_str = await retrieve_knowledge.ainvoke({"query": input_text})
            if context_str and context_str.strip():
                experience_docs = context_str
                logger.info(f"找到相关经验文档，长度: {len(experience_docs)}")
                trace_event(
                    "knowledge_retrieval_completed",
                    node="planner",
                    data={"found": True, "content_length": len(experience_docs)},
                )
            else:
                logger.info("未找到相关经验文档")
                trace_event(
                    "knowledge_retrieval_completed",
                    node="planner",
                    data={"found": False, "content_length": 0},
                )
        except Exception as e:
            logger.warning(f"查询内部文档失败: {e}")
            trace_event("knowledge_retrieval_failed", node="planner", data={"error": str(e)})

        # 步骤2: 按开关检索经过审核的长期记忆。记忆不可作为当前状态证据。
        memory_context = await load_planner_memory_context(
            input_text,
            enabled=config.agent_memory_enabled,
            tenant_id=state.get("memory_tenant_id", config.agent_memory_tenant_id),
            device_type=state.get("memory_device_type", config.agent_memory_device_type),
            top_k=config.agent_memory_top_k,
            max_chars=config.agent_memory_context_max_chars,
            timeout_s=config.agent_memory_retrieval_timeout_s,
        )
        if not config.agent_memory_enabled:
            trace_event(
                "memory_retrieval_skipped",
                node="planner",
                data={"reason": "disabled"},
            )
        elif memory_context.timed_out:
            trace_event(
                "memory_retrieval_timed_out",
                node="planner",
                data={
                    "mode": memory_context.mode,
                    "recalled_ids": list(memory_context.recalled_ids),
                    "injected_ids": list(memory_context.injected_ids),
                    "injected_count": len(memory_context.injected_ids),
                    "fallback_succeeded": memory_context.mode != "unavailable",
                    "error": memory_context.degraded_reason,
                },
            )
        elif memory_context.degraded_reason and not memory_context.recalled_ids:
            trace_event(
                "memory_retrieval_failed",
                node="planner",
                data={
                    "mode": memory_context.mode,
                    "error": memory_context.degraded_reason,
                },
            )
        else:
            trace_event(
                "memory_retrieval_completed",
                node="planner",
                data={
                    "mode": memory_context.mode,
                    "recalled_ids": list(memory_context.recalled_ids),
                    "injected_ids": list(memory_context.injected_ids),
                    "injected_count": len(memory_context.injected_ids),
                    "content_length": len(memory_context.text),
                    "truncated": memory_context.truncated,
                    "degraded": bool(memory_context.degraded_reason),
                },
            )

        # 步骤3: 获取可用工具列表
        # 获取本地工具
        local_tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")
        trace_event(
            "tool_inventory_loaded",
            node="planner",
            data={
                "local_tools": [getattr(tool, "name", str(tool)) for tool in local_tools],
                "mcp_tools": [getattr(tool, "name", str(tool)) for tool in mcp_tools],
            },
        )

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)

        # 步骤4: 格式化经验文档上下文
        if experience_docs:
            experience_context = dedent(f"""
                ## 相关经验文档

                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验制定执行计划：

                {experience_docs}

                ---
            """).strip()
        else:
            experience_context = ""

        # 步骤5: 创建 LLM 并生成计划
        llm = ChatQwen(model=config.rag_model, api_key=config.dashscope_api_key, temperature=0)

        planner_chain = planner_prompt | llm.with_structured_output(Plan)

        # 调用 LLM 生成计划
        # structured output 偶发返回 None（LLM 抖动），最多重试一次
        plan_steps: list[str] = []
        reported_memory_ids: list[str] = []
        for attempt in range(2):
            trace_event(
                "model_call_started",
                node="planner",
                data={"purpose": "create_plan", "attempt": attempt + 1},
            )
            plan_result = await planner_chain.ainvoke(
                {
                    "messages": [("user", input_text)],
                    "tools_description": tools_description,
                    "experience_context": experience_context,
                    "memory_context": memory_context.text,
                }
            )
            trace_event(
                "model_call_completed",
                node="planner",
                data={"purpose": "create_plan", "attempt": attempt + 1},
            )

            # 提取步骤列表
            if isinstance(plan_result, Plan):
                plan_steps = plan_result.steps
                reported_memory_ids = plan_result.memory_ids_used
            elif isinstance(plan_result, dict):
                # 如果返回的是字典，提取 steps 字段
                plan_steps = plan_result.get("steps", [])  # type: ignore
                raw_memory_ids = plan_result.get("memory_ids_used", [])
                reported_memory_ids = (
                    [memory_id for memory_id in raw_memory_ids if isinstance(memory_id, str)]
                    if isinstance(raw_memory_ids, list)
                    else []
                )

            if plan_steps:
                break
            logger.warning(f"LLM 未返回有效计划（第 {attempt + 1}/2 次）")

        if not plan_steps:
            raise ValueError("LLM 连续返回空计划")

        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

        allowed_memory_ids = set(memory_context.injected_ids)
        used_memory_ids = list(
            dict.fromkeys(
                memory_id for memory_id in reported_memory_ids if memory_id in allowed_memory_ids
            )
        )
        invalid_memory_ids = list(
            dict.fromkeys(
                memory_id
                for memory_id in reported_memory_ids
                if memory_id not in allowed_memory_ids
            )
        )
        trace_event(
            "memory_usage_reported",
            node="planner",
            data={
                "source": "model_self_report",
                "used_ids": used_memory_ids,
                "invalid_ids": invalid_memory_ids,
                "injected_ids": list(memory_context.injected_ids),
            },
        )

        trace_event("node_completed", node="planner", data={"plan": plan_steps, "fallback": False})

        return {"plan": plan_steps}

    except Exception as e:
        logger.error(f"生成计划失败: {e}", exc_info=True)
        fallback_plan = ["收集相关信息", "分析数据", "生成报告"]
        trace_event(
            "node_failed",
            node="planner",
            data={"error": str(e), "fallback_plan": fallback_plan},
        )
        # 返回一个默认计划
        return {"plan": fallback_plan}
