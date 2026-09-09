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

from .state import PlanExecuteState
from .utils import format_tools_description


class Plan(BaseModel):
    """计划的输出格式"""

    steps: list[str] = Field(
        description="完成任务所需的不同步骤。这些步骤应该按顺序执行，每一步都建立在前一步的基础上。"
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

        # 步骤2: 获取可用工具列表
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

        # 步骤3: 格式化经验文档上下文
        if experience_docs:
            experience_context = dedent(f"""
                ## 相关经验文档

                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验制定执行计划：

                {experience_docs}

                ---
            """).strip()
        else:
            experience_context = ""

        # 步骤4: 创建 LLM 并生成计划
        llm = ChatQwen(model=config.rag_model, api_key=config.dashscope_api_key, temperature=0)

        planner_chain = planner_prompt | llm.with_structured_output(Plan)

        # 调用 LLM 生成计划
        # structured output 偶发返回 None（LLM 抖动），最多重试一次
        plan_steps: list[str] = []
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
            elif isinstance(plan_result, dict):
                # 如果返回的是字典，提取 steps 字段
                plan_steps = plan_result.get("steps", [])  # type: ignore

            if plan_steps:
                break
            logger.warning(f"LLM 未返回有效计划（第 {attempt + 1}/2 次）")

        if not plan_steps:
            raise ValueError("LLM 连续返回空计划")

        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

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
