"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.observability import trace_event
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS

from .model_retry import invoke_model_with_retry
from .state import PlanExecuteState
from .utils import format_execution_context


async def executor(state: PlanExecuteState) -> dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤

    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        trace_event("node_skipped", node="executor", data={"reason": "empty_plan"})
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")
    trace_event(
        "node_started",
        node="executor",
        data={"step": task, "remaining_steps_before": len(plan)},
    )

    try:
        # 获取本地工具
        local_tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        trace_event(
            "tool_inventory_loaded",
            node="executor",
            data={
                "local_tools": [getattr(tool, "name", str(tool)) for tool in local_tools],
                "mcp_tools": [getattr(tool, "name", str(tool)) for tool in mcp_tools],
            },
        )

        # 创建 LLM（绑定工具）
        llm = ChatQwen(model=config.rag_model, api_key=config.dashscope_api_key, temperature=0)
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        execution_context = format_execution_context(
            state.get("input", ""), state.get("past_steps", [])
        )

        # 当前步骤是执行目标；原始任务和近期结果只作为标识符与真实状态上下文。
        messages = [
            SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要擅自执行后续步骤
- 严格遵守工具参数 Schema，不得把规则名称填写到 rule_id 参数
- rule_id 必须使用用户明确给出的值或此前工具真实返回的值（格式如 rule-003），禁止猜测或生成 new-rule-001 等虚假 ID
- 只有规则名称而没有 rule_id 时，应先调用 list_firewall_rules 查出真实 ID，再在后续步骤使用
- get_firewall_rule、update_firewall_rule、delete_firewall_rule、move_firewall_rule 的规则定位参数都是 rule_id
- 已执行历史仅用于复用真实结果；如果历史已显示提交成功，不要重复 commit_config"""),
            HumanMessage(content=f"{execution_context}\n\n当前只执行这一步：\n{task}"),
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await invoke_model_with_retry(
            lambda: llm_with_tools.ainvoke(messages),
            node="executor",
            purpose="select_tools",
            max_attempts=config.agent_model_max_attempts,
            delay_s=config.agent_model_retry_delay_s,
        )
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")
            for tool_call in llm_response.tool_calls:
                trace_event(
                    "tool_call_requested",
                    node="executor",
                    data={
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args", {}),
                        "step": task,
                    },
                )

            # 使用 ToolNode 自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})
            for tool_message in tool_messages["messages"]:
                trace_event(
                    "tool_call_completed",
                    node="executor",
                    data={
                        "tool_call_id": getattr(tool_message, "tool_call_id", None),
                        "name": getattr(tool_message, "name", None),
                        "status": getattr(tool_message, "status", "success"),
                        "content": getattr(tool_message, "content", str(tool_message)),
                    },
                )

            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await invoke_model_with_retry(
                lambda: llm_with_tools.ainvoke(messages),
                node="executor",
                purpose="summarize_step",
                max_attempts=config.agent_model_max_attempts,
                delay_s=config.agent_model_retry_delay_s,
            )
            result = (
                final_response.content
                if hasattr(final_response, "content")
                else str(final_response)
            )
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            trace_event("no_tool_selected", node="executor", data={"step": task})
            result = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")
        trace_event(
            "node_completed",
            node="executor",
            data={"step": task, "result": result, "remaining_steps_after": len(plan) - 1},
        )

        # 返回更新：移除已执行的步骤，添加执行历史
        return {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        trace_event("node_failed", node="executor", data={"step": task, "error": str(e)})
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }
