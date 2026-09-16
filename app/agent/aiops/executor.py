"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

# 这些工具会修改 Candidate/Running 状态。只要同一批 tool_calls 中出现任意
# 写操作，整批调用就必须按模型给出的顺序串行执行，避免 commit 与后续核实
# 同时发出，产生“核实读到提交前状态”的竞态。
STATEFUL_TOOL_NAMES = frozenset(
    {
        "add_firewall_rule",
        "update_firewall_rule",
        "delete_firewall_rule",
        "move_firewall_rule",
        "commit_config",
        "discard_candidate",
    }
)

REVISION_SOURCE_TOOL_NAMES = frozenset(
    {
        "get_firewall_overview",
        "get_config_diff",
        "commit_config",
        "discard_candidate",
    }
)


class RevisionUnavailableError(RuntimeError):
    """Raised when a commit is attempted without an observed firewall revision."""


class CandidateRevisionConflictError(RuntimeError):
    """Raised when the observed Running and Candidate base revisions are stale."""


def requires_sequential_tool_execution(tool_calls: list[dict[str, Any]]) -> bool:
    """Return whether a model-selected tool batch has state-order dependencies."""
    return len(tool_calls) > 1 and any(
        str(tool_call.get("name", "")) in STATEFUL_TOOL_NAMES for tool_call in tool_calls
    )


def blocked_write_tools(
    tool_calls: list[dict[str, Any]], *, allow_write: bool
) -> list[str]:
    """返回被只读执行策略拦截的配置写工具。"""
    if allow_write:
        return []
    return [
        str(tool_call.get("name", ""))
        for tool_call in tool_calls
        if str(tool_call.get("name", "")) in STATEFUL_TOOL_NAMES
    ]


async def execute_selected_tools(
    tool_node: ToolNode,
    messages: list[Any],
    llm_response: Any,
    *,
    revision_context: dict[str, Any] | None = None,
    step: str = "",
) -> tuple[list[Any], str]:
    """Execute tool calls in parallel or sequentially according to state dependencies.

    ``ToolNode`` executes multiple calls from one ``AIMessage`` concurrently. For a
    stateful batch, create one AIMessage per call so every ToolMessage is completed
    before the next call starts. The caller still appends the original AIMessage and
    all returned ToolMessages to the model conversation, preserving tool-call IDs.
    """
    tool_calls = list(llm_response.tool_calls)
    revision_context = revision_context if revision_context is not None else {}
    if not requires_sequential_tool_execution(tool_calls):
        for tool_call in tool_calls:
            apply_commit_revision_guard(tool_call, revision_context)
            _trace_tool_call_requested(tool_call, step)
        result = await tool_node.ainvoke({"messages": [*messages, llm_response]})
        tool_messages = list(result["messages"])
        update_revision_context(tool_messages, revision_context)
        return tool_messages, "parallel"

    tool_messages: list[Any] = []
    for tool_call in tool_calls:
        apply_commit_revision_guard(tool_call, revision_context)
        _trace_tool_call_requested(tool_call, step)
        single_call_response = AIMessage(content="", tool_calls=[tool_call])
        result = await tool_node.ainvoke({"messages": [*messages, single_call_response]})
        current_messages = list(result["messages"])
        tool_messages.extend(current_messages)
        # 让同一批中排在 get_config_diff 后面的 commit_config 也能使用
        # 刚刚查询到的真实 Revision。
        update_revision_context(current_messages, revision_context)
    return tool_messages, "sequential"


def apply_commit_revision_guard(
    tool_call: dict[str, Any], revision_context: dict[str, Any]
) -> None:
    """Inject the observed revision into commit_config before tool execution."""
    if str(tool_call.get("name", "")) != "commit_config":
        return

    running_revision = revision_context.get("latest_running_revision")
    candidate_base_revision = revision_context.get("candidate_base_revision")
    if running_revision is None:
        trace_event(
            "commit_revision_guard",
            node="executor",
            data={"action": "blocked", "reason": "revision_unavailable"},
        )
        raise RevisionUnavailableError(
            "commit_config 已被 Revision 守卫拦截：提交前必须先调用 "
            "get_config_diff 或 get_firewall_overview 获取真实 Running Revision"
        )

    if (
        candidate_base_revision is not None
        and candidate_base_revision != running_revision
    ):
        trace_event(
            "commit_revision_guard",
            node="executor",
            data={
                "action": "blocked",
                "reason": "candidate_is_stale",
                "running_revision": running_revision,
                "candidate_base_revision": candidate_base_revision,
            },
        )
        raise CandidateRevisionConflictError(
            "commit_config 已被 Revision 守卫拦截："
            f"Running 为 R{running_revision}，Candidate 基于 R{candidate_base_revision}，"
            "需要重新读取状态并重规划"
        )

    args = tool_call.setdefault("args", {})
    model_revision = args.get("expected_revision")
    args["expected_revision"] = running_revision
    action = "preserved" if model_revision == running_revision else "overridden"
    logger.info(
        "Commit Revision 守卫: model_revision={}, actual_revision={}, action={}",
        model_revision,
        running_revision,
        action,
    )
    trace_event(
        "commit_revision_guard",
        node="executor",
        data={
            "action": action,
            "model_revision": model_revision,
            "actual_revision": running_revision,
            "candidate_base_revision": candidate_base_revision,
            "change_set_id": revision_context.get("change_set_id"),
        },
    )


def update_revision_context(
    tool_messages: list[Any], revision_context: dict[str, Any]
) -> None:
    """Update revision state only from trusted firewall tool responses."""
    for tool_message in tool_messages:
        name = str(getattr(tool_message, "name", "") or "")
        if name not in REVISION_SOURCE_TOOL_NAMES:
            continue
        payload = _parse_tool_payload(getattr(tool_message, "content", None))
        if not payload or payload.get("success") is False:
            continue

        running_revision = payload.get("running_revision")
        candidate_base_revision = payload.get("candidate_base_revision")
        change_set_id = payload.get("change_set_id")
        if isinstance(running_revision, int):
            revision_context["latest_running_revision"] = running_revision
        if isinstance(candidate_base_revision, int):
            revision_context["candidate_base_revision"] = candidate_base_revision
        if isinstance(change_set_id, str) and change_set_id:
            revision_context["change_set_id"] = change_set_id


def _parse_tool_payload(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                payload = _parse_tool_payload(block["text"])
                if payload is not None:
                    return payload
        return None
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _trace_tool_call_requested(tool_call: dict[str, Any], step: str) -> None:
    trace_event(
        "tool_call_requested",
        node="executor",
        data={
            "tool_call_id": tool_call.get("id"),
            "name": tool_call.get("name"),
            "args": tool_call.get("args", {}),
            "step": step,
        },
    )


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
- 如果当前步骤确实需要多个有先后依赖的工具，必须按业务顺序生成 tool_calls；配置写操作、提交和提交后验证不得颠倒
- 严格遵守工具参数 Schema，不得把规则名称填写到 rule_id 参数
- rule_id 必须使用用户明确给出的值或此前工具真实返回的值（格式如 rule-003），禁止猜测或生成 new-rule-001 等虚假 ID
- 只有规则名称而没有 rule_id 时，应先调用 list_firewall_rules 查出真实 ID，再在后续步骤使用
- get_firewall_rule、update_firewall_rule、delete_firewall_rule、move_firewall_rule 的规则定位参数都是 rule_id
- test_traffic 的 src_addr 和 dst_addr 必须传单个主机 IP，禁止传 10.1.9.0/24 形式的 CIDR；
  若任务只给出网段，选择该网段内的合法主机 IP（如 10.1.9.1）作为模拟报文地址
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
            blocked_tools = blocked_write_tools(
                llm_response.tool_calls,
                allow_write=state.get("allow_write", True),
            )
            if blocked_tools:
                result = (
                    "安全策略已拦截配置写操作："
                    f"{', '.join(blocked_tools)}。本次任务只允许调用只读工具。"
                )
                logger.warning(result)
                trace_event(
                    "write_tools_blocked",
                    node="executor",
                    data={"tool_names": blocked_tools, "step": task},
                )
                return {
                    "plan": plan[1:],
                    "past_steps": [(task, result)],
                }
            # ToolNode 默认并发执行同一 AIMessage 中的多个调用。配置写操作参与时，
            # 改为逐个调用，保证 commit 完成后才开始状态核实和流量验证。
            revision_context = {
                key: state[key]
                for key in (
                    "latest_running_revision",
                    "candidate_base_revision",
                    "change_set_id",
                )
                if key in state
            }
            tool_messages, execution_mode = await execute_selected_tools(
                tool_node,
                messages,
                llm_response,
                revision_context=revision_context,
                step=task,
            )
            trace_event(
                "tool_execution_mode_selected",
                node="executor",
                data={
                    "mode": execution_mode,
                    "tool_names": [call.get("name") for call in llm_response.tool_calls],
                    "reason": (
                        "stateful_tool_dependency"
                        if execution_mode == "sequential"
                        else "read_only_or_single_call"
                    ),
                },
            )
            for tool_message in tool_messages:
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
            messages.append(llm_response)
            messages.extend(tool_messages)
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
        state_update: dict[str, Any] = {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
        }
        for key in (
            "latest_running_revision",
            "candidate_base_revision",
            "change_set_id",
        ):
            if "revision_context" in locals() and key in revision_context:
                state_update[key] = revision_context[key]
        return state_update

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        trace_event("node_failed", node="executor", data={"step": task, "error": str(e)})
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }
