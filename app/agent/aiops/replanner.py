"""
Replanner 节点：重新规划或生成最终响应
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
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS

from .state import PlanExecuteState
from .utils import format_tools_description


class Response(BaseModel):
    """最终响应的格式"""

    response: str = Field(description="对用户的最终响应")


class Act(BaseModel):
    """重新规划的输出格式"""

    action: str = Field(description="""下一步的行动，必须是以下三种之一：
        - 'continue': 当前计划合理，继续执行下一个步骤
        - 'replan': 当前计划需要调整，提供新的步骤列表
        - 'respond': 计划已完成且信息充足，生成最终响应""")
    # action 为 'replan' 时，新的步骤列表（会替换当前剩余计划）
    new_steps: list[str] = Field(
        default_factory=list,
        description="新的步骤列表（如果 action 是 'replan'，这些步骤会替换剩余计划）",
    )


# Replanner 提示词
replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

                你有三个选择（按优先级排序）：

                **0. 任务完成守卫 - 凌驾于以下所有决策之上**
                   - 检查原始任务要求的目标动作是否已在「已执行步骤的结果」中真正完成
                   - 尤其对变更类任务（添加/修改/删除/提交/验证配置）：只要剩余计划中还承载着
                     任务要求的动作（如下发后尚未 commit、commit 后尚未验证），禁止 respond
                   - "计划里写过"不等于"已完成"，以已执行步骤的实际结果为准

                **1. 'respond' - 任务目标已实际达成，生成最终响应**
                   - 使用场景：原始任务的目标已通过「已执行步骤的结果」真正达成
                   - 决策标准：
                     * 变更类任务：改动已下发并提交生效，且已验证（缺一不可）
                     * 查询类任务：已执行步骤获取了回答所需的全部关键信息
                     * 或者已执行步骤 >= 5（无论结果如何，防止无限执行）
                   - ⚠️ 对变更类任务，"信息足够"不等于"任务完成"

                **2. 'continue' - 当前计划合理，继续执行** 【次优先级】
                   - 使用场景：剩余计划合理且必要
                   - 决策标准：剩余步骤确实能提供关键信息
                   - ⚠️ 如果剩余步骤不是"必需"的，应选择 respond

                **3. 'replan' - 当前计划有严重问题** 【最低优先级，谨慎使用】
                   - 使用场景：原计划明显错误或遗漏关键步骤
                   - ⚠️ **严格限制**：
                     * 新步骤数量必须 <= 剩余执行预算（最多 8 步减去已执行步骤数）
                     * 优先简化计划，不要添加不必要的步骤
                     * 总步骤数已执行 >= 5 次时，禁止 replan，只能 respond

                评估标准：
                - 原始任务要求的目标动作是否已在已执行步骤中真正完成？【最关键】
                - 剩余步骤是否承载着任务要求但尚未执行的动作？如果是，必须 continue
                - 已执行步骤数是否过多（>= 5）？如果是，立即 respond

                **决策优先级口诀：**
                "先看任务是否真的完成 > 保持不变 > 调整计划"
                "变更类任务：未提交、未验证，就不算完成"
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 修复前的旧版提示词（commit 6ac25dd），仅用于 A/B 对照实验（REPLANNER_LEGACY=1）
replanner_prompt_legacy = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

                你有三个选择（按优先级排序）：

                **1. 'respond' - 信息充足，立即生成最终响应** 【最高优先级】
                   - 使用场景：当前信息已经足够回答用户问题
                   - 决策标准：
                     * 已执行步骤 >= 3 且获取了关键信息
                     * 或者已执行步骤 >= 5（无论结果如何）
                     * 或者当前信息完全满足任务需求
                   - ⚠️ 不要等到"完美"才响应，"足够好"就应该立即 respond

                **2. 'continue' - 当前计划合理，继续执行** 【次优先级】
                   - 使用场景：剩余计划合理且必要
                   - 决策标准：剩余步骤确实能提供关键信息
                   - ⚠️ 如果剩余步骤不是"必需"的，应选择 respond

                **3. 'replan' - 当前计划有严重问题** 【最低优先级，谨慎使用】
                   - 使用场景：原计划明显错误或遗漏关键步骤
                   - ⚠️ **严格限制**：
                     * 新步骤数量必须 <= 当前剩余步骤数
                     * 优先简化计划，不要添加不必要的步骤
                     * 总步骤数已执行 >= 5 次时，禁止 replan，只能 respond

                评估标准：
                - 当前信息是否已经足够解决用户问题？【最关键】
                - 已执行步骤是否成功获取了核心信息？
                - 剩余步骤是否真的"必需"？
                - 已执行步骤数是否过多（>= 5）？如果是，立即 respond

                **决策优先级口诀：**
                "优先结束 > 保持不变 > 调整计划"
                "信息足够就响应，不要追求完美"
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 最终响应生成提示词
response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

                响应要求：
                - 清晰、结构化
                - 基于实际数据，不要编造
                - 如果某些步骤失败，要诚实说明
                - 使用 Markdown 格式
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def replanner(state: PlanExecuteState) -> dict[str, Any]:
    """
    重新规划节点：决定是继续、调整计划还是生成最终响应

    三种决策：
    1. continue - 继续执行当前计划
    2. replan - 调整计划（替换剩余步骤）
    3. respond - 生成最终响应
    """
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    logger.info(f"剩余计划步骤: {len(plan)}")
    logger.info(f"已执行步骤: {len(past_steps)}")
    trace_event(
        "node_started",
        node="replanner",
        data={
            "remaining_plan": plan,
            "completed_steps": len(past_steps),
            "legacy": config.replanner_legacy,
        },
    )

    # ⚠️ 强制限制：如果已执行步骤过多，直接生成响应
    MAX_STEPS = 8
    if len(past_steps) >= MAX_STEPS:
        logger.warning(
            f"已执行 {len(past_steps)} 个步骤，超过最大限制 {MAX_STEPS}，强制生成最终响应"
        )
        trace_event(
            "replanner_decision",
            node="replanner",
            data={
                "action": "respond",
                "forced": True,
                "reason": "max_steps_reached",
                "completed_steps": len(past_steps),
                "max_steps": MAX_STEPS,
            },
        )
        llm = ChatQwen(model=config.rag_model, api_key=config.dashscope_api_key, temperature=0)
        return await _generate_response(state, llm)

    # 获取可用工具列表
    try:
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
            node="replanner",
            data={
                "local_tools": [getattr(tool, "name", str(tool)) for tool in local_tools],
                "mcp_tools": [getattr(tool, "name", str(tool)) for tool in mcp_tools],
            },
        )

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)
    except Exception as e:
        logger.warning(f"获取工具列表失败: {e}")
        tools_description = "无法获取工具列表"
        trace_event("tool_inventory_failed", node="replanner", data={"error": str(e)})

    # 创建 LLM
    llm = ChatQwen(model=config.rag_model, api_key=config.dashscope_api_key, temperature=0)

    # 格式化已执行的步骤
    steps_summary = "\n".join(
        [f"步骤: {step}\n结果: {result[:300]}..." for step, result in past_steps]
    )

    # 如果还有剩余计划，进行决策
    if plan:
        logger.info("还有剩余计划，评估下一步行动")

        legacy = config.replanner_legacy
        if legacy:
            logger.warning("REPLANNER_LEGACY=1：使用修复前的旧提示词（A/B 对照模式）")

        prompt = replanner_prompt_legacy if legacy else replanner_prompt
        replanner_chain = prompt | llm.with_structured_output(Act)

        try:
            if legacy:
                hint = (
                    f"⚠️ 重要提示：已执行 {len(past_steps)} 个步骤，"
                    "请优先考虑是否信息已足够生成响应（respond）"
                )
            else:
                hint = (
                    f"⚠️ 重要提示：已执行 {len(past_steps)} 个步骤。"
                    "若剩余步骤仍承载任务要求的动作（如提交、验证），请选择 continue；"
                    "仅当任务目标已实际达成时才选择 respond"
                )
            messages = [
                ("user", f"原始任务: {input_text}"),
                ("user", f"已执行的步骤:\n{steps_summary}"),
                ("user", f"剩余计划: {', '.join(plan)}"),
                ("user", hint),
            ]

            trace_event(
                "model_call_started",
                node="replanner",
                data={"purpose": "decide_next_action"},
            )
            act = await replanner_chain.ainvoke(
                {"messages": messages, "tools_description": tools_description}
            )
            trace_event(
                "model_call_completed",
                node="replanner",
                data={"purpose": "decide_next_action"},
            )

            # 处理返回结果
            if act is None:
                # structured output 偶发返回 None（LLM 抖动），继续执行剩余计划
                logger.warning("LLM 未返回有效决策，继续执行剩余计划")
                trace_event(
                    "replanner_decision",
                    node="replanner",
                    data={
                        "action": "continue",
                        "forced": True,
                        "reason": "empty_structured_output",
                    },
                )
                return {}

            if isinstance(act, Act):
                action = act.action
                new_steps = act.new_steps
            else:
                # 如果返回的是字典
                action = act.get("action", "continue")  # type: ignore
                new_steps = act.get("new_steps", [])  # type: ignore

            logger.info(f"Replanner 决策: {action}")

            if action == "respond":
                logger.info("决定生成最终响应")
                trace_event(
                    "replanner_decision",
                    node="replanner",
                    data={"action": "respond", "forced": False, "new_steps": []},
                )
                return await _generate_response(state, llm)

            elif action == "replan":
                # ⚠️ 二次检查：如果已执行步骤 >= 5，禁止 replan
                if len(past_steps) >= 5:
                    logger.warning(f"已执行 {len(past_steps)} 个步骤，禁止重新规划，强制生成响应")
                    trace_event(
                        "replanner_decision",
                        node="replanner",
                        data={
                            "action": "respond",
                            "requested_action": "replan",
                            "forced": True,
                            "reason": "replan_disabled_after_five_steps",
                        },
                    )
                    return await _generate_response(state, llm)

                # ⚠️ 强制限制新步骤数：
                # - 对照模式：旧策略，不超过当前剩余步骤数（会随执行越截越少）
                # - 正常模式：不超过剩余执行预算（MAX_STEPS - 已执行），防失控且不丢步骤
                step_cap = len(plan) if legacy else max(1, MAX_STEPS - len(past_steps))
                if len(new_steps) > step_cap:
                    logger.warning(
                        f"新步骤数 {len(new_steps)} 超出上限 {step_cap}，"
                        f"强制截断为 {step_cap} 个步骤"
                    )
                    new_steps = new_steps[:step_cap]

                logger.info(f"决定调整计划，新步骤数量: {len(new_steps)}")
                if new_steps:
                    # 替换剩余计划
                    trace_event(
                        "replanner_decision",
                        node="replanner",
                        data={
                            "action": "replan",
                            "forced": False,
                            "new_steps": new_steps,
                            "step_cap": step_cap,
                        },
                    )
                    return {"plan": new_steps}
                else:
                    logger.warning("replan 但未提供新步骤，继续执行原计划")
                    trace_event(
                        "replanner_decision",
                        node="replanner",
                        data={
                            "action": "continue",
                            "requested_action": "replan",
                            "forced": True,
                            "reason": "empty_replan",
                        },
                    )
                    return {}

            else:  # action == "continue"
                logger.info("决定继续执行当前计划")
                trace_event(
                    "replanner_decision",
                    node="replanner",
                    data={
                        "action": "continue",
                        "raw_action": action,
                        "forced": False,
                    },
                )
                return {}  # 不修改状态，继续执行

        except Exception as e:
            logger.error(f"重新规划失败: {e}, 继续执行剩余计划")
            trace_event(
                "replanner_decision",
                node="replanner",
                data={
                    "action": "continue",
                    "forced": True,
                    "reason": "replanner_error",
                    "error": str(e),
                },
            )
            return {}

    else:
        # 没有剩余计划，生成最终响应
        logger.info("计划已执行完毕，生成最终响应")
        trace_event(
            "replanner_decision",
            node="replanner",
            data={"action": "respond", "forced": True, "reason": "plan_exhausted"},
        )
        return await _generate_response(state, llm)


async def _generate_response(state: PlanExecuteState, llm: ChatQwen) -> dict[str, Any]:
    """生成最终响应"""
    logger.info("生成最终响应...")
    trace_event("model_call_started", node="replanner", data={"purpose": "generate_response"})

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])

    # 格式化执行历史
    execution_history = "\n\n".join(
        [f"### 步骤: {step}\n**结果:**\n{result}" for step, result in past_steps]
    )

    response_gen = response_prompt | llm.with_structured_output(Response)

    try:
        messages = [
            ("user", f"原始任务: {input_text}"),
            ("user", f"执行历史:\n{execution_history}"),
            ("user", "请基于以上信息生成全面的最终响应"),
        ]

        response_obj = await response_gen.ainvoke({"messages": messages})

        # 处理返回结果
        if response_obj is None:
            # structured output 偶发返回 None（LLM 抖动），走后备响应
            raise ValueError("LLM 返回空的最终响应")

        if isinstance(response_obj, Response):
            final_response = response_obj.response
        else:
            # 如果返回的是字典
            final_response = response_obj.get("response", "")  # type: ignore

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")
        trace_event(
            "model_call_completed",
            node="replanner",
            data={"purpose": "generate_response", "response_length": len(final_response)},
        )
        trace_event(
            "node_completed",
            node="replanner",
            data={"response_generated": True, "fallback": False},
        )

        return {"response": final_response}

    except Exception as e:
        logger.error(f"生成响应失败: {e}")
        # 生成简单的后备响应
        fallback_response = f"""# 任务执行结果

## 原始任务
{input_text}

## 执行的步骤
{_format_simple_steps(past_steps)}

## 说明
由于系统异常，无法生成完整响应。以上是已收集的信息。
"""
        trace_event(
            "model_call_failed",
            node="replanner",
            data={"purpose": "generate_response", "error": str(e)},
        )
        trace_event(
            "node_completed",
            node="replanner",
            data={"response_generated": True, "fallback": True},
        )
        return {"response": fallback_response}


def _format_simple_steps(past_steps: list) -> str:
    """格式化步骤列表（简单版）"""
    if not past_steps:
        return "无"

    formatted = []
    for i, (step, result) in enumerate(past_steps, 1):
        result_preview = result[:200] + "..." if len(result) > 200 else result
        formatted.append(f"{i}. **{step}**\n   {result_preview}\n")

    return "\n".join(formatted)
