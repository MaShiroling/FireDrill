"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现
"""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

from app.agent.aiops import PlanExecuteState, executor, planner, replanner
from app.config import config
from app.observability import begin_run, finish_run, trace_event

# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsService:
    """通用 Plan-Execute-Replan 服务"""

    def __init__(self):
        """初始化服务"""
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("Plan-Execute-Replan Service 初始化完成")

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        # 创建状态图
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, planner)  # 制定计划
        workflow.add_node(NODE_EXECUTOR, executor)  # 执行步骤
        workflow.add_node(NODE_REPLANNER, replanner)  # 重新规划

        # 设置入口点
        workflow.set_entry_point(NODE_PLANNER)

        # 定义边
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)  # planner -> executor
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)  # executor -> replanner

        # replanner 的条件边
        def should_continue(state: PlanExecuteState) -> str:
            """判断是否继续执行"""
            # 如果已经生成了最终响应，结束
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            # 如果还有计划步骤，继续执行
            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR

            # 计划为空但没有响应，返回 replanner 生成响应
            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER, should_continue, {NODE_EXECUTOR: NODE_EXECUTOR, END: END}
        )

        # 编译工作流
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("工作流图构建完成")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
        trace_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行 Plan-Execute-Replan 流程

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        project_root = Path(__file__).resolve().parents[2]
        trace_dir = Path(config.agent_trace_dir)
        if not trace_dir.is_absolute():
            trace_dir = project_root / trace_dir
        recorder, trace_token = begin_run(
            enabled=config.agent_trace_enabled,
            root=project_root,
            trace_dir=trace_dir,
            session_id=session_id,
            task=user_input,
            runtime={
                "app_name": config.app_name,
                "app_version": config.app_version,
                "model": config.rag_model,
                "replanner_legacy": config.replanner_legacy,
                "max_agent_steps": 8,
                "model_max_attempts": config.agent_model_max_attempts,
                "model_retry_delay_s": config.agent_model_retry_delay_s,
                "mcp_servers": config.mcp_servers,
            },
            metadata=trace_metadata,
            max_value_chars=config.agent_trace_max_value_chars,
        )
        trace_finished = False
        final_values: dict[str, Any] = {}
        config_dict = {"configurable": {"thread_id": session_id}}

        try:
            if recorder is not None:
                yield {
                    "type": "trace_started",
                    "stage": "trace",
                    "message": "执行轨迹已创建",
                    "trace_id": recorder.run_id,
                    "trace_path": recorder.relative_path(),
                }

            # 每次执行前清空该会话的旧检查点：
            # past_steps 走 operator.add 追加式 reducer，initial_state 传 [] 并不会清空历史，
            # 复用同一 session_id 时上一轮执行的 past_steps 会残留，
            # 污染 replanner 的步数统计（>=5 禁 replan、>=8 强制出报告）
            await self.checkpointer.adelete_thread(session_id)

            # 初始化状态
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": "",
            }

            # 流式执行工作流
            async for event in self.graph.astream(
                input=initial_state, config=config_dict, stream_mode="updates"
            ):
                # 解析事件
                for node_name in event:
                    logger.info(f"节点 '{node_name}' 输出事件")

                    # updates 模式吐出的是节点返回的状态增量（past_steps 仅含刚追加的 1 条），
                    # 进度统计需要全量，这里取合并后的当前状态传给事件格式化器
                    current_state = await self.graph.aget_state(config_dict)
                    state_values = current_state.values if current_state else {}
                    trace_event(
                        "graph_state_updated",
                        node=node_name,
                        data={
                            "remaining_steps": len(state_values.get("plan", [])),
                            "completed_steps": len(state_values.get("past_steps", [])),
                            "has_response": bool(state_values.get("response")),
                        },
                    )

                    # 根据节点类型生成不同的事件
                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(state_values)

                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(state_values)

                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(state_values)

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""

            # 安全地获取响应（处理 values 可能为 None 的情况）
            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")
                final_values = dict(final_state.values)

            finish_run(
                recorder,
                trace_token,
                status="completed",
                final_state=final_values,
            )
            trace_finished = True

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response,
                "trace_id": recorder.run_id if recorder else None,
                "trace_path": recorder.relative_path() if recorder else None,
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            try:
                error_state = self.graph.get_state(config_dict)
                if error_state and error_state.values:
                    final_values = dict(error_state.values)
            except Exception:
                # 获取失败现场本身不能覆盖原始异常。
                pass
            finish_run(
                recorder,
                trace_token,
                status="error",
                final_state=final_values,
                error=str(e),
            )
            trace_finished = True
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}",
                "trace_id": recorder.run_id if recorder else None,
                "trace_path": recorder.relative_path() if recorder else None,
            }
        finally:
            if not trace_finished:
                finish_run(
                    recorder,
                    trace_token,
                    status="cancelled",
                    final_state=final_values,
                    error="Agent 流在完成前被取消或关闭",
                )

    async def diagnose(self, session_id: str = "default") -> AsyncGenerator[dict[str, Any], None]:
        """
        AIOps 诊断接口（兼容旧接口）

        Args:
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 诊断过程的流式事件
        """
        # 使用固定的 AIOps 任务描述
        from textwrap import dedent

        aiops_task = dedent(
            """诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过"""
        )

        async for event in self.execute(aiops_task, session_id):
            # 转换事件格式以兼容旧的 API
            if event.get("type") == "complete":
                # 将 response 包装为 diagnosis 格式
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {"status": "completed", "report": event.get("response", "")},
                    "trace_id": event.get("trace_id"),
                    "trace_path": event.get("trace_path"),
                }
            else:
                yield event

    def _format_planner_event(self, state: dict | None) -> dict:
        """格式化 Planner 节点事件"""
        if not state:
            return {"type": "status", "stage": "planner", "message": "规划节点执行中"}

        plan = state.get("plan", [])

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan,
        }

    def _format_executor_event(self, state: dict | None) -> dict:
        """格式化 Executor 节点事件"""
        if not state:
            return {"type": "status", "stage": "executor", "message": "执行节点运行中"}

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan),
            }
        else:
            return {"type": "status", "stage": "executor", "message": "开始执行步骤"}

    def _format_replanner_event(self, state: dict | None) -> dict:
        """格式化 Replanner 节点事件"""
        if not state:
            return {"type": "status", "stage": "replanner", "message": "评估节点运行中"}

        response = state.get("response", "")
        plan = state.get("plan", [])

        if response:
            # 已生成最终响应
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response,
            }
        else:
            # 重新规划
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
                "remaining_steps": len(plan),
            }


# 全局单例
aiops_service = AIOpsService()
