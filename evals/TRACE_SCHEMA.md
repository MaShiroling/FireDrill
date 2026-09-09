# Agent execution trace

每次 AIOps/评测任务会在 `AGENT_TRACE_DIR` 下生成一个独立 JSON 文件。默认路径为：

```text
evals/artifacts/runs/YYYY-MM-DD/<run_id>.json
```

该目录默认不提交到 Git。评测结果 JSONL 通过 `trace_id` 和 `trace_path` 指向对应轨迹。

## 顶层字段

- `schema_version`：轨迹结构版本。
- `run_id`、`session_id`、`task`：运行身份和原始任务。
- `started_at`、`ended_at`、`duration_s`、`status`：生命周期信息。
- `version_manifest`：Git revision、工作区状态、模型和 MCP 配置。
- `metadata`：评测时包含 case、category、run、tag、scenario 等信息。
- `events`：按 `sequence` 排序的完整执行事件。
- `final_state`：LangGraph 最终状态。
- `metrics`：事件数、工具调用数和 MCP 尝试次数等运行统计。

## 关键事件

- `node_started`、`node_completed`、`node_failed`：节点生命周期。
- `model_call_started`、`model_call_completed`、`model_call_failed`：模型调用尝试及结果。
- `model_retry_scheduled`、`model_retry_skipped`、`model_retry_exhausted`：模型调用的安全重试决策。
- `knowledge_retrieval_completed`：Planner 的经验知识检索结果。
- `tool_inventory_loaded`：节点可见的本地及 MCP 工具版本。
- `tool_call_requested`、`tool_call_completed`：工具参数和结果。
- `mcp_attempt_started`、`mcp_attempt_completed`：MCP 尝试及错误类型。
- `mcp_retry_scheduled`、`mcp_retry_skipped`、`mcp_call_exhausted`：重试调度、跳过原因和耗尽终态。
- `replanner_decision`：继续、重规划、响应及强制决策原因。
- `graph_state_updated`：每个节点完成后的计划与执行进度。
- `run_finished`：运行终态。

执行接口会先发送一个 `trace_started` SSE 事件，使调用方即使在任务超时或中途取消时，
也能保留 `trace_id` 和 `trace_path` 并定位未完成轨迹。

## 数据保护

API Key、Authorization、密码和访问令牌等字段会递归脱敏。字符串默认最多保留 8000
字符，避免日志和文档内容导致轨迹无限膨胀。可通过以下配置调整：

```text
AGENT_TRACE_ENABLED=true
AGENT_TRACE_DIR=evals/artifacts/runs
AGENT_TRACE_MAX_VALUE_CHARS=8000
```

轨迹模块采用失败隔离：创建或写入轨迹失败时只记录警告，不改变 Agent 执行结果。
