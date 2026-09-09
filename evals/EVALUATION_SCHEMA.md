# FireDrill 确定性评测与失败分类

`evals/run_eval.py` 不使用 LLM 打分。每次运行结束后，评测器读取三类证据：

1. `/admin/snapshot` 返回的运行配置、候选配置、revision、命中计数和审计日志；
2. Agent 最终报告；
3. `evals/artifacts/runs/` 下的结构化 Trace（缺失时仍可完成终态评分）。

结果 JSONL 保留原有 `passed`、`fake_complete`、`asserts` 等字段，并新增：

- `failure_codes`：稳定的机器可读失败标签；
- `failure_evidence`：每个标签对应的审计、Trace 或断言证据；
- `evaluation`：完整的 `1.0` 版本化评测结果。

## 判定口径

- `passed`：全部带外终态断言通过，并且 Agent 运行本身没有异常；
- `fake_complete`：报告声称成功，但 `passed=false`；
- `false_failure`：预期成功且终态通过，但报告声称执行失败；
- `correct_failure`：预期失败用例中，报告没有声称成功；
- Trace 只用于解释失败，不参与核心终态通过判定，因此 Trace 缺失不会改变成功率。

## 失败码

| 失败码 | 含义 |
|---|---|
| `run_error` | Agent 执行超时或异常 |
| `evaluator_error` | 评测器的流量验证通道异常 |
| `assertion_failed` | 至少一个带外断言失败 |
| `planning_failure` | Planner 失败并进入兜底计划 |
| `tool_selection_failure` | 执行步骤没有选择工具 |
| `tool_execution_failure` | 工具或 MCP 返回错误结果 |
| `retry_exhausted` | 工具重试次数耗尽 |
| `invalid_argument` | 参数、对象或操作前置条件非法 |
| `commit_rejected` | 设备拒绝提交 |
| `commit_state_unknown` | 提交状态未知且未核实 |
| `verification_missing` | 流量、命中或提交后核实未通过 |
| `pending_changes` | 结束时仍有候选配置未提交 |
| `step_budget_exhausted` | 达到 8 步执行预算仍未完成 |
| `false_completion` | 报告声称成功但终态失败 |
| `false_failure` | 终态成功但报告声称执行失败 |
| `report_inconsistent` | 报告内容与带外证据不一致 |

同一次运行可以命中多个标签。失败码由确定性规则产生，适合作为后续失败样本聚类、回放集构建和回归看板的主键。
