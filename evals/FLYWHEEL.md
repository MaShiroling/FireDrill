# FireDrill 失败样本池与回放数据集

该阶段把评测结果转成数据飞轮可直接消费的资产：失败记录会按“用例 + 失败码 + 失败断言类型”生成稳定指纹，重复运行自动合并，并按风险和复现价值排序。

## 构建

```bash
.venv/bin/python evals/build_failure_pool.py \
  --results evals/results/current.jsonl \
  --results evals/results/legacy.jsonl
```

默认只收录有硬证据的模型或 Agent 行为失败，包括终态失败、假完成和反向误报。旧结果中仅由报告关键词推导、但终态断言通过的可疑负例不会自动入池，以避免标签污染。若要同时收录已经正确处理的非法参数、设备拒绝等困难正样本，增加 `--include-handled`。

输出目录默认为 `evals/artifacts/flywheel/`：

- `failure_pool.jsonl`：去重后的失败样本、优先级、出现次数、代表运行和关键轨迹；
- `replay_cases.json`：与防火墙 runner 兼容的回放用例；
- `manifest.json`：输入量、失败量、去重率以及失败码/类别/严重度统计。

`evals/artifacts/` 默认不提交 Git，避免轨迹中的运行环境信息进入仓库。

## 回放

启动主服务和防火墙 MCP 服务后执行：

```bash
NO_PROXY=localhost,127.0.0.1 \
.venv/bin/python evals/run_eval.py \
  --case-file evals/artifacts/flywheel/replay_cases.json \
  --runs 3 \
  --tag flywheel-replay
```

每个回放用例保留原任务、故障注入场景和终态硬断言，并附带 `flywheel.target_failure_codes`。runner 会把这段来源信息继续写入评测结果和 Trace，便于比较修复前后的失败是否消失。

## 回归门禁

回放完成后执行：

```bash
.venv/bin/python evals/check_replay_regression.py \
  --results evals/results/flywheel-replay.jsonl \
  --replay-cases evals/artifacts/flywheel/replay_cases.json \
  --name replanner-fix-v2
```

默认门限：每个已执行用例至少 3 轮、整体健康恢复率至少 66%、目标失败复现率不高于 34%、假完成和反向误报不高于 10%、运行异常不高于 5%。任一用例在 3 轮中没有一次健康恢复也会失败。

报告写入 `evals/artifacts/flywheel/regression/`：

- `regression_report.json`：CI 和后续统计使用；
- `regression_report.md`：面试展示或人工复盘使用。

门禁失败时命令返回退出码 1。可使用 `--require-all-cases` 强制完整覆盖 46 条回放用例，或使用 `--fail-on-new-codes` 将新出现的失败码也设为硬失败。默认只报告新失败码，避免 Trace 升级后出现更细标签导致误拦截。

## 提取长期记忆候选

失败池和回放报告生成后，可以先只读预览候选记忆：

```bash
.venv/bin/python evals/build_memory_candidates.py --dry-run
```

确认后写入 SQLite 事实源：

```bash
.venv/bin/python evals/build_memory_candidates.py \
  --tenant-id local \
  --device-type firewall
```

默认数据库为 `volumes/agent_memory.db`。失败池中只有能够映射为明确操作建议的样本会生成
`failure_lesson`；回放报告中只有状态为 `recovered` 的用例会生成 `successful_case`。
两类记录初始状态均为 `candidate`，不会未经审核直接进入 Agent 上下文。重复执行命令时，
相同 Namespace 和 Fingerprint 的记录会合并运行证据，不会重复插入。

候选写入后通过本地管理命令审核：

```bash
# 列出候选（默认同时显示失败教训和成功案例）
.venv/bin/python evals/review_memory_candidates.py list

# 查看正文、证据和完整审核历史
.venv/bin/python evals/review_memory_candidates.py show <memory_id>

# 批准或废弃；审核人和原因会写入不可变审计记录
.venv/bin/python evals/review_memory_candidates.py approve <memory_id> \
  --reviewer alice --reason "回放证据与操作建议一致"
.venv/bin/python evals/review_memory_candidates.py retire <memory_id> \
  --reviewer alice --reason "该经验不适用于当前设备版本"
```

审核状态只能按 `candidate → approved → retired` 前进，不能恢复已废弃记忆；重复提交相同
状态不会重复生成审核记录。管理命令没有删除能力，避免丢失证据链。

## 优先级

优先级为确定性规则分数，综合考虑：

- 假完成、反向误报、运行异常；
- 提交状态未知、重试耗尽、缺少验证；
- 失败断言数量和是否有可解释 Trace；
- 同一失败签名的重复出现次数。

分数只用于安排修复顺序，不参与任务成功率计算。

## 第一轮策略优化：MCP 安全重试

失败池中 `commit_flaky` 是明确的高价值失败模式。重试拦截器现在同时识别 MCP 协议错误、`success=false` 应用错误和传输异常，并按以下策略处理：

- `设备繁忙/请稍后重试/429/503` 等明确临时错误：指数退避重试；
- 非法参数、对象不存在、重复、设备拒绝等永久错误：直接返回；
- `commit_lose` 或变更操作的未知传输异常：不盲目重试，交由 Agent 查询真实状态；
- Trace 记录 `mcp_retry_scheduled`、`mcp_retry_skipped` 和 `mcp_call_exhausted`，用于下一轮失败归因。

本地真实 MCP 传输冒烟中，`commit_flaky(fail_times=2)` 的提交审计由原来的单次失败变为 `error → error → success`，最终运行版本从 R1 更新为 R2；`commit_lose`、提交拒绝和永久参数错误均验证为只调用一次。

### 定向回放结果（mcp-safe-retry-v1）

从失败池筛选 5 条 `commit_flaky` 样本，每条运行 3 轮，共完成 15 次完整 Agent 回放：

- 任务成功率：13/15（86.7%）；
- 终态断言通过率：32/36（88.9%）；
- 历史目标失败复现率：2/15（13.3%）；
- 假完成率：0/15；
- 回归门禁：PASS（3 条 recovered、2 条 unstable、0 条 not recovered）。

两次失败均被 Agent 如实报告：一次是提交步骤调用模型时发生 30 秒超时，另一次是模型混淆规则名与规则 ID，误判待删除规则不存在。它们分别进入下一轮“模型调用韧性”和“工具参数语义约束”优化候选，不归因于 MCP 重试策略。

可复现资产：

- `evals/baselines/firewall-flywheel-retry-v1.json`：本轮冻结基线及结果文件哈希；
- `evals/artifacts/flywheel/regression/regression_report.md`：逐用例门禁报告（运行产物，默认不提交）；
- `evals/results/flywheel-retry-v1.jsonl`：15 次原始结果，用于复核断言和校验基线哈希。

冻结定向子集时可使用：

```bash
.venv/bin/python evals/freeze_baseline.py \
  --tag flywheel-retry-v1 \
  --comparison-tag no-comparison \
  --cases evals/artifacts/flywheel/replay_cases.json \
  --case-ids REPLAY-FW-F05-5ceef15c,REPLAY-FW-F01-feba241e,REPLAY-FW-F01-7b511cad,REPLAY-FW-F02-16460419,REPLAY-FW-F05-7b1b2cba \
  --expected-runs 3 \
  --output evals/baselines/firewall-flywheel-retry-v1.json
```

## 第二轮策略优化：模型调用韧性与参数语义约束

第一轮留下的两条不稳定样本分别对应 Executor 模型调用瞬时超时，以及规则名称与规则 ID 混淆。本轮增加以下能力：

- 仅对 Executor 的模型调用做有限重试；只有超时、连接中断、限流和明确的临时服务错误才重试，永久错误直接返回；
- 模型重试与工具执行边界隔离，避免因重试模型而重复执行有副作用的防火墙工具；
- Executor 每一步都携带不可变的原始任务和最近实际执行结果，让后续步骤复用工具返回的真实规则 ID；
- Planner 与 Executor 明确区分 `rule name` 和 `rule_id`，禁止猜测 `new-rule-001` 等虚构 ID；
- Trace、失败分类和失败池新增模型调用失败、重试调度、重试耗尽事件。

### 定向回放结果（flywheel-resilience-v2）

只回放第一轮仍不稳定的 2 条样本，每条运行 3 轮，共 6 次完整 Agent 执行：

- 任务成功率：4/6（66.7%）→ 6/6（100%）；
- 终态断言通过率：8/12（66.7%）→ 12/12（100%）；
- 历史目标失败复现率：2/6（33.3%）→ 0/6；
- 假完成率：0/6 → 0/6；
- 平均步骤数：4.83 → 4.00；
- 回归门禁：PASS，2 条样本均为 recovered。

本轮真实回放没有自然触发模型超时，因此“瞬时异常后成功恢复”由注入 `TimeoutError` 的单元测试覆盖；真实回放证明了新增边界不会破坏完整 Agent 链路。Milvus Lite 因本地数据库锁降级不可用，但这两个防火墙用例不依赖 RAG，评分仍由带外终态快照和硬断言完成。

可复现资产：

- `evals/baselines/firewall-flywheel-resilience-v2.json`：6 次回放的冻结基线与结果哈希；
- `evals/results/flywheel-resilience-v2.jsonl`：原始运行结果；
- `evals/artifacts/flywheel/regression/regression_report.md`：回归门禁报告（运行产物，默认不提交）。
