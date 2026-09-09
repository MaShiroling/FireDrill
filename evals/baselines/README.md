# Evaluation baselines

该目录存放已完成评测的不可变基线清单。清单不复制大体积运行结果，而是记录：

- 数据集、结果文件及 SHA-256，用于识别输入和结果是否发生变化；
- Git revision、模型和关键运行参数；
- 指标定义、全量指标和分类指标；
- 主版本与对照版本共有运行的同口径 A/B 指标。

## 冻结当前基线

```bash
python evals/freeze_baseline.py --tag current --comparison-tag legacy
```

主结果必须覆盖 `cases_firewall.json` 的全部用例和指定轮次，默认是 30 条用例 × 3 轮。
对照结果可以只覆盖子集；A/B 指标仅根据两个结果文件共有的 `(case_id, run)` 计算。

当前历史 JSONL 没有逐次记录模型和运行参数，因此首份基线中的配置是从当前源码默认值
补录的，并在 `configuration.provenance` 中明确标记。后续轨迹采集需要在每次运行时直接
写入版本清单，避免依赖事后推断。

## 指标口径

- `task_success_rate`：全部确定性终态断言通过，并且运行无错误。
- `fake_completion_rate`：报告声称成功，但终态断言未通过。
- `run_error_rate`：运行记录中存在非空 `error`。
- `assertion_pass_rate`：通过的确定性断言占全部断言的比例。
- `average_steps`：每次运行产生的步骤完成事件数均值。
- `average_duration_s`：每次运行墙钟耗时均值。

简历或报告必须标明指标作用域。当前 `current.jsonl` 的 90 次全量运行和
`current`/`legacy` 共有的 48 次变更类运行不是同一个分母，禁止混用。
