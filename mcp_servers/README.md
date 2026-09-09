# MCP Servers

为 AIOps 智能诊断提供日志查询、监控数据工具，以及用于配置变更全链路演练的假防火墙。

## 📚 服务列表

### CLS Server (`cls_server.py`)
**日志查询服务** - 端口 8003

**核心工具：**
- `get_current_timestamp` - 获取当前毫秒时间戳（供 search_log 计算时间范围）
- `get_region_code_by_name` - 按中文地区名查地区代码
- `get_topic_info_by_name` - 按主题名精确查询日志主题
- `search_topic_by_service_name` - 按服务名搜索日志主题（"服务名 → topic_id"的主入口）
- `search_log` - 按 topic_id + 毫秒时间范围搜索日志（核心查询工具）

### Monitor Server (`monitor_server.py`)
**监控数据服务** - 端口 8004

**核心工具：**
- `query_cpu_metrics` - CPU 使用率时间序列查询（含 80% 阈值告警判断）
- `query_memory_metrics` - 内存使用率时间序列查询（含 70% 阈值告警判断）

### Firewall Server (`firewall_server.py`)
**有状态假防火墙** - 端口 8005

模拟「读配置 → 下发 → 验证 → 出错处理」全链路，采用两阶段下发语义：
所有写操作落在候选配置（candidate）上，`commit_config` 生效，`discard_candidate` 回滚。

服务默认使用 SQLite 持久化到 `volumes/firewall_state.db`。Running 配置按 Revision
留存，Candidate 以 ChangeSet 管理，Commit/Discard 和审计日志写入同一事务；进程重启后
可恢复未提交改动。设置 `FIREWALL_STATE_BACKEND=memory` 可切回纯内存模式。

**核心工具：**
- 读配置: `get_firewall_overview` / `list_security_zones` / `list_firewall_rules` / `get_firewall_rule`
- 下发: `add_firewall_rule` / `update_firewall_rule` / `delete_firewall_rule` / `move_firewall_rule` / `commit_config` / `discard_candidate`
- 验证: `get_config_diff` / `test_traffic`（模拟报文首条命中匹配）/ `get_rule_hit_count`

**评测管理通道**（HTTP 路由，不暴露给 Agent）：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/admin/reset` | 恢复出厂状态（body 可选 `{"keep_fault": true}`） |
| POST | `/admin/scenario` | 注入故障：`{"fault": "none\|commit_reject\|commit_flaky\|commit_lose", "fail_times": N}` |
| GET | `/admin/snapshot` | 导出完整状态（规则、revision、命中计数、操作审计日志），供评测打分 |
| GET | `/admin/health` | 健康检查 |

故障注入说明：
- `commit_reject` - commit 永远被设备拒绝
- `commit_flaky` - 前 N 次 commit 报「设备繁忙」，之后恢复（测重试）
- `commit_lose` - commit 实际生效但返回超时失败（测 Agent 核实状态的能力）

## 🚀 快速开始

### 安装依赖
```bash
pip install -e .
# 或：.tools/uv sync
```

### 启动服务

**方式一：使用 Makefile（推荐）**
```bash
make mcp-start   # 启动所有 MCP 服务
make mcp-stop    # 停止所有 MCP 服务
make mcp-status  # 查看服务状态
```

**方式二：手动启动**
```bash
python mcp_servers/cls_server.py
python mcp_servers/monitor_server.py
python mcp_servers/firewall_server.py
```

## 💡 使用示例

### AIOps 诊断场景

```
用户: data-sync-service 出现告警，请排查

Agent 自动执行（计划由 Planner 动态生成，以下为典型路径）:
1. query_prometheus_alerts()（本地工具）→ 获取当前活动告警
2. query_cpu_metrics("data-sync-service") → CPU 趋势分析
3. query_memory_metrics("data-sync-service") → 内存趋势分析
4. search_topic_by_service_name("data-sync-service") → 定位日志主题
5. search_log(topic_id, start_time, end_time) → 查询日志
6. 综合分析 → 生成诊断报告和修复建议
```

### 工具参数示例

**查询 CPU 指标：**
```python
query_cpu_metrics(
    service_name="data-sync-service",
    start_time="2024-02-14 02:00:00",
    interval="1m"
)
```

**搜索日志：**
```python
# 先用 get_current_timestamp() 拿当前毫秒时间戳，再计算时间范围
search_log(
    topic_id="topic-001",
    start_time=1708011445000,   # 15 分钟前（毫秒时间戳）
    end_time=1708012345000,     # 当前时间
    limit=100
)
```

**防火墙变更全链路（读配置 → 下发 → 验证）：**
```python
list_firewall_rules(source="running")                       # 1. 读现有配置
add_firewall_rule(                                          # 2. 下发到候选配置
    name="allow-office-to-gitlab",
    src_zone="trust", dst_zone="dmz",
    src_addr="10.1.8.0/24", dst_addr="172.16.1.20/32",
    protocol="tcp", dst_port="22", action="allow",
    description="运维通道临时放行"
)
get_config_diff()                                           # 3. 确认改动
commit_config(                                              # 4. 提交生效
    idempotency_key="change-20260908-001",                  #    相同键重试不会重复提交
    expected_revision=1,                                    #    乐观锁，防止覆盖新版本
)
test_traffic(src_zone="trust", dst_zone="dmz",              # 5. 模拟报文验证
             src_addr="10.1.8.5", dst_addr="172.16.1.20",
             protocol="tcp", dst_port="22")
```

评测脚本可通过 `POST http://127.0.0.1:8005/admin/reset` 重置状态、
`GET /admin/snapshot` 拉取最终配置与操作审计日志打分。

可用 `FIREWALL_DB_PATH` 修改 SQLite 文件位置。`/admin/reset` 会清空历史 Revision、
ChangeSet、幂等记录和审计日志并恢复出厂配置，适合每条评测用例开始前调用。

## 🔧 高级配置

### 接入真实 API

当前返回模拟数据。接入真实 API 步骤：

**腾讯云 CLS：**
```bash
# 安装 SDK
pip install tencentcloud-sdk-python

# 配置环境变量
export TENCENTCLOUD_SECRET_ID="your-id"
export TENCENTCLOUD_SECRET_KEY="your-key"

# 在 cls_server.py 中集成
from tencentcloud.cls.v20201016 import cls_client
```

**其他监控系统：**
- Prometheus
- Grafana
- 云监控（腾讯云/阿里云/AWS）
- 自建监控平台

### 自定义 Mock 数据

修改各 Server 文件中的数据生成逻辑，模拟实际场景。

## 📚 参考资料

- [FastMCP 文档](https://github.com/jlowin/fastmcp)
- [MCP 协议](https://modelcontextprotocol.io/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [主项目 README](../README.md)

---

**注意**: 当前版本返回模拟数据，生产环境需配置真实 API。
