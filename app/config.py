"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "FireDrill"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒
    # Milvus Lite（嵌入式，免 Docker）：设置为本地的 .db 文件路径即启用，
    # 如 "./volumes/milvus_lite.db"；留空则使用 host:port 连接 Milvus 服务
    # 注意：不能用 MILVUS_URI 作为环境变量名（pymilvus 导入期会解析并校验它）
    milvus_lite_path: str = ""

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 评测对照开关：True 时 replanner 使用修复前的旧提示词与旧截断策略，
    # 用于复现"变更任务提前收尾（假完成）"行为做 A/B 实验
    replanner_legacy: bool = False

    # Agent 结构化执行轨迹（用于离线评测与数据飞轮）
    agent_trace_enabled: bool = True
    agent_trace_dir: str = "evals/artifacts/runs"
    agent_trace_max_value_chars: int = 8000
    agent_model_max_attempts: int = 2
    agent_model_retry_delay_s: float = 0.5

    # Agent 长期记忆事实源；向量索引后续单独接入，可由这里的数据重建
    agent_memory_db_path: str = "volumes/agent_memory.db"

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置（transport: stdio | sse | streamable-http）
    # 腾讯云托管 MCP 的 URL 通常含 /sse/，需使用 sse；本地 FastMCP 使用 streamable-http
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"
    mcp_firewall_transport: str = "streamable-http"
    mcp_firewall_url: str = "http://localhost:8005/mcp"

    # Prometheus
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    @property
    def milvus_lite_mode(self) -> bool:
        """是否使用 Milvus Lite（嵌入式本地文件模式）"""
        return bool(self.milvus_lite_path)

    @property
    def mcp_servers(self) -> dict[str, dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
            "firewall": {
                "transport": self.mcp_firewall_transport,
                "url": self.mcp_firewall_url,
            },
        }


# 全局配置实例
config = Settings()
