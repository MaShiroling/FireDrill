"""
MCP 客户端管理
提供全局单例的 MCP 客户端，避免重复初始化
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from loguru import logger
from mcp.types import CallToolResult, TextContent

from app.config import config
from app.observability import trace_event

# 全局 MCP 客户端（延迟初始化）
_mcp_client: MultiServerMCPClient | None = None

_MUTATING_TOOLS = {
    "add_firewall_rule",
    "update_firewall_rule",
    "delete_firewall_rule",
    "move_firewall_rule",
    "commit_config",
    "discard_candidate",
}
_AMBIGUOUS_MARKERS = (
    "状态未知",
    "未返回确认",
    "响应丢失",
    "unknown state",
    "unknown status",
)
_PERMANENT_MARKERS = (
    "非法",
    "无效",
    "不存在",
    "重复",
    "无改动",
    "无需",
    "拒绝",
    "权限",
    "禁止",
    "invalid",
    "not found",
    "duplicate",
    "permission",
    "denied",
)
_EXPLICIT_SAFE_RETRY_MARKERS = (
    "请稍后重试",
    "设备繁忙",
    "暂时不可用",
    "temporarily unavailable",
    "rate limit",
    "too many requests",
    "429",
    "503",
)
_TRANSIENT_MARKERS = (
    "临时",
    "暂时",
    "超时",
    "timeout",
    "timed out",
    "temporar",
    "unavailable",
    "connection",
    "连接",
)


@dataclass(frozen=True)
class MCPFailure:
    message: str
    kind: str
    retryable: bool
    source: str


def _content_texts(result: Any) -> list[str]:
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []

    texts: list[str] = []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            value = item.get("text")
            if isinstance(value, str):
                texts.append(value)
        else:
            value = getattr(item, "text", None)
            if isinstance(value, str):
                texts.append(value)
    return texts


def _semantic_error(result: Any) -> str | None:
    payloads: list[Any] = [result] if isinstance(result, dict) else []
    structured_content = getattr(result, "structuredContent", None)
    if structured_content is not None:
        payloads.append(structured_content)
    for text in _content_texts(result):
        try:
            payloads.append(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            continue
    for payload in payloads:
        if isinstance(payload, dict) and payload.get("success") is False:
            return str(payload.get("error") or payload.get("message") or "工具返回 success=false")
    return None


def _classify_failure(
    tool_name: str,
    message: str,
    *,
    source: str,
    transport_exception: bool = False,
) -> MCPFailure:
    lowered = message.lower()
    if any(marker in lowered for marker in _AMBIGUOUS_MARKERS):
        return MCPFailure(message=message, kind="ambiguous", retryable=False, source=source)
    if any(marker in lowered for marker in _PERMANENT_MARKERS):
        return MCPFailure(message=message, kind="permanent", retryable=False, source=source)
    if any(marker in lowered for marker in _EXPLICIT_SAFE_RETRY_MARKERS):
        return MCPFailure(message=message, kind="transient", retryable=True, source=source)
    if transport_exception and tool_name in _MUTATING_TOOLS:
        return MCPFailure(message=message, kind="ambiguous", retryable=False, source=source)
    if transport_exception or any(marker in lowered for marker in _TRANSIENT_MARKERS):
        return MCPFailure(message=message, kind="transient", retryable=True, source=source)
    return MCPFailure(message=message, kind="unknown", retryable=False, source=source)


def _result_failure(tool_name: str, result: Any) -> MCPFailure | None:
    semantic_error = _semantic_error(result)
    if semantic_error:
        return _classify_failure(tool_name, semantic_error, source="application_result")

    protocol_error = bool(getattr(result, "isError", False))
    message_status = str(getattr(result, "status", "")).lower()
    if protocol_error or message_status == "error":
        texts = _content_texts(result)
        message = "\n".join(texts).strip() or "MCP 工具返回错误结果"
        return _classify_failure(tool_name, message, source="protocol_result")
    return None


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], isError=True)


def format_exception_chain(exc: BaseException) -> str:
    """展开 ExceptionGroup / TaskGroup，便于日志定位真实子异常。"""
    sub_exceptions = getattr(exc, "exceptions", None)
    if sub_exceptions is not None:
        lines = [str(exc)]
        for i, sub in enumerate(sub_exceptions):
            lines.append(f"  [{i}] {format_exception_chain(sub)}")
        return "\n".join(lines)
    msg = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return f"{msg}\n  caused by: {format_exception_chain(cause)}"
    return msg


async def load_mcp_tools_safe(
    client: MultiServerMCPClient,
) -> tuple[list[BaseTool | Any], str | None]:
    """加载 MCP 工具；失败时返回空列表与可读错误信息，不向上抛出。"""
    try:
        tools = await client.get_tools()
        return tools, None
    except BaseException as e:
        return [], format_exception_chain(e)


async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    """MCP 工具调用重试拦截器。

    同时识别协议错误（isError/status）、应用错误（JSON success=false）和异常。
    只重试明确的临时故障；永久错误直接返回，变更工具的未知传输结果按
    “状态可能不确定”处理，避免盲目重复提交或重复新增。

    MCPToolCallRequest 结构：
    - name: str - 工具名称
    - args: dict[str, Any] - 工具参数
    - server_name: str - 服务器名称

    Args:
        request: MCP 工具调用请求
        handler: 实际的工具调用处理器
        max_retries: 最大重试次数（默认3次）
        delay: 初始延迟时间（秒，默认1秒）

    Returns:
        CallToolResult: 工具调用结果或错误信息
    """
    max_attempts = max(1, max_retries)
    last_failure: MCPFailure | None = None
    last_result: Any = None

    for attempt in range(max_attempts):
        attempt_started = time.perf_counter()
        trace_event(
            "mcp_attempt_started",
            node="executor",
            data={
                "server": request.server_name,
                "tool": request.name,
                "args": request.args,
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
            },
        )
        try:
            logger.info(
                f"调用 MCP 工具: {request.name} "
                f"(服务器: {request.server_name}, 第 {attempt + 1}/{max_attempts} 次尝试)"
            )
            result = await handler(request)
            failure = _result_failure(request.name, result)
            if failure is None:
                logger.info(f"MCP 工具 {request.name} 调用成功")
                trace_event(
                    "mcp_attempt_completed",
                    node="executor",
                    data={
                        "server": request.server_name,
                        "tool": request.name,
                        "attempt": attempt + 1,
                        "success": True,
                        "returned_error_result": False,
                        "duration_s": round(time.perf_counter() - attempt_started, 3),
                    },
                )
                return result

            last_failure = failure
            last_result = result
            will_retry = failure.retryable and attempt < max_attempts - 1
            logger.warning(
                f"MCP 工具 {request.name} 返回{failure.kind}错误 "
                f"(第 {attempt + 1}/{max_attempts} 次): {failure.message}"
            )
            trace_event(
                "mcp_attempt_completed",
                node="executor",
                data={
                    "server": request.server_name,
                    "tool": request.name,
                    "attempt": attempt + 1,
                    "success": False,
                    "returned_error_result": True,
                    "error": failure.message,
                    "error_kind": failure.kind,
                    "error_source": failure.source,
                    "retryable": failure.retryable,
                    "will_retry": will_retry,
                    "duration_s": round(time.perf_counter() - attempt_started, 3),
                },
            )
            if not failure.retryable:
                trace_event(
                    "mcp_retry_skipped",
                    node="executor",
                    data={
                        "server": request.server_name,
                        "tool": request.name,
                        "attempt": attempt + 1,
                        "reason": failure.kind,
                        "error": failure.message,
                    },
                )
                return result

        except Exception as e:
            failure = _classify_failure(
                request.name,
                str(e),
                source="exception",
                transport_exception=True,
            )
            last_failure = failure
            will_retry = failure.retryable and attempt < max_attempts - 1
            logger.warning(
                f"MCP 工具 {request.name} 调用失败 "
                f"(第 {attempt + 1}/{max_attempts} 次): {str(e)}"
            )
            trace_event(
                "mcp_attempt_completed",
                node="executor",
                data={
                    "server": request.server_name,
                    "tool": request.name,
                    "attempt": attempt + 1,
                    "success": False,
                    "duration_s": round(time.perf_counter() - attempt_started, 3),
                    "error": str(e),
                    "error_kind": failure.kind,
                    "error_source": failure.source,
                    "retryable": failure.retryable,
                    "will_retry": will_retry,
                },
            )
            if not failure.retryable:
                trace_event(
                    "mcp_retry_skipped",
                    node="executor",
                    data={
                        "server": request.server_name,
                        "tool": request.name,
                        "attempt": attempt + 1,
                        "reason": failure.kind,
                        "error": failure.message,
                    },
                )
                return _error_result(f"工具 {request.name} 调用结果不确定: {failure.message}")

        if attempt < max_attempts - 1:
            wait_time = delay * (2**attempt)
            logger.info(f"等待 {wait_time:.1f} 秒后重试...")
            trace_event(
                "mcp_retry_scheduled",
                node="executor",
                data={
                    "server": request.server_name,
                    "tool": request.name,
                    "next_attempt": attempt + 2,
                    "wait_s": wait_time,
                    "error_kind": last_failure.kind if last_failure else "unknown",
                },
            )
            await asyncio.sleep(wait_time)

    failure_message = last_failure.message if last_failure else "未知错误"
    error_msg = f"工具 {request.name} 在 {max_attempts} 次尝试后仍然失败: {failure_message}"
    logger.error(error_msg)
    trace_event(
        "mcp_call_exhausted",
        node="executor",
        data={
            "server": request.server_name,
            "tool": request.name,
            "attempts": max_attempts,
            "error": failure_message,
            "error_kind": last_failure.kind if last_failure else "unknown",
            "returned_error_result": last_result is not None,
        },
    )
    return last_result if last_result is not None else _error_result(error_msg)


# 使用配置文件中定义的完整 MCP 服务器配置
DEFAULT_MCP_SERVERS = config.mcp_servers


async def get_mcp_client(
    servers: dict[str, dict[str, str]] | None = None,
    tool_interceptors: list | None = None,
    force_new: bool = False,
) -> MultiServerMCPClient:
    """
    获取或初始化 MCP 客户端（不带重试拦截器）

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）

    从 langchain-mcp-adapters 0.1.0 开始，MultiServerMCPClient 不再支持作为上下文管理器使用。
    直接创建实例即可使用。

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: MCP 客户端实例
    """
    global _mcp_client

    # 如果请求新实例，直接创建并返回（不缓存）
    if force_new:
        logger.info("创建新的 MCP 客户端实例（非单例）")
        client = _create_mcp_client(servers or DEFAULT_MCP_SERVERS, tool_interceptors)
        # 不再需要 __aenter__()，直接返回即可
        return client

    # 单例模式：如果已存在，直接返回
    if _mcp_client is None:
        logger.info("初始化全局 MCP 客户端...")
        _mcp_client = _create_mcp_client(servers or DEFAULT_MCP_SERVERS, tool_interceptors)
        # 不再需要 __aenter__()，直接使用即可
        logger.info("全局 MCP 客户端初始化完成")

    return _mcp_client


async def get_mcp_client_with_retry(
    servers: dict[str, dict[str, str]] | None = None,
    tool_interceptors: list | None = None,
    force_new: bool = False,
) -> MultiServerMCPClient:
    """
    获取或初始化带重试功能的 MCP 客户端

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）
    重试拦截器会自动添加到拦截器列表的开头

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表（会在重试拦截器之后添加）
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: 带重试功能的 MCP 客户端实例
    """
    # 构建拦截器列表：重试拦截器在最前面
    interceptors = [retry_interceptor]
    if tool_interceptors:
        interceptors.extend(tool_interceptors)

    return await get_mcp_client(
        servers=servers, tool_interceptors=interceptors, force_new=force_new
    )


def _create_mcp_client(
    servers: dict[str, dict[str, str]], tool_interceptors: list | None = None
) -> MultiServerMCPClient:
    """
    创建 MCP 客户端实例

    Args:
        servers: MCP 服务器配置
        tool_interceptors: 工具拦截器列表

    Returns:
        MultiServerMCPClient: 未初始化的客户端实例
    """
    # MultiServerMCPClient 的第一个参数直接接收 servers 配置字典
    # 格式: {server_name: {"transport": "...", "url": "..."}}
    kwargs: dict[str, Any] = {}

    if tool_interceptors:
        kwargs["tool_interceptors"] = tool_interceptors

    # 第一个参数是 servers 配置，直接传递
    return MultiServerMCPClient(servers, **kwargs)  # type: ignore[arg-type]


def suggest_mcp_transport(url: str, transport: str) -> str | None:
    """URL 与 transport 明显不匹配时给出建议（不自动改写配置）。"""
    lower_url = url.lower()
    if "/sse" in lower_url and transport.replace("_", "-") in (
        "streamable-http",
        "http",
    ):
        return (
            f"MCP URL 含 /sse/ 但 transport={transport!r}，" "腾讯云等托管端点应使用 transport=sse"
        )
    if transport == "sse" and "/mcp" in lower_url and "/sse" not in lower_url:
        return (
            f"MCP URL 为本地 FastMCP 路径但 transport={transport!r}，"
            "本地服务通常应使用 transport=streamable-http"
        )
    return None
