"""Safe retry wrapper for side-effect-free model API calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from loguru import logger

from app.observability import trace_event

T = TypeVar("T")

_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "temporary failure",
    "connection reset",
    "connection aborted",
    "connection closed",
    "server disconnected",
    "rate limit",
    "too many requests",
    "429",
    "502",
    "503",
    "504",
    "超时",
    "暂时不可用",
    "连接重置",
    "限流",
)


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    children = getattr(exc, "exceptions", None)
    if not children:
        return [exc]
    leaves: list[BaseException] = []
    for child in children:
        leaves.extend(_leaf_exceptions(child))
    return leaves


def is_transient_model_error(exc: BaseException) -> bool:
    """Return true only when every leaf error has an explicit transient signal."""
    leaves = _leaf_exceptions(exc)
    return bool(leaves) and all(
        isinstance(leaf, (TimeoutError, ConnectionError))
        or any(marker in f"{type(leaf).__name__}: {leaf}".lower() for marker in _TRANSIENT_MARKERS)
        for leaf in leaves
    )


def _error_summary(exc: BaseException) -> str:
    leaves = _leaf_exceptions(exc)
    details = [f"{type(leaf).__name__}: {leaf}" for leaf in leaves]
    return " | ".join(details)[:500]


async def invoke_model_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    node: str,
    purpose: str,
    max_attempts: int = 2,
    delay_s: float = 0.5,
) -> T:
    """Retry a model API call without replaying any surrounding tool execution."""
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        trace_event(
            "model_call_started",
            node=node,
            data={"purpose": purpose, "attempt": attempt},
        )
        try:
            result = await call()
        except Exception as exc:
            retryable = is_transient_model_error(exc)
            will_retry = retryable and attempt < attempts
            summary = _error_summary(exc)
            trace_event(
                "model_call_failed",
                node=node,
                data={
                    "purpose": purpose,
                    "attempt": attempt,
                    "error": summary,
                    "retryable": retryable,
                    "will_retry": will_retry,
                },
            )
            if not retryable:
                trace_event(
                    "model_retry_skipped",
                    node=node,
                    data={"purpose": purpose, "attempt": attempt, "reason": "permanent"},
                )
                raise
            if not will_retry:
                trace_event(
                    "model_retry_exhausted",
                    node=node,
                    data={"purpose": purpose, "attempts": attempt, "error": summary},
                )
                raise

            wait_s = max(0.0, delay_s) * (2 ** (attempt - 1))
            logger.warning(
                f"模型调用 {node}/{purpose} 暂时失败，第 {attempt}/{attempts} 次: {summary}；"
                f"{wait_s:.1f}s 后重试"
            )
            trace_event(
                "model_retry_scheduled",
                node=node,
                data={
                    "purpose": purpose,
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "delay_s": wait_s,
                },
            )
            if wait_s:
                await asyncio.sleep(wait_s)
        else:
            trace_event(
                "model_call_completed",
                node=node,
                data={"purpose": purpose, "attempt": attempt},
            )
            return result

    raise RuntimeError("unreachable model retry state")  # pragma: no cover
