"""Best-effort redaction and size limits for structured traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

REDACTED = "[REDACTED]"
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "passwd",
    "client_secret",
    "access_token",
    "refresh_token",
)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}...[TRUNCATED {omitted} chars]"


def sanitize(value: Any, max_chars: int = 8000) -> Any:
    """Convert arbitrary values into bounded, JSON-safe, redacted structures."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, bytes):
        return _truncate(value.decode("utf-8", errors="replace"), max_chars)
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize(asdict(value), max_chars=max_chars)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return sanitize(model_dump(), max_chars=max_chars)
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_sensitive_key(key) else sanitize(item, max_chars)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item, max_chars) for item in value]
    return _truncate(str(value), max_chars)
