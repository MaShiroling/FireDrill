"""Stable fingerprints used to deduplicate operational memories."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


def normalize_memory_text(value: object) -> str:
    """Normalize harmless formatting differences without changing meaning."""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return " ".join(normalized.split())


def _ordered_unique(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_memory_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            normalize_memory_text(key): _normalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalize_json_value(item) for item in value), key=str)
    if isinstance(value, str):
        return normalize_memory_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return normalize_memory_text(value)


def build_memory_fingerprint(
    *,
    namespace: Iterable[object],
    scenario: str,
    lesson: str,
    preconditions: Mapping[str, Any] | None = None,
    recommended_actions: Iterable[object] = (),
    failure_codes: Iterable[object] = (),
) -> str:
    """Build a content identity; evidence and lifecycle fields are intentionally excluded.

    Failure-code order is not meaningful, while recommended-action order can describe an
    operational procedure and is therefore preserved.
    """
    payload = {
        "namespace": [normalize_memory_text(part) for part in namespace],
        "scenario": normalize_memory_text(scenario),
        "lesson": normalize_memory_text(lesson),
        "preconditions": _normalize_json_value(preconditions or {}),
        "recommended_actions": _ordered_unique(recommended_actions),
        "failure_codes": sorted(set(_ordered_unique(failure_codes))),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
