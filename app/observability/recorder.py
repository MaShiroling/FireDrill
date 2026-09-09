"""Context-local, failure-isolated recorder for Agent execution trajectories."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from collections import Counter
from contextvars import ContextVar, Token
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.observability.redaction import sanitize
from app.observability.schemas import RunTrace, TraceEvent

_current_recorder: ContextVar[RunRecorder | None] = ContextVar("agent_run_recorder", default=None)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return safe.strip("-")[:80] or "run"


def _git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_version_manifest(root: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    status = _git_value(root, "status", "--short")
    return {
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_short_commit": _git_value(root, "rev-parse", "--short", "HEAD"),
        "worktree_dirty": bool(status and status != "unknown"),
        "runtime": sanitize(runtime),
    }


class RunRecorder:
    """Collect trace events and atomically persist a per-run JSON document."""

    def __init__(
        self,
        *,
        root: Path,
        trace_dir: Path,
        session_id: str,
        task: str,
        runtime: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        max_value_chars: int = 8000,
    ) -> None:
        timestamp = datetime.now().astimezone()
        suffix = uuid.uuid4().hex[:10]
        self.root = root
        self.max_value_chars = max_value_chars
        self._started_monotonic = time.perf_counter()
        self._lock = threading.Lock()
        self._finished = False
        self.trace = RunTrace(
            schema_version="1.0",
            run_id=f"{timestamp:%Y%m%dT%H%M%S}-{_safe_name(session_id)}-{suffix}",
            session_id=session_id,
            task=str(sanitize(task, max_value_chars)),
            started_at=_now(),
            status="running",
            version_manifest=build_version_manifest(root, runtime),
            metadata=sanitize(metadata or {}, max_value_chars),
        )
        self.path = trace_dir / f"{timestamp:%Y-%m-%d}" / f"{self.trace.run_id}.json"
        self.record("run_started", data={"session_id": session_id})

    @property
    def run_id(self) -> str:
        return self.trace.run_id

    def relative_path(self) -> str:
        try:
            return str(self.path.relative_to(self.root))
        except ValueError:
            return str(self.path)

    def record(
        self, event_type: str, *, node: str | None = None, data: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            if self._finished:
                return
            self.trace.events.append(
                TraceEvent(
                    sequence=len(self.trace.events) + 1,
                    timestamp=_now(),
                    event_type=event_type,
                    node=node,
                    data=sanitize(data or {}, self.max_value_chars),
                )
            )
            self._persist()

    def finish(
        self,
        *,
        status: str,
        final_state: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        with self._lock:
            if self._finished:
                return
            duration = round(time.perf_counter() - self._started_monotonic, 3)
            self.trace.events.append(
                TraceEvent(
                    sequence=len(self.trace.events) + 1,
                    timestamp=_now(),
                    event_type="run_finished",
                    data={"status": status, "duration_s": duration},
                )
            )
            event_counts = Counter(event.event_type for event in self.trace.events)
            node_counts = Counter(event.node for event in self.trace.events if event.node)
            self.trace.status = status
            self.trace.ended_at = _now()
            self.trace.duration_s = duration
            self.trace.final_state = sanitize(final_state or {}, self.max_value_chars)
            self.trace.error = str(sanitize(error, self.max_value_chars))
            self.trace.metrics = {
                "event_count": len(self.trace.events),
                "event_counts": dict(sorted(event_counts.items())),
                "node_event_counts": dict(sorted(node_counts.items())),
                "model_calls": event_counts.get("model_call_started", 0),
                "model_call_failures": event_counts.get("model_call_failed", 0),
                "model_retries_scheduled": event_counts.get("model_retry_scheduled", 0),
                "model_retry_exhausted": event_counts.get("model_retry_exhausted", 0),
                "replanner_decisions": event_counts.get("replanner_decision", 0),
                "mcp_attempts": event_counts.get("mcp_attempt_completed", 0),
                "mcp_retries_scheduled": event_counts.get("mcp_retry_scheduled", 0),
                "mcp_retries_skipped": event_counts.get("mcp_retry_skipped", 0),
                "mcp_retry_exhausted": event_counts.get("mcp_call_exhausted", 0),
                "tool_calls": event_counts.get("tool_call_completed", 0),
            }
            self._finished = True
            self._persist()

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(self.trace.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)


def get_current_recorder() -> RunRecorder | None:
    return _current_recorder.get()


def trace_event(
    event_type: str, *, node: str | None = None, data: dict[str, Any] | None = None
) -> None:
    """Record an event without allowing observability failures to affect the Agent."""
    recorder = get_current_recorder()
    if recorder is None:
        return
    try:
        recorder.record(event_type, node=node, data=data)
    except Exception as exc:  # pragma: no cover - depends on filesystem failures
        logger.warning(f"轨迹事件写入失败，不影响 Agent 执行: {exc}")


def begin_run(
    *,
    enabled: bool,
    root: Path,
    trace_dir: Path,
    session_id: str,
    task: str,
    runtime: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    max_value_chars: int = 8000,
) -> tuple[RunRecorder | None, Token[RunRecorder | None] | None]:
    if not enabled:
        return None, None
    try:
        recorder = RunRecorder(
            root=root,
            trace_dir=trace_dir,
            session_id=session_id,
            task=task,
            runtime=runtime,
            metadata=metadata,
            max_value_chars=max_value_chars,
        )
        return recorder, _current_recorder.set(recorder)
    except Exception as exc:  # pragma: no cover - depends on filesystem failures
        logger.warning(f"轨迹初始化失败，不影响 Agent 执行: {exc}")
        return None, None


def finish_run(
    recorder: RunRecorder | None,
    token: Token[RunRecorder | None] | None,
    *,
    status: str,
    final_state: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    try:
        if recorder is not None:
            recorder.finish(status=status, final_state=final_state, error=error)
    except Exception as exc:  # pragma: no cover - depends on filesystem failures
        logger.warning(f"轨迹收尾失败，不影响 Agent 执行: {exc}")
    finally:
        if token is not None:
            try:
                _current_recorder.reset(token)
            except ValueError:  # pragma: no cover - defensive context isolation
                pass
