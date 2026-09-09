"""Human review service for candidate operational memories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from app.memory.repository import MemoryRepository, MemoryReviewEvent
from app.memory.schemas import MemoryNamespace, MemoryRecord, MemoryStatus, MemoryType

if TYPE_CHECKING:
    from app.memory.retrieval import MemoryIndexSyncResult


class MemoryIndexSynchronizer(Protocol):
    """Minimal index refresh boundary used after a review transition."""

    def sync_namespace(self, namespace: MemoryNamespace) -> MemoryIndexSyncResult: ...


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class MemoryReviewService:
    """Scope-aware review operations that never bypass the memory lifecycle."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        index_synchronizer: MemoryIndexSynchronizer | None = None,
    ) -> None:
        self.repository = repository
        self.index_synchronizer = index_synchronizer

    def _sync_after_transition(self, record: MemoryRecord) -> MemoryIndexSyncResult | None:
        """Refresh one namespace without rolling back the authoritative review."""
        if self.index_synchronizer is None:
            return None
        try:
            result = self.index_synchronizer.sync_namespace(record.namespace)
        except Exception as exc:  # noqa: BLE001 - SQLite review is already committed
            logger.warning(
                "长期记忆审核已保存，但向量索引自动同步异常: memory_id={} error={}",
                record.memory_id,
                exc,
            )
            return None
        if result.degraded:
            logger.warning(
                "长期记忆审核已保存，但向量索引自动同步降级: memory_id={} error={}",
                record.memory_id,
                result.error,
            )
        else:
            logger.info(
                "长期记忆向量索引已自动同步: memory_id={} indexed={} removed={}",
                record.memory_id,
                result.indexed,
                result.removed,
            )
        return result

    @staticmethod
    def _namespace(tenant_id: str, device_type: str, memory_type: MemoryType) -> MemoryNamespace:
        return MemoryNamespace(
            tenant_id=tenant_id,
            device_type=device_type,
            memory_type=memory_type,
        )

    def _get_scoped(self, memory_id: str, tenant_id: str, device_type: str) -> MemoryRecord:
        record = self.repository.get(memory_id)
        if record is None:
            raise KeyError(f"memory not found: {memory_id}")
        expected = self._namespace(tenant_id, device_type, record.memory_type)
        if record.namespace != expected:
            # Avoid revealing whether a memory exists in another tenant or device scope.
            raise KeyError(f"memory not found: {memory_id}")
        return record

    def list_for_review(
        self,
        *,
        tenant_id: str,
        device_type: str,
        status: MemoryStatus = MemoryStatus.CANDIDATE,
        memory_type: MemoryType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must not be negative")
        types = (MemoryType(memory_type),) if memory_type is not None else tuple(MemoryType)
        fetch_limit = limit + offset
        records = [
            record
            for target_type in types
            for record in self.repository.list_memories(
                self._namespace(tenant_id, device_type, target_type),
                status=MemoryStatus(status),
                limit=fetch_limit,
            )
        ]
        records.sort(key=lambda item: (_timestamp(item.updated_at), item.memory_id), reverse=True)
        return records[offset : offset + limit]

    def get_detail(self, memory_id: str, *, tenant_id: str, device_type: str) -> MemoryRecord:
        return self._get_scoped(memory_id, tenant_id, device_type)

    def approve(
        self,
        memory_id: str,
        *,
        tenant_id: str,
        device_type: str,
        reviewer: str,
        reason: str,
    ) -> MemoryRecord:
        self._get_scoped(memory_id, tenant_id, device_type)
        updated = self.repository.set_status(
            memory_id,
            MemoryStatus.APPROVED,
            reviewer=reviewer,
            reason=reason,
        )
        self._sync_after_transition(updated)
        return updated

    def retire(
        self,
        memory_id: str,
        *,
        tenant_id: str,
        device_type: str,
        reviewer: str,
        reason: str,
    ) -> MemoryRecord:
        self._get_scoped(memory_id, tenant_id, device_type)
        updated = self.repository.set_status(
            memory_id,
            MemoryStatus.RETIRED,
            reviewer=reviewer,
            reason=reason,
        )
        self._sync_after_transition(updated)
        return updated

    def review_history(
        self, memory_id: str, *, tenant_id: str, device_type: str
    ) -> list[MemoryReviewEvent]:
        self._get_scoped(memory_id, tenant_id, device_type)
        return self.repository.list_reviews(memory_id)


def build_default_memory_review_service(
    *,
    db_path: str | Path | None = None,
) -> MemoryReviewService:
    """Build the CLI/runtime review service with automatic index refresh."""
    from app.config import config
    from app.memory.repository import SQLiteMemoryRepository
    from app.memory.retrieval import build_default_memory_retrieval_service

    resolved_path = db_path or config.agent_memory_db_path
    return MemoryReviewService(
        SQLiteMemoryRepository(resolved_path),
        index_synchronizer=build_default_memory_retrieval_service(db_path=resolved_path),
    )
