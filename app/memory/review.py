"""Human review service for candidate operational memories."""

from __future__ import annotations

from datetime import datetime

from app.memory.repository import MemoryRepository, MemoryReviewEvent
from app.memory.schemas import MemoryNamespace, MemoryRecord, MemoryStatus, MemoryType


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class MemoryReviewService:
    """Scope-aware review operations that never bypass the memory lifecycle."""

    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

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
        return self.repository.set_status(
            memory_id,
            MemoryStatus.APPROVED,
            reviewer=reviewer,
            reason=reason,
        )

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
        return self.repository.set_status(
            memory_id,
            MemoryStatus.RETIRED,
            reviewer=reviewer,
            reason=reason,
        )

    def review_history(
        self, memory_id: str, *, tenant_id: str, device_type: str
    ) -> list[MemoryReviewEvent]:
        self._get_scoped(memory_id, tenant_id, device_type)
        return self.repository.list_reviews(memory_id)
