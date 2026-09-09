"""Vector-index boundary for approved long-term operational memories.

SQLite remains the source of truth.  This module owns a rebuildable Milvus
projection in a collection that is deliberately separate from the RAG corpus.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.memory.schemas import MemoryNamespace


@dataclass(frozen=True)
class MemoryVectorEntry:
    """One searchable projection of a reviewed memory."""

    memory_id: str
    namespace: MemoryNamespace
    vector: tuple[float, ...]
    embedding_model: str
    embedding_version: str
    updated_at: str


@dataclass(frozen=True)
class MemoryVectorMatch:
    """Minimal vector candidate; authoritative metadata is read from SQLite."""

    memory_id: str
    score: float


@dataclass(frozen=True)
class MemoryVectorSyncStats:
    """Changes made while replacing one namespace projection."""

    indexed: int
    removed: int


class MemoryVectorIndex(Protocol):
    """Replaceable vector projection used by the retrieval service."""

    def replace_namespace(
        self,
        namespace: MemoryNamespace,
        entries: Sequence[MemoryVectorEntry],
    ) -> MemoryVectorSyncStats: ...

    def search(
        self,
        namespace: MemoryNamespace,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> list[MemoryVectorMatch]: ...


class LazyMemoryVectorIndex:
    """Delay Milvus connection until vector operations are actually attempted."""

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        collection_name: str = "agent_memory_v1",
        vector_dim: int = 1024,
    ) -> None:
        self.client_factory = client_factory
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self._delegate: MilvusMemoryVectorIndex | None = None

    def _get_delegate(self) -> MilvusMemoryVectorIndex:
        if self._delegate is None:
            self._delegate = MilvusMemoryVectorIndex(
                self.client_factory(),
                collection_name=self.collection_name,
                vector_dim=self.vector_dim,
            )
        return self._delegate

    def replace_namespace(
        self,
        namespace: MemoryNamespace,
        entries: Sequence[MemoryVectorEntry],
    ) -> MemoryVectorSyncStats:
        return self._get_delegate().replace_namespace(namespace, entries)

    def search(
        self,
        namespace: MemoryNamespace,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> list[MemoryVectorMatch]:
        return self._get_delegate().search(namespace, query_vector, limit=limit)


def _literal(value: str) -> str:
    """Encode a safe Milvus string literal."""
    return json.dumps(value, ensure_ascii=False)


def _namespace_filter(namespace: MemoryNamespace) -> str:
    tenant_id, device_type, memory_type = namespace.as_tuple()
    return (
        f"tenant_id == {_literal(tenant_id)} and "
        f"device_type == {_literal(device_type)} and "
        f"memory_type == {_literal(memory_type)}"
    )


class MilvusMemoryVectorIndex:
    """Milvus-backed, rebuildable projection of approved memories."""

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str = "agent_memory_v1",
        vector_dim: int = 1024,
    ) -> None:
        clean_name = collection_name.strip()
        if not clean_name:
            raise ValueError("collection_name must not be empty")
        if vector_dim <= 0:
            raise ValueError("vector_dim must be positive")
        self.client = client
        self.collection_name = clean_name
        self.vector_dim = vector_dim
        self._ready = False

    def _ensure_collection(self) -> None:
        if self._ready:
            return
        if not self.client.has_collection(collection_name=self.collection_name):
            # Simple collection creation also creates a COSINE index and loads it.
            # Extra fields are dynamic projection metadata, never the source of truth.
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.vector_dim,
                primary_field_name="memory_id",
                id_type="string",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
                consistency_level="Strong",
            )
        else:
            self.client.load_collection(collection_name=self.collection_name)
        self._ready = True

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"memory embedding dimension mismatch: {len(vector)} != {self.vector_dim}"
            )
        return [float(value) for value in vector]

    def replace_namespace(
        self,
        namespace: MemoryNamespace,
        entries: Sequence[MemoryVectorEntry],
    ) -> MemoryVectorSyncStats:
        self._ensure_collection()
        for entry in entries:
            if entry.namespace != namespace:
                raise ValueError("all vector entries must belong to the target namespace")

        rows = self.client.query(
            collection_name=self.collection_name,
            filter=_namespace_filter(namespace),
            output_fields=["memory_id"],
            consistency_level="Strong",
        )
        existing_ids = {str(row["memory_id"]) for row in rows}
        current_ids = {entry.memory_id for entry in entries}

        if entries:
            self.client.upsert(
                collection_name=self.collection_name,
                data=[
                    {
                        "memory_id": entry.memory_id,
                        "vector": self._validate_vector(entry.vector),
                        "tenant_id": entry.namespace.tenant_id,
                        "device_type": entry.namespace.device_type,
                        "memory_type": entry.namespace.memory_type.value,
                        "embedding_model": entry.embedding_model,
                        "embedding_version": entry.embedding_version,
                        "updated_at": entry.updated_at,
                    }
                    for entry in entries
                ],
            )

        stale_ids = sorted(existing_ids - current_ids)
        if stale_ids:
            self.client.delete(collection_name=self.collection_name, ids=stale_ids)
        return MemoryVectorSyncStats(indexed=len(entries), removed=len(stale_ids))

    def search(
        self,
        namespace: MemoryNamespace,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> list[MemoryVectorMatch]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._ensure_collection()
        result_sets = self.client.search(
            collection_name=self.collection_name,
            data=[self._validate_vector(query_vector)],
            filter=_namespace_filter(namespace),
            limit=limit,
            output_fields=["memory_id"],
            search_params={"metric_type": "COSINE", "params": {}},
            consistency_level="Strong",
        )
        if not result_sets:
            return []

        matches: list[MemoryVectorMatch] = []
        for hit in result_sets[0]:
            if isinstance(hit, dict):
                entity = hit.get("entity") or {}
                memory_id = hit.get("id") or entity.get("memory_id")
                score = hit.get("distance", hit.get("score", 0.0))
            else:
                entity = getattr(hit, "entity", {}) or {}
                memory_id = getattr(hit, "id", None) or entity.get("memory_id")
                score = getattr(hit, "distance", 0.0)
            if memory_id is not None:
                matches.append(MemoryVectorMatch(memory_id=str(memory_id), score=float(score)))
        return matches
