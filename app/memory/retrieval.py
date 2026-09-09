"""Approved-only retrieval with vector search and a local lexical fallback."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from loguru import logger

from app.memory.repository import MemoryRepository
from app.memory.schemas import MemoryNamespace, MemoryRecord, MemoryType
from app.memory.vector_index import (
    LazyMemoryVectorIndex,
    MemoryVectorEntry,
    MemoryVectorIndex,
    MemoryVectorMatch,
)


class MemoryEmbedder(Protocol):
    """Subset of the LangChain embeddings interface used by memory retrieval."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LazyMemoryEmbedder:
    """Delay provider initialization so lexical fallback remains available."""

    def __init__(self, factory: Callable[[], MemoryEmbedder]) -> None:
        self.factory = factory
        self._delegate: MemoryEmbedder | None = None

    def _get_delegate(self) -> MemoryEmbedder:
        if self._delegate is None:
            self._delegate = self.factory()
        return self._delegate

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_delegate().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_delegate().embed_query(text)


class MemorySearchSource(StrEnum):
    VECTOR = "vector"
    LEXICAL = "lexical"


@dataclass(frozen=True)
class MemorySearchHit:
    record: MemoryRecord
    score: float
    source: MemorySearchSource


@dataclass(frozen=True)
class MemorySearchResult:
    hits: tuple[MemorySearchHit, ...]
    mode: str
    degraded_reason: str | None = None


@dataclass(frozen=True)
class MemoryIndexSyncResult:
    namespace: MemoryNamespace
    indexed: int
    removed: int
    degraded: bool = False
    error: str | None = None


def render_memory_text(record: MemoryRecord) -> str:
    """Build stable semantic text without exposing audit-only metadata."""
    parts = [f"场景：{record.scenario}", f"经验：{record.lesson}"]
    if record.preconditions:
        parts.append(
            "前置条件：" + json.dumps(record.preconditions, ensure_ascii=False, sort_keys=True)
        )
    if record.recommended_actions:
        parts.append("建议动作：" + "；".join(record.recommended_actions))
    if record.failure_codes:
        parts.append("失败码：" + "；".join(record.failure_codes))
    return "\n".join(parts)


_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return set(_TOKEN_PATTERN.findall(normalized))


def _lexical_score(query_tokens: set[str], record: MemoryRecord) -> float:
    record_tokens = _tokens(render_memory_text(record))
    overlap = len(query_tokens & record_tokens)
    if not overlap:
        return 0.0
    return overlap / math.sqrt(len(query_tokens) * len(record_tokens))


class MemoryRetrievalService:
    """Retrieve reviewed memory while treating Milvus as an untrusted cache."""

    def __init__(
        self,
        repository: MemoryRepository,
        vector_index: MemoryVectorIndex,
        embedder: MemoryEmbedder,
        *,
        default_top_k: int = 3,
        default_min_score: float = 0.35,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
        if not -1.0 <= default_min_score <= 1.0:
            raise ValueError("default_min_score must be between -1 and 1")
        self.repository = repository
        self.vector_index = vector_index
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score

    def sync_namespace(
        self,
        namespace: MemoryNamespace,
        *,
        now: datetime | None = None,
    ) -> MemoryIndexSyncResult:
        """Rebuild one namespace from currently approved, unexpired SQLite rows."""
        records = self.repository.list_memories(
            namespace,
            searchable_only=True,
            limit=100_000,
            now=now,
        )
        try:
            texts = [render_memory_text(record) for record in records]
            vectors = self.embedder.embed_documents(texts) if texts else []
            if len(vectors) != len(records):
                raise RuntimeError(
                    "embedding result count mismatch: "
                    f"received {len(vectors)} for {len(records)} memories"
                )
            entries = [
                MemoryVectorEntry(
                    memory_id=record.memory_id,
                    namespace=record.namespace,
                    vector=tuple(vector),
                    embedding_model=record.embedding_model,
                    embedding_version=record.embedding_version,
                    updated_at=record.updated_at,
                )
                for record, vector in zip(records, vectors, strict=True)
            ]
            stats = self.vector_index.replace_namespace(namespace, entries)
            return MemoryIndexSyncResult(
                namespace=namespace,
                indexed=stats.indexed,
                removed=stats.removed,
            )
        except Exception as exc:  # noqa: BLE001 - degradation must cover provider failures
            logger.warning("长期记忆向量索引同步失败，SQLite 事实源不受影响: {}", exc)
            return MemoryIndexSyncResult(
                namespace=namespace,
                indexed=0,
                removed=0,
                degraded=True,
                error=str(exc),
            )

    def sync_scope(
        self,
        *,
        tenant_id: str,
        device_type: str,
        memory_types: Sequence[MemoryType] = tuple(MemoryType),
        now: datetime | None = None,
    ) -> tuple[MemoryIndexSyncResult, ...]:
        return tuple(
            self.sync_namespace(
                MemoryNamespace(
                    tenant_id=tenant_id,
                    device_type=device_type,
                    memory_type=memory_type,
                ),
                now=now,
            )
            for memory_type in memory_types
        )

    def _approved_record(
        self,
        match: MemoryVectorMatch,
        namespace: MemoryNamespace,
        *,
        now: datetime | None,
    ) -> MemoryRecord | None:
        record = self.repository.get(match.memory_id)
        if record is None or record.namespace != namespace or not record.is_searchable(now):
            return None
        return record

    def _lexical_hits(
        self,
        query: str,
        namespaces: Sequence[MemoryNamespace],
        *,
        limit: int,
        excluded_ids: set[str],
        now: datetime | None,
    ) -> list[MemorySearchHit]:
        if limit <= 0:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        ranked: list[MemorySearchHit] = []
        for namespace in namespaces:
            for record in self.repository.list_memories(
                namespace,
                searchable_only=True,
                limit=100_000,
                now=now,
            ):
                if record.memory_id in excluded_ids:
                    continue
                score = _lexical_score(query_tokens, record)
                if score > 0:
                    ranked.append(
                        MemorySearchHit(
                            record=record,
                            score=score,
                            source=MemorySearchSource.LEXICAL,
                        )
                    )
        ranked.sort(key=lambda hit: (-hit.score, -hit.record.confidence, hit.record.memory_id))
        return ranked[:limit]

    def search_lexical(
        self,
        query: str,
        *,
        tenant_id: str,
        device_type: str,
        memory_types: Sequence[MemoryType] = tuple(MemoryType),
        top_k: int | None = None,
        now: datetime | None = None,
        degraded_reason: str | None = None,
    ) -> MemorySearchResult:
        """Search approved SQLite memories without embeddings or Milvus.

        This path is intentionally provider-free so the Planner can still receive
        a small, reviewed context when semantic retrieval exceeds its latency
        budget or an external dependency is unavailable.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        limit = self.default_top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        namespaces = tuple(
            MemoryNamespace(
                tenant_id=tenant_id,
                device_type=device_type,
                memory_type=memory_type,
            )
            for memory_type in memory_types
        )
        hits = tuple(
            self._lexical_hits(
                clean_query,
                namespaces,
                limit=limit,
                excluded_ids=set(),
                now=now,
            )
        )
        return MemorySearchResult(
            hits=hits,
            mode=MemorySearchSource.LEXICAL.value,
            degraded_reason=degraded_reason,
        )

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        device_type: str,
        memory_types: Sequence[MemoryType] = tuple(MemoryType),
        top_k: int | None = None,
        min_score: float | None = None,
        now: datetime | None = None,
    ) -> MemorySearchResult:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        limit = self.default_top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        score_threshold = self.default_min_score if min_score is None else min_score
        if not -1.0 <= score_threshold <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
        namespaces = tuple(
            MemoryNamespace(
                tenant_id=tenant_id,
                device_type=device_type,
                memory_type=memory_type,
            )
            for memory_type in memory_types
        )

        vector_hits: list[MemorySearchHit] = []
        degraded_reason: str | None = None
        try:
            query_vector = self.embedder.embed_query(clean_query)
            candidate_limit = max(limit * 8, 20)
            candidates: list[tuple[MemoryVectorMatch, MemoryNamespace]] = []
            for namespace in namespaces:
                candidates.extend(
                    (match, namespace)
                    for match in self.vector_index.search(
                        namespace,
                        query_vector,
                        limit=candidate_limit,
                    )
                )
            candidates.sort(key=lambda item: (-item[0].score, item[0].memory_id))
            seen: set[str] = set()
            for match, namespace in candidates:
                if match.score < score_threshold:
                    continue
                if match.memory_id in seen:
                    continue
                record = self._approved_record(match, namespace, now=now)
                if record is None:
                    continue
                seen.add(record.memory_id)
                vector_hits.append(
                    MemorySearchHit(
                        record=record,
                        score=match.score,
                        source=MemorySearchSource.VECTOR,
                    )
                )
                if len(vector_hits) == limit:
                    break
        except Exception as exc:  # noqa: BLE001 - degradation must cover provider failures
            degraded_reason = str(exc)
            logger.warning("长期记忆向量检索失败，降级为本地词法检索: {}", exc)

        excluded_ids = {hit.record.memory_id for hit in vector_hits}
        lexical_hits = self._lexical_hits(
            clean_query,
            namespaces,
            limit=limit - len(vector_hits),
            excluded_ids=excluded_ids,
            now=now,
        )
        hits = (*vector_hits, *lexical_hits)
        if vector_hits and lexical_hits:
            mode = "hybrid"
        elif vector_hits:
            mode = MemorySearchSource.VECTOR.value
        else:
            mode = MemorySearchSource.LEXICAL.value
        return MemorySearchResult(hits=hits, mode=mode, degraded_reason=degraded_reason)


def build_default_memory_retrieval_service(
    *,
    db_path: str | Path | None = None,
) -> MemoryRetrievalService:
    """Wire the local SQLite source, shared Milvus client and DashScope embedder lazily."""
    from app.config import config
    from app.memory.repository import SQLiteMemoryRepository

    def connect_milvus():
        from app.core.milvus_client import milvus_manager

        return milvus_manager.connect()

    def load_embedder() -> MemoryEmbedder:
        from app.services.vector_embedding_service import vector_embedding_service

        return vector_embedding_service

    return MemoryRetrievalService(
        repository=SQLiteMemoryRepository(db_path or config.agent_memory_db_path),
        vector_index=LazyMemoryVectorIndex(
            connect_milvus,
            collection_name=config.agent_memory_collection_name,
            vector_dim=config.agent_memory_vector_dim,
        ),
        embedder=LazyMemoryEmbedder(load_embedder),
        default_top_k=config.agent_memory_top_k,
        default_min_score=config.agent_memory_min_score,
    )
