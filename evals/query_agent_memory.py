"""Synchronize and inspect the approved-only long-term memory index.

Examples:
    .venv/bin/python evals/query_agent_memory.py sync
    .venv/bin/python evals/query_agent_memory.py search "commit 超时后怎么办"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import config  # noqa: E402
from app.memory import (  # noqa: E402
    MemoryType,
    build_default_memory_retrieval_service,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步或检索 Agent 长期记忆")
    parser.add_argument("--db-path", default=str(ROOT / config.agent_memory_db_path))
    parser.add_argument("--tenant-id", default="local")
    parser.add_argument("--device-type", default="firewall")
    parser.add_argument(
        "--memory-type",
        action="append",
        choices=[memory_type.value for memory_type in MemoryType],
        help="可重复指定；默认检索全部记忆类型",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sync", help="把已批准且未过期的 SQLite 记忆同步到独立 Milvus 集合")
    search_parser = commands.add_parser("search", help="检索已批准的长期记忆")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=config.agent_memory_top_k)
    search_parser.add_argument(
        "--sync-first",
        action="store_true",
        help="检索前先刷新当前租户和设备的向量索引",
    )
    return parser.parse_args()


def _memory_types(values: list[str] | None) -> tuple[MemoryType, ...]:
    return tuple(MemoryType(value) for value in values) if values else tuple(MemoryType)


def main() -> None:
    args = _parse_args()
    try:
        service = build_default_memory_retrieval_service(db_path=args.db_path)
        memory_types = _memory_types(args.memory_type)
        if args.command == "sync":
            results = service.sync_scope(
                tenant_id=args.tenant_id,
                device_type=args.device_type,
                memory_types=memory_types,
            )
            payload = [
                {
                    "namespace": result.namespace.to_dict(),
                    "indexed": result.indexed,
                    "removed": result.removed,
                    "degraded": result.degraded,
                    "error": result.error,
                }
                for result in results
            ]
        else:
            if args.sync_first:
                service.sync_scope(
                    tenant_id=args.tenant_id,
                    device_type=args.device_type,
                    memory_types=memory_types,
                )
            result = service.search(
                args.query,
                tenant_id=args.tenant_id,
                device_type=args.device_type,
                memory_types=memory_types,
                top_k=args.top_k,
            )
            payload = {
                "mode": result.mode,
                "degraded_reason": result.degraded_reason,
                "hits": [
                    {
                        "memory_id": hit.record.memory_id,
                        "memory_type": hit.record.memory_type.value,
                        "scenario": hit.record.scenario,
                        "lesson": hit.record.lesson,
                        "recommended_actions": list(hit.record.recommended_actions),
                        "confidence": hit.record.confidence,
                        "score": hit.score,
                        "source": hit.source.value,
                    }
                    for hit in result.hits
                ],
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"memory retrieval failed: {exc}") from exc


if __name__ == "__main__":
    main()
