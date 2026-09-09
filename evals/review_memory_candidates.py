"""Inspect, approve and retire long-term memory candidates from the local CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import config  # noqa: E402
from app.memory import (  # noqa: E402
    MemoryReviewService,
    MemoryStatus,
    MemoryType,
    SQLiteMemoryRepository,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审核 Agent 长期记忆候选")
    parser.add_argument("--db-path", default=str(ROOT / config.agent_memory_db_path))
    parser.add_argument("--tenant-id", default="local")
    parser.add_argument("--device-type", default="firewall")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="列出待审核或历史记忆")
    list_parser.add_argument(
        "--status",
        choices=[status.value for status in MemoryStatus],
        default=MemoryStatus.CANDIDATE.value,
    )
    list_parser.add_argument(
        "--memory-type",
        choices=[memory_type.value for memory_type in MemoryType],
    )
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--offset", type=int, default=0)

    show_parser = commands.add_parser("show", help="查看完整记忆和审核历史")
    show_parser.add_argument("memory_id")

    for command, help_text in (
        ("approve", "批准候选记忆"),
        ("retire", "废弃候选或已批准记忆"),
    ):
        review_parser = commands.add_parser(command, help=help_text)
        review_parser.add_argument("memory_id")
        review_parser.add_argument("--reviewer", required=True)
        review_parser.add_argument("--reason", required=True)
    return parser.parse_args()


def _summary(record) -> dict[str, object]:
    return {
        "memory_id": record.memory_id,
        "status": record.status.value,
        "memory_type": record.memory_type.value,
        "scenario": record.scenario,
        "lesson": record.lesson,
        "confidence": record.confidence,
        "evidence_count": len(record.evidence_run_ids) + len(record.evidence_case_ids),
        "updated_at": record.updated_at,
    }


def main() -> None:
    args = _parse_args()
    service = MemoryReviewService(SQLiteMemoryRepository(args.db_path))
    try:
        if args.command == "list":
            records = service.list_for_review(
                tenant_id=args.tenant_id,
                device_type=args.device_type,
                status=MemoryStatus(args.status),
                memory_type=MemoryType(args.memory_type) if args.memory_type else None,
                limit=args.limit,
                offset=args.offset,
            )
            for record in records:
                print(json.dumps(_summary(record), ensure_ascii=False, sort_keys=True))
            print(f"count={len(records)}", file=sys.stderr)
            return

        if args.command == "show":
            record = service.get_detail(
                args.memory_id,
                tenant_id=args.tenant_id,
                device_type=args.device_type,
            )
            payload = record.to_dict()
            payload["reviews"] = [
                review.to_dict()
                for review in service.review_history(
                    args.memory_id,
                    tenant_id=args.tenant_id,
                    device_type=args.device_type,
                )
            ]
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return

        operation = service.approve if args.command == "approve" else service.retire
        updated = operation(
            args.memory_id,
            tenant_id=args.tenant_id,
            device_type=args.device_type,
            reviewer=args.reviewer,
            reason=args.reason,
        )
        print(json.dumps(updated.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"review failed: {exc}") from exc


if __name__ == "__main__":
    main()
