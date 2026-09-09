"""Extract reviewable long-term memory candidates from flywheel artifacts.

Example:
    .venv/bin/python evals/build_memory_candidates.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import config  # noqa: E402
from app.memory import (  # noqa: E402
    SQLiteMemoryRepository,
    extract_failure_memories,
    extract_recovered_memories,
    persist_memory_candidates,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从数据飞轮产物提取长期记忆候选")
    parser.add_argument(
        "--failure-pool",
        default=str(ROOT / "evals" / "artifacts" / "flywheel" / "failure_pool.jsonl"),
    )
    parser.add_argument(
        "--regression-report",
        default=str(
            ROOT / "evals" / "artifacts" / "flywheel" / "regression" / "regression_report.json"
        ),
    )
    parser.add_argument(
        "--replay-cases",
        default=str(ROOT / "evals" / "artifacts" / "flywheel" / "replay_cases.json"),
    )
    parser.add_argument("--db-path", default=str(ROOT / config.agent_memory_db_path))
    parser.add_argument("--tenant-id", default="local")
    parser.add_argument("--device-type", default="firewall")
    parser.add_argument("--skip-successes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"failure pool record must be an object: {path}:{line_number}")
        records.append(value)
    return records


def _artifact_revision(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return f"flywheel-{digest.hexdigest()[:12]}"


def main() -> None:
    args = _parse_args()
    failure_path = Path(args.failure_pool)
    regression_path = Path(args.regression_report)
    replay_path = Path(args.replay_cases)
    source_paths = [failure_path]
    if not args.skip_successes:
        source_paths.extend((regression_path, replay_path))
    source_revision = _artifact_revision(source_paths)

    failure_samples = _load_jsonl(failure_path)
    failures = extract_failure_memories(
        failure_samples,
        tenant_id=args.tenant_id,
        device_type=args.device_type,
        source_revision=source_revision,
    )
    successes = []
    if not args.skip_successes:
        report = _load_json(regression_path)
        replay_cases = _load_json(replay_path)
        if not isinstance(report, dict) or not isinstance(replay_cases, list):
            raise ValueError("regression report must be an object and replay cases must be a list")
        replay_catalog = {
            str(case["id"]): case
            for case in replay_cases
            if isinstance(case, dict) and case.get("id")
        }
        successes = extract_recovered_memories(
            report,
            replay_catalog,
            tenant_id=args.tenant_id,
            device_type=args.device_type,
            source_revision=source_revision,
        )

    candidates = [*failures, *successes]
    print(
        f"source_revision={source_revision} failures={len(failures)} "
        f"successes={len(successes)} total={len(candidates)}"
    )
    if args.dry_run:
        for candidate in candidates:
            print(json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True))
        return

    repository = SQLiteMemoryRepository(args.db_path)
    stored = persist_memory_candidates(repository, candidates)
    print(f"stored={len(stored)} db={repository.db_path}")


if __name__ == "__main__":
    main()
