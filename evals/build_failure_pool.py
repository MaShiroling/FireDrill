"""Mine evaluation results into a deduplicated failure pool and replay suite.

Example:
    .venv/bin/python evals/build_failure_pool.py \
      --results evals/results/current.jsonl \
      --results evals/results/legacy.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.flywheel import (  # noqa: E402
    build_failure_pool,
    load_case_catalog,
    load_result_records,
    write_failure_artifacts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建失败样本池和可执行回放用例")
    parser.add_argument(
        "--results",
        action="append",
        default=[],
        help="评测 JSONL，可重复传入；默认 evals/results/current.jsonl",
    )
    parser.add_argument("--cases", default=str(ROOT / "evals" / "cases_firewall.json"))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "evals" / "artifacts" / "flywheel"),
    )
    parser.add_argument(
        "--include-handled",
        action="store_true",
        help="同时收录已正确处理的非法参数、提交拒绝等困难正样本",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_paths = [Path(path) for path in args.results] or [
        ROOT / "evals" / "results" / "current.jsonl"
    ]
    records = load_result_records(result_paths)
    catalog = load_case_catalog(Path(args.cases))
    pool, replay_cases, manifest = build_failure_pool(
        records,
        catalog,
        root=ROOT,
        include_handled=args.include_handled,
    )
    paths = write_failure_artifacts(Path(args.output_dir), pool, replay_cases, manifest)

    print(
        f"输入 {manifest['input_record_count']} 次运行 -> "
        f"失败 {manifest['failure_occurrence_count']} 次 -> "
        f"去重样本 {manifest['deduplicated_sample_count']} 条 -> "
        f"回放用例 {manifest['replay_case_count']} 条"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
