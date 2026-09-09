"""Top-level deterministic evaluator for one Agent run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.deterministic import (
    TrafficProbe,
    claims_failure,
    claims_success,
    evaluate_assertions,
)
from app.evaluation.failure_classifier import classify_failures
from app.evaluation.schemas import EvaluationResult


def load_trace(trace_path: str | Path | None, *, root: Path | None = None) -> dict[str, Any] | None:
    """Load a trace if present; absence never prevents terminal-state scoring."""
    if not trace_path:
        return None
    path = Path(trace_path)
    if not path.is_absolute() and root is not None:
        path = root / path
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


async def evaluate_run(
    *,
    assertions: list[dict[str, Any]],
    snapshot: dict[str, Any],
    report: str,
    expect_success: bool,
    run_error: str = "",
    steps: int = 0,
    trace: dict[str, Any] | None = None,
    traffic_probe: TrafficProbe | None = None,
) -> EvaluationResult:
    assertion_results = await evaluate_assertions(
        assertions,
        snapshot,
        report,
        traffic_probe=traffic_probe,
    )
    success = all(result.passed for result in assertion_results) and not run_error
    report_claims_success = claims_success(report)
    report_claims_failure = claims_failure(report)
    fake_completion = report_claims_success and not success
    false_failure = expect_success and success and report_claims_failure
    correct_failure = not expect_success and not report_claims_success
    failure_codes, failure_evidence = classify_failures(
        success=success,
        expect_success=expect_success,
        fake_completion=fake_completion,
        false_failure=false_failure,
        assertion_results=assertion_results,
        snapshot=snapshot,
        trace=trace,
        run_error=run_error,
        steps=steps,
    )
    return EvaluationResult(
        schema_version="1.0",
        success=success,
        expect_success=expect_success,
        claims_success=report_claims_success,
        claims_failure=report_claims_failure,
        fake_completion=fake_completion,
        false_failure=false_failure,
        correct_failure=correct_failure,
        assertion_results=assertion_results,
        failure_codes=failure_codes,
        failure_evidence=failure_evidence,
        trace_available=trace is not None,
    )
