"""Serializable schemas for deterministic Agent evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FailureCode(StrEnum):
    """Stable machine-readable labels used by the data flywheel."""

    RUN_ERROR = "run_error"
    EVALUATOR_ERROR = "evaluator_error"
    ASSERTION_FAILED = "assertion_failed"
    PLANNING_FAILURE = "planning_failure"
    MODEL_CALL_FAILURE = "model_call_failure"
    MODEL_RETRY_EXHAUSTED = "model_retry_exhausted"
    TOOL_SELECTION_FAILURE = "tool_selection_failure"
    TOOL_EXECUTION_FAILURE = "tool_execution_failure"
    RETRY_EXHAUSTED = "retry_exhausted"
    INVALID_ARGUMENT = "invalid_argument"
    COMMIT_REJECTED = "commit_rejected"
    COMMIT_STATE_UNKNOWN = "commit_state_unknown"
    VERIFICATION_MISSING = "verification_missing"
    PENDING_CHANGES = "pending_changes"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    FALSE_COMPLETION = "false_completion"
    FALSE_FAILURE = "false_failure"
    REPORT_INCONSISTENT = "report_inconsistent"


@dataclass(frozen=True)
class AssertionResult:
    assertion: dict[str, Any]
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "assert": self.assertion,
            "pass": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class EvaluationResult:
    schema_version: str
    success: bool
    expect_success: bool
    claims_success: bool
    claims_failure: bool
    fake_completion: bool
    false_failure: bool
    correct_failure: bool
    assertion_results: list[AssertionResult]
    failure_codes: list[FailureCode] = field(default_factory=list)
    failure_evidence: dict[str, list[str]] = field(default_factory=dict)
    trace_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["assertion_results"] = [result.to_dict() for result in self.assertion_results]
        data["failure_codes"] = [code.value for code in self.failure_codes]
        return data
