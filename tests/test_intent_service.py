"""意图与风险识别测试。"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import intent
from app.models.intent import (
    DecisionSource,
    IntentType,
    LLMIntentResult,
    RecommendedRoute,
    RiskLevel,
)
from app.services.intent_service import IntentRiskService, merge_decisions


@pytest.fixture()
def service() -> IntentRiskService:
    return IntentRiskService(llm_enabled=False)


class StubLLMClassifier:
    def __init__(self, result: LLMIntentResult) -> None:
        self.result = result
        self.calls = 0

    async def classify(self, _text: str) -> LLMIntentResult:
        self.calls += 1
        return self.result


class SlowLLMClassifier:
    async def classify(self, _text: str) -> LLMIntentResult:
        await asyncio.sleep(0.05)
        raise AssertionError("wait_for 应在这里之前超时")


@pytest.mark.parametrize(
    ("text", "intent", "risk", "route"),
    [
        ("CPU 使用率持续超过 90% 时应该如何排查？", IntentType.KNOWLEDGE_QA,
         RiskLevel.READ_ONLY, RecommendedRoute.RAG),
        ("请查询当前防火墙规则，不要修改配置", IntentType.OPS_READ,
         RiskLevel.READ_ONLY, RecommendedRoute.AGENT),
        ("当前防火墙是否已经放通 TCP 443？", IntentType.OPS_READ,
         RiskLevel.READ_ONLY, RecommendedRoute.AGENT),
        ("请确认当前防火墙是否已经放通 TCP 443？", IntentType.OPS_READ,
         RiskLevel.READ_ONLY, RecommendedRoute.AGENT),
        ("如何新增一条防火墙规则？", IntentType.KNOWLEDGE_QA,
         RiskLevel.READ_ONLY, RecommendedRoute.RAG),
        ("请新增一条 TCP 443 放通规则并提交", IntentType.CONFIG_CHANGE,
         RiskLevel.WRITE, RecommendedRoute.CONFIRM),
        ("删除 rule-003", IntentType.CONFIG_CHANGE,
         RiskLevel.WRITE, RecommendedRoute.CONFIRM),
    ],
)
def test_classification(service, text, intent, risk, route) -> None:
    decision = service.classify_rules(text)

    assert decision.intent == intent
    assert decision.risk == risk
    assert decision.route == route


def test_ambiguous_ops_request_requires_clarification(service) -> None:
    decision = service.classify_rules("防火墙规则")

    assert decision.intent == IntentType.AMBIGUOUS
    assert decision.risk == RiskLevel.UNKNOWN
    assert decision.route == RecommendedRoute.ASK_USER
    assert decision.requires_confirmation is True


def test_classify_intent_api_returns_explainable_decision() -> None:
    app = FastAPI()
    app.include_router(intent.router, prefix="/api")

    with TestClient(app) as client:
        response = client.post(
            "/api/intent/classify",
            json={"text": "请删除 rule-003 并提交"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "intent": "config_change",
        "risk": "write",
        "confidence": 0.84,
        "route": "confirm",
        "requires_confirmation": True,
        "reason": "检测到配置变更动作；执行前必须获得用户确认。",
        "matched_signals": ["删除", "提交"],
        "decision_source": "rule",
        "llm_used": False,
        "rule_decision": None,
        "llm_decision": None,
        "degraded_reason": None,
    }


@pytest.mark.asyncio
async def test_ambiguous_request_uses_llm_as_supplement() -> None:
    classifier = StubLLMClassifier(
        LLMIntentResult(
            intent=IntentType.OPS_READ,
            risk=RiskLevel.READ_ONLY,
            confidence=0.92,
            reason="用户希望查看当前规则。",
            evidence=["查看当前状态"],
        )
    )
    service = IntentRiskService(llm_classifier=classifier, llm_enabled=True)

    decision = await service.classify("防火墙规则")

    assert classifier.calls == 1
    assert decision.intent == IntentType.OPS_READ
    assert decision.route == RecommendedRoute.AGENT
    assert decision.decision_source == DecisionSource.RULE_AND_LLM
    assert decision.llm_used is True


@pytest.mark.asyncio
async def test_high_confidence_rule_result_skips_llm() -> None:
    classifier = StubLLMClassifier(
        LLMIntentResult(
            intent=IntentType.OPS_READ,
            risk=RiskLevel.READ_ONLY,
            confidence=0.99,
            reason="不应被调用",
        )
    )
    service = IntentRiskService(llm_classifier=classifier, llm_enabled=True)

    decision = await service.classify("请新增一条 TCP 443 规则并提交")

    assert classifier.calls == 0
    assert decision.route == RecommendedRoute.CONFIRM
    assert decision.decision_source == DecisionSource.RULE


@pytest.mark.asyncio
async def test_llm_write_decision_requires_confirmation() -> None:
    classifier = StubLLMClassifier(
        LLMIntentResult(
            intent=IntentType.CONFIG_CHANGE,
            risk=RiskLevel.WRITE,
            confidence=0.91,
            reason="用户想调整规则。",
            evidence=["调整规则"],
        )
    )
    service = IntentRiskService(llm_classifier=classifier, llm_enabled=True)

    decision = await service.classify("防火墙规则")

    assert decision.route == RecommendedRoute.CONFIRM
    assert decision.risk == RiskLevel.WRITE
    assert decision.requires_confirmation is True


@pytest.mark.asyncio
async def test_low_confidence_llm_result_still_asks_user() -> None:
    classifier = StubLLMClassifier(
        LLMIntentResult(
            intent=IntentType.OPS_READ,
            risk=RiskLevel.READ_ONLY,
            confidence=0.6,
            reason="语义不明确。",
        )
    )
    service = IntentRiskService(
        llm_classifier=classifier,
        llm_enabled=True,
        llm_min_confidence=0.8,
    )

    decision = await service.classify("防火墙规则")

    assert decision.route == RecommendedRoute.ASK_USER
    assert decision.risk == RiskLevel.UNKNOWN


@pytest.mark.asyncio
async def test_llm_timeout_falls_back_to_ask_user() -> None:
    service = IntentRiskService(
        llm_classifier=SlowLLMClassifier(),
        llm_enabled=True,
        llm_timeout_s=0.001,
    )

    decision = await service.classify("防火墙规则")

    assert decision.route == RecommendedRoute.ASK_USER
    assert decision.decision_source == DecisionSource.FALLBACK
    assert decision.degraded_reason is not None
    assert "超过" in decision.degraded_reason


def test_rule_write_risk_cannot_be_downgraded_by_llm(service) -> None:
    rule_decision = service.classify_rules("删除 rule-003")
    llm_decision = LLMIntentResult(
        intent=IntentType.OPS_READ,
        risk=RiskLevel.READ_ONLY,
        confidence=0.99,
        reason="错误的只读判断",
    )

    decision = merge_decisions(rule_decision, llm_decision, min_confidence=0.8)

    assert decision.risk == RiskLevel.WRITE
    assert decision.route == RecommendedRoute.CONFIRM
