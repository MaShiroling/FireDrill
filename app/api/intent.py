"""意图与风险识别接口。"""

from fastapi import APIRouter

from app.models.intent import IntentClassifyRequest, IntentDecision
from app.services.intent_service import intent_risk_service

router = APIRouter()


@router.post("/intent/classify", response_model=IntentDecision)
async def classify_intent(request: IntentClassifyRequest) -> IntentDecision:
    """返回可解释的意图、风险等级和推荐路由。"""
    return await intent_risk_service.classify(request.text)
