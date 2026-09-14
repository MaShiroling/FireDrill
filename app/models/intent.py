"""意图与风险识别的数据模型。"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IntentType(StrEnum):
    KNOWLEDGE_QA = "knowledge_qa"
    OPS_READ = "ops_read"
    CONFIG_CHANGE = "config_change"
    AMBIGUOUS = "ambiguous"


class RiskLevel(StrEnum):
    READ_ONLY = "read_only"
    WRITE = "write"
    UNKNOWN = "unknown"


class RecommendedRoute(StrEnum):
    RAG = "rag"
    AGENT = "agent"
    CONFIRM = "confirm"
    ASK_USER = "ask_user"


class DecisionSource(StrEnum):
    RULE = "rule"
    RULE_AND_LLM = "rule+llm"
    FALLBACK = "fallback"


class IntentClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text 不能为空")
        return value


class IntentDecision(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    intent: IntentType
    risk: RiskLevel
    confidence: float = Field(..., ge=0, le=1)
    route: RecommendedRoute
    requires_confirmation: bool
    reason: str
    matched_signals: list[str] = Field(default_factory=list)
    decision_source: DecisionSource = DecisionSource.RULE
    llm_used: bool = False
    rule_decision: IntentType | None = None
    llm_decision: IntentType | None = None
    degraded_reason: str | None = None


class LLMIntentResult(BaseModel):
    """LLM 只负责语义分类，最终路由仍由确定性代码裁决。"""

    intent: IntentType
    risk: RiskLevel
    confidence: float = Field(..., ge=0, le=1)
    reason: str
    evidence: list[str] = Field(default_factory=list)
