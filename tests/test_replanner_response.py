from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.agent.aiops.replanner import _extract_text_content, _generate_response


def test_extract_text_content_supports_text_blocks() -> None:
    response = AIMessage(
        content=[
            {"type": "text", "text": "第一段"},
            {"type": "text", "text": "第二段"},
        ]
    )

    assert _extract_text_content(response) == "第一段\n第二段"


@pytest.mark.asyncio
async def test_generate_response_retries_an_empty_model_result() -> None:
    responses = iter([AIMessage(content=""), AIMessage(content="任务已完成，验证通过。")])
    llm = RunnableLambda(lambda _: next(responses))
    state = {
        "input": "放通 SSH 流量",
        "past_steps": [("提交并验证", "提交成功，流量验证允许")],
    }

    result = await _generate_response(state, llm)  # type: ignore[arg-type]

    assert result == {"response": "任务已完成，验证通过。"}


@pytest.mark.asyncio
async def test_generate_response_falls_back_after_two_empty_results() -> None:
    llm = RunnableLambda(lambda _: AIMessage(content=""))
    state = {
        "input": "放通 SSH 流量",
        "past_steps": [("提交并验证", "提交成功，流量验证允许")],
    }

    result = await _generate_response(state, llm)  # type: ignore[arg-type]

    assert "任务执行结果" in result["response"]
    assert "提交成功，流量验证允许" in result["response"]
