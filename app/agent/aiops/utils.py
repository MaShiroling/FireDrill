"""AIOps Agent 通用工具函数。"""

from typing import Any


def format_tools_description(tools: list[Any]) -> str:
    """格式化工具列表为描述文本"""
    tool_descriptions = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_descriptions)


def format_execution_context(
    original_input: str,
    past_steps: list[tuple[Any, Any]],
    *,
    max_steps: int = 3,
    max_result_chars: int = 1200,
) -> str:
    """Preserve identifiers from the original task and recent real tool results."""
    sections = [f"原始用户任务（仅用于保留目标和标识符）：\n{original_input}"]
    if past_steps:
        history = []
        for step, result in past_steps[-max_steps:]:
            result_text = str(result)
            if len(result_text) > max_result_chars:
                result_text = result_text[:max_result_chars] + "...[截断]"
            history.append(f"- 已执行：{step}\n  实际结果：{result_text}")
        sections.append("最近执行历史（只复用其中真实返回的值）：\n" + "\n".join(history))
    return "\n\n".join(sections)
