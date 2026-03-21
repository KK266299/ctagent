# ============================================================================
# 模块职责: LLM 响应解析 — 将 LLM 原始文本解析为结构化数据
#   专注于规划侧 (Plan JSON 解析)
#   诊断侧解析当前仍由 src/downstream/response_parser.py 承载
# 参考: 4KAgent — response parsing & validation
#       AgenticIR — structured output parsing
# ============================================================================
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.planner.base import Plan, ToolCall

logger = logging.getLogger(__name__)


def parse_plan_json(raw_text: str, max_steps: int = 5) -> Plan:
    """从 LLM 原始文本中解析出 Plan。

    解析策略 (优先级从高到低):
    1. 提取 ```json ... ``` 代码块
    2. 提取最外层 { ... } JSON 对象
    3. 直接尝试 json.loads
    """
    data = _extract_json_object(raw_text)
    if data is None:
        logger.warning("Cannot parse LLM response as JSON, returning empty plan")
        return Plan(reasoning="LLM response unparseable")

    steps: list[ToolCall] = []
    for s in data.get("steps", [])[:max_steps]:
        tool_name = s.get("tool_name", "")
        if tool_name:
            steps.append(ToolCall(tool_name=tool_name, params=s.get("params", {})))

    return Plan(steps=steps, reasoning=data.get("reasoning", ""))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从文本中提取 JSON 对象。"""
    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
        r"(\{[\s\S]*\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
