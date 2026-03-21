# ============================================================================
# 模块职责: Prompt 构建器 — 统一管理所有 LLM prompt 模板
#   包含: planner prompt (规划用) + diagnosis prompt (诊断用)
#   当前阶段: 定义规划侧 prompt，诊断侧仍由 src/downstream/prompt_builder.py 承载
#   后续阶段: 合并诊断侧 prompt 到此处
# 参考: 4KAgent — perception / planning / reflection prompt templates
#       AgenticIR — prompt management
#       MedAgent-Pro — medical diagnosis prompt design
# ============================================================================
from __future__ import annotations

import json
from typing import Any


# ---- Planning prompts ----

PLANNING_SYSTEM_PROMPT = """\
You are a CT image restoration planning agent. You analyze CT image quality
reports and decide which restoration tools to apply, in what order.

## Available Restoration Tools
{tool_descriptions}

## Decision Criteria
- Match tool to degradation type and severity
- Order tools from most critical to refinement
- Avoid redundant or conflicting tools
- Consider downstream diagnostic requirements

## Output Format
Respond with ONLY a JSON object:
{{
  "reasoning": "<analysis of degradation and restoration strategy>",
  "steps": [
    {{"tool_name": "<tool_name>", "params": {{}}}},
    ...
  ]
}}
"""

PLANNING_USER_PROMPT = """\
Plan restoration for this CT image based on the following analysis:

## Degradation Analysis
{analysis_json}

## Quality Perception
{perception_json}

## Image Statistics
{statistics_json}

{extra_context}
Decide which tools to apply and in what order. Output your plan as JSON.
"""

# ---- Replan prompts (judge 判定失败后重新规划) ----

REPLAN_USER_PROMPT = """\
The previous restoration attempt did not meet quality requirements.

## Previous Plan
{previous_plan_json}

## Judge Verdict
{judge_verdict_json}

## Current Image Quality
{current_quality_json}

Please generate a revised restoration plan. Avoid repeating the same failing strategy.
Output your revised plan as JSON.
"""

# ---- Diagnosis prompts (skeleton, 后续从 src/downstream/prompt_builder.py 合并) ----

DIAGNOSIS_SYSTEM_PROMPT = """\
You are an expert radiologist AI assistant specialized in CT image analysis.
Provide structured diagnostic assessments based on CT images and any
supplementary analysis data provided.

Always respond in the following JSON format:
{{
  "findings": ["<finding1>", "<finding2>", ...],
  "diagnosis": "<primary diagnosis>",
  "confidence": <float 0-1>,
  "severity": "<normal|mild|moderate|severe>",
  "reasoning": "<brief reasoning>"
}}
"""


def build_planning_system_prompt(tool_descriptions: dict[str, str]) -> str:
    """构建规划 system prompt。"""
    tool_desc_text = "\n".join(f"- {name}: {desc}" for name, desc in tool_descriptions.items())
    return PLANNING_SYSTEM_PROMPT.format(tool_descriptions=tool_desc_text)


def build_planning_user_prompt(
    analysis: dict[str, Any],
    perception: dict[str, Any],
    statistics: dict[str, Any],
    extra_context: str = "",
) -> str:
    """构建规划 user prompt。"""
    return PLANNING_USER_PROMPT.format(
        analysis_json=json.dumps(analysis, indent=2, ensure_ascii=False),
        perception_json=json.dumps(perception, indent=2, ensure_ascii=False),
        statistics_json=json.dumps(statistics, indent=2, ensure_ascii=False),
        extra_context=extra_context,
    ).strip()


def build_replan_user_prompt(
    previous_plan: dict[str, Any],
    judge_verdict: dict[str, Any],
    current_quality: dict[str, Any],
) -> str:
    """构建 replan user prompt (judge 判定失败后)。"""
    return REPLAN_USER_PROMPT.format(
        previous_plan_json=json.dumps(previous_plan, indent=2, ensure_ascii=False),
        judge_verdict_json=json.dumps(judge_verdict, indent=2, ensure_ascii=False),
        current_quality_json=json.dumps(current_quality, indent=2, ensure_ascii=False),
    ).strip()
