# ============================================================================
# 模块职责: Agent-based Planner — 基于 LLM Agent 的感知-规划-执行循环
#   升级 rule-based routing 为 agent 驱动:
#   1. Perception: 用 MCP-style tools 获取图像信息
#   2. Planning: LLM 根据信息决策工具调用序列
#   3. Execution: 执行工具，获取反馈，可迭代调整
# 参考: 4KAgent (https://github.com/taco-group/4KAgent) — perception-planning-execution
#       MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — agent loop
#       Earth-Agent (https://github.com/opendatalab/Earth-Agent/tree/main/agent/tools)
# ============================================================================
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from src.degradations.types import DegradationReport
from src.planner.base import BasePlanner, Plan, ToolCall
from src.tools.mcp_style import AnalysisTool, PerceptionTool, StatisticsTool

logger = logging.getLogger(__name__)


# Agent system prompt
AGENT_SYSTEM_PROMPT = """\
You are a CT image restoration planning agent. Your job is to analyze CT images
and decide which restoration tools to apply, in what order.

## Available Restoration Tools
{tool_descriptions}

## Your Workflow
1. Review the perception and analysis results provided
2. Determine the degradation types and severity
3. Plan an ordered sequence of tool calls to restore the image
4. Output your plan as a JSON array

## Output Format
Respond with ONLY a JSON object:
{{
  "reasoning": "<your analysis of the degradation and plan>",
  "steps": [
    {{"tool_name": "<name>", "params": {{}}}},
    ...
  ]
}}
"""

# Planning request template
PLANNING_REQUEST_TEMPLATE = """\
Please plan the restoration for this CT image based on the following analysis:

## Degradation Analysis
{analysis_results}

## Quality Perception
{perception_results}

## Image Statistics
{statistics_results}

Decide which tools to apply and in what order. Output your plan as JSON.
"""


class AgentBasedPlanner(BasePlanner):
    """Agent-based planner: perception → LLM planning → execution.

    与 RuleBasedPlanner 相比:
    - 使用 MCP-style tools 做感知（而非硬编码阈值）
    - 使用 LLM 做规划决策（而非 if-else 规则）
    - 支持多轮迭代（perception → plan → execute → re-perceive）
    """

    def __init__(
        self,
        api_caller: Any | None = None,
        max_iterations: int = 1,
        max_chain: int = 3,
        tool_descriptions: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            api_caller: 可调用对象 (messages: list[dict]) -> str，用于调用 LLM
                        如果为 None，回退到 rule-based 逻辑
            max_iterations: 最大 perception-plan-execute 迭代次数
            max_chain: 单次计划最大工具数
            tool_descriptions: {tool_name: description} 覆盖
        """
        self.api_caller = api_caller
        self.max_iterations = max_iterations
        self.max_chain = max_chain
        self.tool_descriptions = tool_descriptions or self._default_tool_descriptions()

        # MCP-style perception tools
        self.analysis_tool = AnalysisTool()
        self.perception_tool = PerceptionTool()
        self.statistics_tool = StatisticsTool()

    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        """根据退化报告生成计划（兼容 BasePlanner 接口）。

        如果提供了 image kwarg，则使用 agent 模式；否则回退到规则。
        """
        image = kwargs.get("image")
        if image is not None and self.api_caller is not None:
            return self.plan_with_perception(image)
        # 回退: 从 report 直接用规则
        return self._fallback_rule_plan(report)

    def plan_with_perception(self, image: np.ndarray) -> Plan:
        """完整 agent 流程: 感知 → LLM 规划。"""
        # Step 1: Perception — 收集图像信息
        analysis = self.analysis_tool(image)
        perception = self.perception_tool(image)
        statistics = self.statistics_tool(image)
        logger.info("Perception complete: %s", analysis.get("primary_degradation", "unknown"))

        # Step 2: LLM Planning
        plan = self._llm_plan(analysis, perception, statistics)
        return plan

    def collect_perceptions(self, image: np.ndarray) -> dict[str, Any]:
        """仅执行感知步骤，返回所有工具的结果（供外部使用）。"""
        return {
            "analysis": self.analysis_tool(image),
            "perception": self.perception_tool(image),
            "statistics": self.statistics_tool(image),
        }

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def _llm_plan(
        self,
        analysis: dict,
        perception: dict,
        statistics: dict,
    ) -> Plan:
        """用 LLM 生成修复计划。"""
        tool_desc_text = "\n".join(
            f"- {name}: {desc}" for name, desc in self.tool_descriptions.items()
        )
        system_prompt = AGENT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc_text)

        user_prompt = PLANNING_REQUEST_TEMPLATE.format(
            analysis_results=json.dumps(analysis, indent=2, ensure_ascii=False),
            perception_results=json.dumps(perception, indent=2, ensure_ascii=False),
            statistics_results=json.dumps(statistics, indent=2, ensure_ascii=False),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_response = self.api_caller(messages)
            return self._parse_llm_response(raw_response)
        except Exception as e:
            logger.error("LLM planning failed: %s, falling back to analysis-based rules", e)
            return self._analysis_to_plan(analysis)

    def _parse_llm_response(self, response: str) -> Plan:
        """解析 LLM 返回的 JSON 计划。"""
        import re
        # 提取 JSON
        patterns = [r"```json\s*\n?(.*?)\n?\s*```", r"(\{.*\})"]
        data = None
        for p in patterns:
            m = re.search(p, response, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Cannot parse LLM response as JSON")
                return Plan(reasoning="LLM response unparseable")

        steps = []
        for s in data.get("steps", [])[:self.max_chain]:
            steps.append(ToolCall(
                tool_name=s.get("tool_name", ""),
                params=s.get("params", {}),
            ))
        return Plan(steps=steps, reasoning=data.get("reasoning", ""))

    # ------------------------------------------------------------------
    # 回退 / 辅助
    # ------------------------------------------------------------------

    def _fallback_rule_plan(self, report: DegradationReport) -> Plan:
        """无 LLM 时的规则回退。"""
        from src.planner.rule_planner import RuleBasedPlanner
        return RuleBasedPlanner(max_chain=self.max_chain).plan(report)

    def _analysis_to_plan(self, analysis: dict) -> Plan:
        """从分析结果直接生成简单计划。"""
        primary = analysis.get("primary_degradation", "none")
        mapping = {
            "noise": "denoise_nlm",
            "blur": "sharpen_usm",
            "artifact_metal": "mar_rise",
            "low_resolution": "sr_ct",
            "low_dose": "ldct_denoiser",
        }
        tool_name = mapping.get(primary)
        if tool_name:
            return Plan(
                steps=[ToolCall(tool_name=tool_name)],
                reasoning=f"Fallback: primary degradation is {primary}",
            )
        return Plan(reasoning="No degradation detected")

    def _default_tool_descriptions(self) -> dict[str, str]:
        """默认工具描述。"""
        return {
            "denoise_nlm": "Non-Local Means denoising for mild to moderate noise",
            "denoise_gaussian": "Gaussian filter denoising, fast but may blur edges",
            "sharpen_usm": "Unsharp mask sharpening for blurred images",
            "histogram_clahe": "Adaptive histogram equalization for low contrast",
            "ldct_denoiser": "Deep learning Low-Dose CT denoiser (severe noise)",
            "mar_rise": "Metal Artifact Reduction for CT with metal implants",
            "sr_ct": "CT Super Resolution for low-resolution images",
        }
