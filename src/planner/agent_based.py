# ============================================================================
# 模块职责: Agent-based Planner — 基于 LLM Agent 的感知-规划循环
#   本模块职责:
#     1. 调用 MCP-style perception tools 获取图像分析信息
#     2. 将分析信息交给 llm/PlannerCaller 获取 Plan
#     3. 编排 perception → plan 的流程
#   LLM 调用逻辑已委托给 llm/planner_caller.py
#   向后兼容: 仍支持 api_caller=callable 构造方式
# 参考: 4KAgent (https://github.com/taco-group/4KAgent) — perception-planning-execution
#       MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — agent loop
#       AgenticIR — planner with LLM delegation
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.degradations.types import DegradationReport
from src.planner.base import BasePlanner, Plan, ToolCall
from src.tools.mcp_style import AnalysisTool, PerceptionTool, StatisticsTool

logger = logging.getLogger(__name__)


class AgentBasedPlanner(BasePlanner):
    """Agent-based planner: perception → LLM planning → execution.

    构造方式:
    1. (新) planner_caller-based:
         planner = AgentBasedPlanner(planner_caller=PlannerCaller(client))
    2. (旧) api_caller-based (向后兼容):
         planner = AgentBasedPlanner(api_caller=my_callable)
    3. 无 LLM: 自动回退到 rule-based
    """

    def __init__(
        self,
        api_caller: Any | None = None,
        planner_caller: Any | None = None,
        max_iterations: int = 1,
        max_chain: int = 3,
        tool_descriptions: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            api_caller: [兼容] 可调用对象 (messages) -> str
            planner_caller: [推荐] llm/PlannerCaller 实例
            max_iterations: 最大 perception-plan 迭代次数
            max_chain: 单次计划最大工具数
            tool_descriptions: {tool_name: description}
        """
        self.max_iterations = max_iterations
        self.max_chain = max_chain
        self.tool_descriptions = tool_descriptions or self._default_tool_descriptions()

        # LLM 调用层: 优先使用 PlannerCaller，其次包装 legacy callable
        self._planner_caller = planner_caller
        self._legacy_api_caller = api_caller

        # 如果传了 legacy api_caller 但没传 planner_caller，自动桥接
        if self._planner_caller is None and self._legacy_api_caller is not None:
            self._planner_caller = self._bridge_legacy_caller(
                self._legacy_api_caller, self.tool_descriptions, self.max_chain
            )

        # MCP-style perception tools
        self.analysis_tool = AnalysisTool()
        self.perception_tool = PerceptionTool()
        self.statistics_tool = StatisticsTool()

    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        """根据退化报告生成计划 (兼容 BasePlanner 接口)。"""
        image = kwargs.get("image")
        if image is not None and self._planner_caller is not None:
            return self.plan_with_perception(image)
        return self._fallback_rule_plan(report)

    def plan_with_perception(self, image: np.ndarray) -> Plan:
        """完整 agent 流程: 感知 → LLM 规划。"""
        perceptions = self.collect_perceptions(image)
        logger.info(
            "Perception complete: %s",
            perceptions["analysis"].get("primary_degradation", "unknown"),
        )

        if self._planner_caller is not None:
            return self._planner_caller.call(
                analysis=perceptions["analysis"],
                perception=perceptions["perception"],
                statistics=perceptions["statistics"],
            )

        # 无 LLM: 从 analysis 直接生成简单计划
        return self._analysis_to_plan(perceptions["analysis"])

    def collect_perceptions(self, image: np.ndarray) -> dict[str, Any]:
        """仅执行感知步骤，返回所有工具的结果。"""
        return {
            "analysis": self.analysis_tool(image),
            "perception": self.perception_tool(image),
            "statistics": self.statistics_tool(image),
        }

    # ------------------------------------------------------------------
    # 回退 / 辅助
    # ------------------------------------------------------------------

    def _fallback_rule_plan(self, report: DegradationReport) -> Plan:
        from src.planner.rule_planner import RuleBasedPlanner
        return RuleBasedPlanner(max_chain=self.max_chain).plan(report)

    @staticmethod
    def _analysis_to_plan(analysis: dict) -> Plan:
        """从分析结果直接生成简单计划 (无 LLM 时的降级策略)。"""
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

    @staticmethod
    def _default_tool_descriptions() -> dict[str, str]:
        return {
            "denoise_nlm": "Non-Local Means denoising for mild to moderate noise",
            "denoise_gaussian": "Gaussian filter denoising, fast but may blur edges",
            "sharpen_usm": "Unsharp mask sharpening for blurred images",
            "histogram_clahe": "Adaptive histogram equalization for low contrast",
            "ldct_denoiser": "Deep learning Low-Dose CT denoiser (severe noise)",
            "mar_rise": "Metal Artifact Reduction for CT with metal implants",
            "sr_ct": "CT Super Resolution for low-resolution images",
        }

    # ------------------------------------------------------------------
    # Legacy api_caller 桥接
    # ------------------------------------------------------------------

    @staticmethod
    def _bridge_legacy_caller(
        api_caller: Any,
        tool_descriptions: dict[str, str],
        max_steps: int,
    ) -> Any:
        """将旧版 api_caller (callable) 桥接为 PlannerCaller 兼容对象。"""
        from llm.api_client import BaseLLMClient, LLMResponse
        from llm.planner_caller import PlannerCaller

        class _LegacyBridge(BaseLLMClient):
            """将 (messages) -> str 的 callable 包装为 BaseLLMClient。"""
            def __init__(self, fn: Any) -> None:
                self._fn = fn

            def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
                text = self._fn(messages)
                return LLMResponse(text=text)

        bridge_client = _LegacyBridge(api_caller)
        return PlannerCaller(
            llm_client=bridge_client,
            tool_descriptions=tool_descriptions,
            max_steps=max_steps,
        )
