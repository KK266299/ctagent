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
        detector_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            api_caller: [兼容] 可调用对象 (messages) -> str
            planner_caller: [推荐] llm/PlannerCaller 实例
            max_iterations: 最大 perception-plan 迭代次数
            max_chain: 单次计划最大工具数
            tool_descriptions: {tool_name: description}
            detector_config: 传递给 AnalysisTool 内部 DegradationDetector 的阈值配置
        """
        self.max_iterations = max_iterations
        self.max_chain = max_chain
        self.tool_descriptions = tool_descriptions or self._default_tool_descriptions()

        self._planner_caller = planner_caller
        self._legacy_api_caller = api_caller

        if self._planner_caller is None and self._legacy_api_caller is not None:
            self._planner_caller = self._bridge_legacy_caller(
                self._legacy_api_caller, self.tool_descriptions, self.max_chain
            )

        # MCP-style perception tools — 使用校准后的 detector config
        from src.degradations.detector import DegradationDetector
        detector = DegradationDetector(detector_config) if detector_config else None
        self.analysis_tool = AnalysisTool(detector=detector)
        self.perception_tool = PerceptionTool()
        self.statistics_tool = StatisticsTool()

    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        """根据退化报告生成计划 (兼容 BasePlanner 接口)。"""
        image = kwargs.get("image")
        extra_context = kwargs.get("extra_context", "")
        if image is not None and self._planner_caller is not None:
            return self.plan_with_perception(image, extra_context=extra_context)
        return self._fallback_rule_plan(report)

    def plan_with_perception(self, image: np.ndarray, extra_context: str = "") -> Plan:
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
                extra_context=extra_context,
            )

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
        mapping: dict[str, list[str]] = {
            "noise": ["denoise_wavelet"],
            "blur": ["deblur_richardson_lucy", "sharpen_usm"],
            "artifact_metal": ["denoise_median", "denoise_bilateral"],
            "low_resolution": ["sharpen_usm", "enhance_laplacian"],
            "low_dose": ["denoise_wavelet", "denoise_bilateral"],
            "contrast": ["histogram_clahe"],
        }
        tools = mapping.get(primary, [])
        if tools:
            return Plan(
                steps=[ToolCall(tool_name=t) for t in tools],
                reasoning=f"Fallback: primary degradation is {primary}",
            )
        return Plan(reasoning="No degradation detected")

    @staticmethod
    def _default_tool_descriptions() -> dict[str, str]:
        return {
            "denoise_nlm": "Non-Local Means denoising: structure-aware, for mild-moderate noise",
            "denoise_gaussian": "Gaussian filter denoising: fast but blurs edges",
            "denoise_bilateral": "Bilateral filter: edge-preserving, for noise and light artifacts",
            "denoise_tv": "Total Variation: strong edge-preserving, for moderate-severe noise",
            "denoise_wavelet": "Wavelet thresholding (BayesShrink): multi-scale, preserves fine CT structures",
            "denoise_median": "Median filter: effective against impulse noise and streak artifacts",
            "denoise_wiener": "Wiener filter: frequency-domain, optimal for uniform Gaussian noise",
            "deblur_richardson_lucy": "Richardson-Lucy deconvolution: restores PSF-blurred images",
            "sharpen_usm": "Unsharp mask sharpening for blurred images",
            "enhance_laplacian": "Laplacian edge enhancement: boosts detail, may amplify noise",
            "histogram_clahe": "CLAHE adaptive histogram equalization for low contrast",
            "histogram_match": "Histogram matching: align intensity to a reference image",
            "inpaint_biharmonic": "Biharmonic inpainting: fills small artifact/damaged regions",
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
