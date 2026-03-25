# ============================================================================
# 模块职责: Planner Caller — 通过 LLM 生成修复计划
#   封装: perception 数据 → prompt 构建 → LLM 调用 → Plan 解析
#   此模块是 pipeline/planner/agent_planner.py 的 LLM 交互后端
# 参考: 4KAgent — LLM-based planning module
#       AgenticIR — planner/llm_planner.py
#       MedAgent-Pro — agent planning calls
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

from llm.api_client import BaseLLMClient
from llm.prompt_builder import (
    build_planning_system_prompt,
    build_planning_user_prompt,
    build_replan_user_prompt,
)
from llm.response_parser import parse_plan_json
from src.planner.base import Plan

logger = logging.getLogger(__name__)


class PlannerCaller:
    """LLM Planning Caller — 将感知结果发送给 LLM 并获取 Plan。

    Usage:
        client = OpenAIClient(config)
        caller = PlannerCaller(client, tool_descriptions={...})
        plan = caller.call(analysis={...}, perception={...}, statistics={...})
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tool_descriptions: dict[str, str] | None = None,
        max_steps: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.tool_descriptions = tool_descriptions or self._default_tool_descriptions()
        self.max_steps = max_steps

    def call(
        self,
        analysis: dict[str, Any],
        perception: dict[str, Any],
        statistics: dict[str, Any],
        extra_context: str = "",
    ) -> Plan:
        """发送感知结果给 LLM，获取修复计划。"""
        system_prompt = build_planning_system_prompt(self.tool_descriptions)
        user_prompt = build_planning_user_prompt(
            analysis, perception, statistics, extra_context
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.llm_client.chat(messages)
            logger.info("LLM planning response length: %d chars", len(response.text))
            return parse_plan_json(response.text, max_steps=self.max_steps)
        except Exception as e:
            logger.error("LLM planning call failed: %s", e)
            return Plan(reasoning=f"LLM call failed: {e}")

    def call_replan(
        self,
        previous_plan: dict[str, Any],
        judge_verdict: dict[str, Any],
        current_quality: dict[str, Any],
    ) -> Plan:
        """Judge 判定失败后，调用 LLM 重新规划。"""
        system_prompt = build_planning_system_prompt(self.tool_descriptions)
        user_prompt = build_replan_user_prompt(
            previous_plan, judge_verdict, current_quality
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.llm_client.chat(messages)
            return parse_plan_json(response.text, max_steps=self.max_steps)
        except Exception as e:
            logger.error("LLM replan call failed: %s", e)
            return Plan(reasoning=f"LLM replan failed: {e}")

    @staticmethod
    def _default_tool_descriptions() -> dict[str, str]:
        return {
            "clip_extreme": "Clip μ values to [0, 0.55]. MUST be first step for metal artifact images. Zero cost.",
            "denoise_tv": "Total Variation: strong edge-preserving, for moderate-severe noise",
            "denoise_wavelet": "Wavelet thresholding (BayesShrink): multi-scale, preserves fine CT structures",
            "denoise_bm3d": "BM3D denoising: best classical denoiser for PSNR",
            "denoise_bilateral": "Bilateral filter: edge-preserving, for noise and light artifacts",
            "denoise_nlm": "Non-Local Means denoising: structure-aware, for mild-moderate noise",
            "denoise_median": "Median filter: effective against impulse noise and streak artifacts",
            "denoise_gaussian": "Gaussian filter denoising: fast but blurs edges",
            "denoise_wiener": "Wiener filter: frequency-domain, optimal for uniform Gaussian noise",
            "deblur_richardson_lucy": "Richardson-Lucy deconvolution: restores PSF-blurred images",
            "sharpen_usm": "Unsharp mask sharpening for blurred images",
            "enhance_laplacian": "Laplacian edge enhancement: boosts detail, may amplify noise",
            "inpaint_biharmonic": "Biharmonic inpainting: fills small artifact/damaged regions",
            "mar_learned": "Deep learning MAR (placeholder: classical pipeline fallback)",
            "denoise_dncnn": "DnCNN deep learning denoiser: best PSNR/SSIM among all denoisers, trained on CT data. Use for moderate-severe noise.",
        }
