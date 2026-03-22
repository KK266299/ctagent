# ============================================================================
# 模块职责: Prompt 构建器 — 统一管理所有 LLM prompt 模板
#   包含:
#     1. Planning prompt — 基础规划 (PlannerCaller 使用)
#     2. Replan prompt  — judge 失败后重新规划
#     3. Guided prompt  — API-guided planner 使用, 含工具目录/参数范围/历史轨迹
#     4. Diagnosis prompt — 诊断 (skeleton)
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
- For metal artifacts: apply mar_rise FIRST, then follow with gentle denoising if needed.
  Metal artifacts manifest as bright/dark streaks, cupping, and photon starvation shadows
  caused by high-attenuation implants. Do NOT apply aggressive denoising before MAR.

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


# ---- Guided planner prompts (API-guided planning + replanning) ----

TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "tool_name": "denoise_tv",
        "category": "denoise",
        "description": "Total Variation denoising: strong edge-preserving, best for moderate-severe noise.",
        "params": {"weight": {"type": "float", "range": [0.02, 0.20], "default": 0.1}},
    },
    {
        "tool_name": "denoise_bilateral",
        "category": "denoise",
        "description": "Bilateral filter: edge-preserving, best for mild-moderate noise.",
        "params": {
            "sigma_color": {"type": "float", "range": [0.02, 0.10], "default": 0.05},
            "sigma_spatial": {"type": "int", "range": [2, 10], "default": 5},
        },
    },
    {
        "tool_name": "denoise_nlm",
        "category": "denoise",
        "description": "Non-Local Means: structure-aware via self-similarity, good for textured regions.",
        "params": {"h": {"type": "float", "range": [0.01, 0.30], "default": "auto"}},
    },
    {
        "tool_name": "denoise_gaussian",
        "category": "denoise",
        "description": "Gaussian filter: fast but blurs edges. Use only when speed matters.",
        "params": {"sigma": {"type": "float", "range": [0.5, 2.0], "default": 1.0}},
    },
    {
        "tool_name": "denoise_wiener",
        "category": "denoise",
        "description": "Wiener filter: frequency-domain, optimal for uniform Gaussian white noise.",
        "params": {"mysize": {"type": "int", "range": [3, 7], "default": 5}},
    },
    {
        "tool_name": "sharpen_usm",
        "category": "sharpen",
        "description": "Unsharp mask sharpening: recovers edges after denoising.",
        "params": {
            "radius": {"type": "float", "range": [1.0, 3.0], "default": 2.0},
            "amount": {"type": "float", "range": [0.3, 2.0], "default": 1.5},
        },
    },
    {
        "tool_name": "histogram_clahe",
        "category": "contrast",
        "description": "CLAHE adaptive histogram equalization. WARNING: may hurt safety score.",
        "params": {"clip_limit": {"type": "float", "range": [0.005, 0.05], "default": 0.02}},
    },
    {
        "tool_name": "mar_rise",
        "category": "mar",
        "description": (
            "Metal Artifact Reduction (MAR) for CT images with metal implants. "
            "Uses deep learning to disentangle metal artifacts from underlying anatomy. "
            "Effective for beam hardening streaks, photon starvation shadows, and cupping "
            "artifacts caused by high-attenuation metal objects (e.g., titanium implants, "
            "dental fillings, surgical clips). Should be applied BEFORE general denoising."
        ),
        "params": {
            "metal_threshold": {
                "type": "float",
                "range": [0.5, 0.95],
                "default": 0.8,
                "description": "Threshold for metal segmentation in normalized image",
            },
            "use_li_prior": {
                "type": "bool",
                "range": [True, False],
                "default": True,
                "description": "Use Linear Interpolation (LI) prior as auxiliary input",
            },
        },
    },
]

GUIDED_SYSTEM_PROMPT = """\
You are a CT image restoration planning agent in a closed-loop system.
Your job is to select the best restoration tool(s) and parameters,
considering quality score, safety score, and downstream diagnostic accuracy.

## Available Tools
{tool_catalog}

## Scoring System
- quality_score [0-1]: measures noise reduction + cleanliness + structure preservation. Pass >= 0.3.
- safety_score [0-1]: measures structural integrity vs original (mean shift, SSIM, edge preservation). Pass >= 0.6.
- aggregate_score = min(quality, safety). Both must pass for overall PASS.
- If safety < quality: the image was over-modified → use gentler params.
- If quality < safety: denoising was insufficient → use stronger params or a different tool.

## Metal Artifact Handling
When metal artifacts are detected (artifact_metal degradation type):
- Metal artifacts arise from beam hardening, photon starvation, and scatter caused by
  high-attenuation metal objects (implants, dental fillings, surgical hardware).
- Symptoms: bright/dark streaks radiating from metal, cupping artifacts, shadow zones
  near metal boundaries, and elevated noise in metal-adjacent regions.
- Apply mar_rise FIRST, before any denoising or sharpening tools. MAR must operate
  on the artifact-corrupted image directly to correctly identify metal regions.
- After MAR, residual noise may remain — follow up with a gentle denoiser (e.g.,
  denoise_bilateral or denoise_nlm with conservative params).
- Do NOT use aggressive denoising before MAR, as it may blur metal boundaries and
  degrade the metal segmentation step.
- metal_threshold param: higher values (0.85-0.95) for clear metal objects, lower
  values (0.5-0.7) for subtle or partial-volume metal regions.
- use_li_prior=True is recommended for most cases; set to False only if linear
  interpolation introduces new artifacts (e.g., near bone-metal interfaces).

## Decision Types
- "retry": select tool(s) + params and re-execute. NEVER repeat a previously failed plan.
- "stop": accept the current best result. Use when scores are not improving.
- "abstain": give up restoration entirely, return the original degraded image.

## Output Format
Respond with ONLY a JSON object, no markdown fences or explanation:
{{
  "decision": "retry" | "stop" | "abstain",
  "steps": [
    {{"tool_name": "<name>", "params": {{...}}}},
    ...
  ],
  "reason": "<one-sentence explanation>"
}}
"steps" is required when decision="retry", ignored otherwise.
"""

GUIDED_PLAN_USER_PROMPT = """\
This is iteration #{iteration} of the restoration loop.

## CT Image
An image of the degraded CT scan is attached.

## Degradation Summary
{degradation_json}

## Current Scores
{scores_json}

## Execution History
{history_json}

Select the best tool(s) and parameters. Output JSON only.
"""

GUIDED_PLAN_USER_PROMPT_NO_IMAGE = """\
This is iteration #{iteration} of the restoration loop.

## Degradation Summary
{degradation_json}

## Current Scores
{scores_json}

## Execution History
{history_json}

Select the best tool(s) and parameters. Output JSON only.
"""


def _format_tool_catalog(catalog: list[dict[str, Any]] | None = None) -> str:
    """将工具目录格式化为 prompt 中的表格。"""
    catalog = catalog or TOOL_CATALOG
    lines = []
    for t in catalog:
        params_str = ", ".join(
            f"{k}: {v['range']} (default={v['default']})"
            for k, v in t.get("params", {}).items()
        )
        lines.append(f"- **{t['tool_name']}** [{t['category']}]: {t['description']}")
        if params_str:
            lines.append(f"  params: {params_str}")
    return "\n".join(lines)


def build_guided_system_prompt(
    catalog: list[dict[str, Any]] | None = None,
) -> str:
    """构建 guided planner 的 system prompt。"""
    return GUIDED_SYSTEM_PROMPT.format(
        tool_catalog=_format_tool_catalog(catalog),
    )


def build_guided_user_prompt(
    iteration: int,
    degradation_summary: dict[str, Any],
    current_scores: dict[str, Any] | None,
    history: list[dict[str, Any]],
    with_image: bool = True,
) -> str:
    """构建 guided planner 的 user prompt。"""
    template = GUIDED_PLAN_USER_PROMPT if with_image else GUIDED_PLAN_USER_PROMPT_NO_IMAGE
    return template.format(
        iteration=iteration,
        degradation_json=json.dumps(degradation_summary, indent=2, ensure_ascii=False),
        scores_json=json.dumps(current_scores or {}, indent=2, ensure_ascii=False),
        history_json=json.dumps(history, indent=2, ensure_ascii=False) if history else "No previous attempts.",
    ).strip()
