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
Images are in μ (linear attenuation coefficient) space, range ~[0, 0.5].

## Available Restoration Tools
{tool_descriptions}

## Decision Criteria
- Match tool to degradation type and severity
- Order: inpaint extreme pixels FIRST → denoise → sharpen/enhance LAST
- NEVER stack two denoisers of the same family (e.g. bilateral after TV)
- For metal artifact: use clip_extreme FIRST to bound range, then inpaint_biharmonic
  to smoothly fill damaged regions, then optionally denoise_tv for residual noise.
  Recommended chain: clip_extreme → inpaint_biharmonic → denoise_tv
- clip_extreme alone hurts SSIM (creates flat regions). Always pair with inpaint_biharmonic.
- PREFER denoise_dncnn over classical denoisers (TV, bilateral, NLM) — it has better PSNR/SSIM.
  Use denoise_dncnn as the primary denoiser when noise is moderate or severe.
- CRITICAL: Do NOT use histogram_clahe on μ-space CT images — destroys SSIM
- CRITICAL: Do NOT use mar_threshold_replace — destroys bone structure
- Maximum 3 tools per plan. Fewer is better if effective.

## CT Artifact-Specific Rules
- Ring artifact: use ring_removal_polar (mild) or ring_removal_wavelet (moderate/severe), optionally followed by denoise_bilateral
- Motion artifact: use motion_correction_tv (moderate/severe) or motion_correction_wiener (mild), optionally followed by denoise_wavelet
- Beam hardening: use bhc_flatfield (mild/moderate) or bhc_polynomial (severe), optionally followed by denoise_bilateral
- Scatter: use scatter_correction_detrend, optionally followed by scatter_correction_clahe for severe cases
- Truncation: use truncation_correction_extrapolate, optionally followed by truncation_correction_tv for severe cases
- If ONLY mild generic degradation (noise, blur) is detected with NO specific artifacts, output an EMPTY steps list — do NOT apply tools (do-no-harm principle)
- When multiple artifact types coexist, prioritize the most severe one first

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

# ---- CQ500 Diagnosis Classification Prompt ----

CQ500_DIAGNOSIS_SYSTEM_PROMPT = """\
You are an expert neuroradiologist. You will be shown axial slice(s) from a \
non-contrast head CT scan. Your task is to classify the presence or absence \
of the following findings.

## Target Labels (binary: 0 = absent, 1 = present)
- ICH: Intracranial Hemorrhage (any type)
- IPH: Intraparenchymal Hemorrhage
- IVH: Intraventricular Hemorrhage
- SDH: Subdural Hemorrhage
- EDH: Epidural Hemorrhage
- SAH: Subarachnoid Hemorrhage
- Fracture: Any skull fracture
- CalvarialFracture: Calvarial fracture specifically
- MassEffect: Mass effect present
- MidlineShift: Midline shift present

## Important
- Examine each slice carefully for ALL findings above.
- ICH should be 1 if ANY hemorrhage subtype is present.
- Provide a confidence score (0.0 to 1.0) for each label.

## Output Format
Respond with ONLY a valid JSON object, no extra text:
{{
  "predictions": {{
    "ICH": 0 or 1,
    "IPH": 0 or 1,
    "IVH": 0 or 1,
    "SDH": 0 or 1,
    "EDH": 0 or 1,
    "SAH": 0 or 1,
    "Fracture": 0 or 1,
    "CalvarialFracture": 0 or 1,
    "MassEffect": 0 or 1,
    "MidlineShift": 0 or 1
  }},
  "confidence": {{
    "ICH": <float 0-1>,
    "IPH": <float 0-1>,
    "IVH": <float 0-1>,
    "SDH": <float 0-1>,
    "EDH": <float 0-1>,
    "SAH": <float 0-1>,
    "Fracture": <float 0-1>,
    "CalvarialFracture": <float 0-1>,
    "MassEffect": <float 0-1>,
    "MidlineShift": <float 0-1>
  }},
  "reasoning": "<brief clinical reasoning>"
}}
"""


def build_cq500_user_prompt(n_slices: int) -> str:
    """构建 CQ500 诊断分类的 user prompt 文本部分。"""
    if n_slices == 1:
        return (
            "Above is 1 axial slice from a non-contrast head CT scan. "
            "Classify the presence of each finding listed in the system prompt."
        )
    return (
        f"Above are {n_slices} axial slices from the same non-contrast head CT scan. "
        "Examine all slices together and classify the presence of each finding "
        "listed in the system prompt."
    )


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
    # ---- Preprocess (always first) ----
    {
        "tool_name": "clip_extreme",
        "category": "preprocess",
        "suitable_for": ["artifact", "noise"],
        "cost": "cheap",
        "safety": "safe",
        "description": "Clip μ values to [0, max_mu]. MUST be first step for metal artifact images.",
        "params": {
            "low": {"type": "float", "range": [0.0, 0.0], "default": 0.0},
            "high": {"type": "float", "range": [0.3, 1.0], "default": 0.5},
        },
    },
    # ---- MAR ----
    # mar_threshold_replace omitted: too risky for μ-space, destroys bone tissue
    # ---- Denoise ----
    {
        "tool_name": "denoise_bm3d",
        "category": "denoise",
        "suitable_for": ["noise"],
        "cost": "expensive",
        "safety": "safe",
        "description": "BM3D denoising: state-of-the-art classical denoiser, best PSNR among classical methods.",
        "params": {"sigma_psd": {"type": "float", "range": [0.001, 0.1], "default": "auto"}},
    },
    {
        "tool_name": "denoise_tv",
        "category": "denoise",
        "suitable_for": ["noise", "artifact"],
        "cost": "medium",
        "safety": "safe",
        "description": "Total Variation denoising: strong edge-preserving, best for moderate-severe noise.",
        "params": {"weight": {"type": "float", "range": [0.02, 0.20], "default": 0.1}},
    },
    {
        "tool_name": "denoise_bilateral",
        "category": "denoise",
        "suitable_for": ["noise", "artifact"],
        "cost": "medium",
        "safety": "safe",
        "description": "Bilateral filter: edge-preserving, best for mild-moderate noise and light artifacts.",
        "params": {
            "sigma_color": {"type": "float", "range": [0.02, 0.10], "default": 0.05},
            "sigma_spatial": {"type": "int", "range": [2, 10], "default": 5},
        },
    },
    {
        "tool_name": "denoise_nlm",
        "category": "denoise",
        "suitable_for": ["noise"],
        "cost": "expensive",
        "safety": "safe",
        "description": "Non-Local Means: structure-aware via self-similarity, good for textured regions.",
        "params": {"h": {"type": "float", "range": [0.01, 0.30], "default": "auto"}},
    },
    {
        "tool_name": "denoise_wavelet",
        "category": "denoise",
        "suitable_for": ["noise", "artifact"],
        "cost": "medium",
        "safety": "safe",
        "description": "Wavelet thresholding (BayesShrink): multi-scale, preserves fine structures well.",
        "params": {
            "wavelet": {"type": "str", "options": ["db1", "db4", "db8", "sym4"], "default": "db4"},
            "method": {"type": "str", "options": ["BayesShrink", "VisuShrink"], "default": "BayesShrink"},
        },
    },
    {
        "tool_name": "denoise_median",
        "category": "denoise",
        "suitable_for": ["noise", "artifact"],
        "cost": "cheap",
        "safety": "safe",
        "description": "Median filter: effective against impulse noise and streak artifacts.",
        "params": {"size": {"type": "int", "range": [3, 9], "default": 3}},
    },
    {
        "tool_name": "denoise_gaussian",
        "category": "denoise",
        "suitable_for": ["noise"],
        "cost": "cheap",
        "safety": "moderate",
        "description": "Gaussian filter: fast but blurs edges. Use only when speed matters.",
        "params": {"sigma": {"type": "float", "range": [0.5, 2.0], "default": 1.0}},
    },
    {
        "tool_name": "denoise_wiener",
        "category": "denoise",
        "suitable_for": ["noise"],
        "cost": "cheap",
        "safety": "moderate",
        "description": "Wiener filter: frequency-domain, optimal for uniform Gaussian white noise.",
        "params": {"mysize": {"type": "int", "range": [3, 7], "default": 5}},
    },
    # ---- Deblur ----
    {
        "tool_name": "deblur_richardson_lucy",
        "category": "deblur",
        "suitable_for": ["blur"],
        "cost": "medium",
        "safety": "moderate",
        "description": "Richardson-Lucy iterative deconvolution: restores PSF-blurred images.",
        "params": {
            "iterations": {"type": "int", "range": [5, 50], "default": 15},
            "psf_size": {"type": "int", "range": [3, 11], "default": 5},
            "psf_sigma": {"type": "float", "range": [0.5, 3.0], "default": 1.0},
        },
    },
    # ---- Sharpen ----
    {
        "tool_name": "sharpen_usm",
        "category": "sharpen",
        "suitable_for": ["blur", "low_resolution"],
        "cost": "cheap",
        "safety": "moderate",
        "description": "Unsharp mask sharpening: recovers edges after denoising.",
        "params": {
            "radius": {"type": "float", "range": [1.0, 3.0], "default": 2.0},
            "amount": {"type": "float", "range": [0.3, 2.0], "default": 1.5},
        },
    },
    {
        "tool_name": "enhance_laplacian",
        "category": "sharpen",
        "suitable_for": ["blur", "low_resolution"],
        "cost": "cheap",
        "safety": "risky",
        "description": "Laplacian edge enhancement: boosts high-frequency detail. May amplify noise.",
        "params": {"alpha": {"type": "float", "range": [0.1, 1.0], "default": 0.3}},
    },
    # ---- Contrast ----
    {
        "tool_name": "histogram_clahe",
        "category": "contrast",
        "suitable_for": ["contrast", "low_resolution"],
        "cost": "cheap",
        "safety": "moderate",
        "description": "CLAHE adaptive histogram equalization. WARNING: may change intensity distribution.",
        "params": {"clip_limit": {"type": "float", "range": [0.005, 0.05], "default": 0.02}},
    },
    {
        "tool_name": "histogram_match",
        "category": "contrast",
        "suitable_for": ["contrast", "artifact"],
        "cost": "cheap",
        "safety": "safe",
        "description": "Histogram matching: align intensity to a reference image. Needs 'reference' param.",
        "params": {"reference": {"type": "ndarray", "required": True}},
    },
    # ---- Deep Learning Denoise ----
    {
        "tool_name": "denoise_dncnn",
        "category": "denoise",
        "suitable_for": ["noise", "artifact"],
        "cost": "medium",
        "safety": "safe",
        "description": "DnCNN deep learning denoiser: residual CNN trained on CT data. Best PSNR/SSIM among all denoisers. Prefer over classical denoisers when available.",
        "params": {
            "blend": {"type": "float", "range": [0.5, 1.0], "default": 1.0},
        },
    },
    # ---- Inpaint ----
    {
        "tool_name": "inpaint_biharmonic",
        "category": "inpaint",
        "suitable_for": ["artifact"],
        "cost": "expensive",
        "safety": "moderate",
        "description": "Biharmonic inpainting: fills small artifact regions. Auto-detects extreme pixels if no mask.",
        "params": {"extreme_percentile": {"type": "float", "range": [95.0, 99.9], "default": 99.5}},
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
    """将工具目录格式化为 prompt 中的表格，含 cost/safety/suitable_for 信息。"""
    catalog = catalog or TOOL_CATALOG
    lines = []
    for t in catalog:
        params_str = ", ".join(
            f"{k}: {v.get('range', v.get('options', '?'))} (default={v.get('default', '?')})"
            for k, v in t.get("params", {}).items()
            if k != "reference"
        )
        suitable = ", ".join(t.get("suitable_for", []))
        cost = t.get("cost", "?")
        safety = t.get("safety", "?")
        lines.append(
            f"- **{t['tool_name']}** [{t['category']}] "
            f"(for: {suitable} | cost: {cost} | safety: {safety}): {t['description']}"
        )
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
