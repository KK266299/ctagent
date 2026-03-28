# ============================================================================
# 模块职责: LLM 响应解析 — 将 LLM 原始文本解析为结构化数据
#   1. parse_plan_json     — 基础 Plan 解析 (PlannerCaller 使用)
#   2. parse_guided_decision — API-guided planner 解析 (含 decision + 参数校验)
# 参考: 4KAgent — response parsing & validation
#       AgenticIR — structured output parsing
# ============================================================================
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.planner.base import Plan, ToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 参数合法范围 — 超出范围的值会被 clip 而非拒绝
# ---------------------------------------------------------------------------

PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "denoise_tv": {"weight": (0.02, 0.20)},
    "denoise_bilateral": {"sigma_color": (0.02, 0.10), "sigma_spatial": (2, 10)},
    "denoise_nlm": {"h": (0.01, 0.30)},
    "denoise_gaussian": {"sigma": (0.3, 3.0)},
    "denoise_wiener": {"mysize": (3, 11)},
    "sharpen_usm": {"radius": (0.5, 5.0), "amount": (0.1, 3.0)},
    "histogram_clahe": {"clip_limit": (0.005, 0.05)},
    "ring_removal_polar": {"sigma": (1.0, 5.0)},
    "ring_removal_wavelet": {"level": (2, 6), "sigma": (1.0, 5.0)},
    "motion_correction_tv": {"weight": (0.05, 0.3)},
    "motion_correction_wiener": {"noise_var": (0.0001, 0.01)},
    "bhc_flatfield": {"strength": (0.1, 1.0)},
    "bhc_polynomial": {"strength": (0.5, 2.0)},
    "scatter_correction_detrend": {"scatter_fraction": (0.1, 0.5)},
    "truncation_correction_extrapolate": {"margin": (5, 40)},
    "truncation_correction_tv": {"weight": (0.05, 0.2)},
}

_EXTRA_VALID_TOOLS: set[str] = {
    "clip_extreme", "denoise_median", "denoise_wavelet", "denoise_bm3d",
    "denoise_dncnn", "deblur_richardson_lucy", "enhance_laplacian",
    "inpaint_biharmonic", "scatter_correction_clahe",
}

VALID_TOOL_NAMES: set[str] = set(PARAM_RANGES.keys()) | _EXTRA_VALID_TOOLS


def clip_params(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """将参数 clip 到合法范围内, 未知参数原样保留。"""
    ranges = PARAM_RANGES.get(tool_name, {})
    clipped = dict(params)
    for key, (lo, hi) in ranges.items():
        if key in clipped:
            try:
                val = float(clipped[key])
                clipped[key] = round(max(lo, min(hi, val)), 6)
            except (TypeError, ValueError):
                pass
    return clipped


# ---------------------------------------------------------------------------
# 基础 Plan 解析 (PlannerCaller 使用)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Guided Decision 解析 (APIGuidedPlanner 使用)
# ---------------------------------------------------------------------------

@dataclass
class GuidedDecision:
    """API-guided planner 解析后的决策。"""
    decision: str  # "retry" | "stop" | "abstain"
    plan: Plan | None = None
    reason: str = ""
    raw: dict[str, Any] | None = None


def parse_guided_decision(
    raw_text: str,
    max_steps: int = 5,
    validate_tools: bool = True,
) -> GuidedDecision:
    """解析 API-guided planner 的 LLM 响应。

    兼容两种 steps 格式:
      1. {"steps": [{"tool_name": ..., "params": {...}}, ...]}
      2. {"tool_name": ..., "params": {...}}  (单工具简写)

    对 tool_name 做合法性校验, 对 params 做范围 clip。
    """
    data = _extract_json_object(raw_text)
    if data is None:
        raise ValueError(f"Cannot parse LLM response as JSON: {raw_text[:200]}")

    decision = data.get("decision", "stop")
    if decision not in ("retry", "stop", "abstain"):
        logger.warning("Unknown decision '%s', defaulting to 'stop'", decision)
        decision = "stop"

    reason = data.get("reason", "")

    if decision != "retry":
        return GuidedDecision(decision=decision, reason=reason, raw=data)

    raw_steps = data.get("steps")
    if not raw_steps:
        tool_name = data.get("tool_name", "")
        params = data.get("params", {})
        if tool_name:
            raw_steps = [{"tool_name": tool_name, "params": params}]
        else:
            logger.warning("decision='retry' but no tool specified, defaulting to 'stop'")
            return GuidedDecision(decision="stop", reason="No tool in retry response", raw=data)

    steps: list[ToolCall] = []
    for s in raw_steps[:max_steps]:
        name = s.get("tool_name", "")
        if not name:
            continue
        if validate_tools and name not in VALID_TOOL_NAMES:
            logger.warning("Unknown tool '%s' from LLM, skipping", name)
            continue
        params = clip_params(name, s.get("params", {}))
        steps.append(ToolCall(tool_name=name, params=params))

    if not steps:
        logger.warning("All tools from LLM response were invalid, defaulting to 'stop'")
        return GuidedDecision(decision="stop", reason="All proposed tools invalid", raw=data)

    plan = Plan(steps=steps, reasoning=reason)
    return GuidedDecision(decision="retry", plan=plan, reason=reason, raw=data)


# ---------------------------------------------------------------------------
# CQ500 诊断分类结果解析
# ---------------------------------------------------------------------------

CQ500_LABEL_NAMES = [
    "ICH", "IPH", "IVH", "SDH", "EDH", "SAH",
    "Fracture", "CalvarialFracture", "MassEffect", "MidlineShift",
]


@dataclass
class CQ500DiagnosisResult:
    """CQ500 诊断分类解析结果。"""
    predictions: dict[str, int] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    parse_success: bool = True
    raw: dict[str, Any] | None = None


def parse_cq500_diagnosis(raw_text: str) -> CQ500DiagnosisResult:
    """从 LLM 响应解析 CQ500 多标签分类结果。"""
    data = _extract_json_object(raw_text)
    if data is None:
        logger.warning("Cannot parse CQ500 diagnosis response as JSON")
        return CQ500DiagnosisResult(
            predictions={lbl: 0 for lbl in CQ500_LABEL_NAMES},
            confidence={lbl: 0.0 for lbl in CQ500_LABEL_NAMES},
            parse_success=False,
        )

    preds_raw = data.get("predictions", data)
    conf_raw = data.get("confidence", {})
    reasoning = data.get("reasoning", "")

    predictions = {}
    confidence = {}
    for lbl in CQ500_LABEL_NAMES:
        val = preds_raw.get(lbl, 0)
        try:
            predictions[lbl] = 1 if int(val) >= 1 else 0
        except (TypeError, ValueError):
            predictions[lbl] = 1 if str(val).lower() in ("1", "true", "yes") else 0

        c_val = conf_raw.get(lbl, 0.5)
        try:
            confidence[lbl] = float(c_val)
        except (TypeError, ValueError):
            confidence[lbl] = 0.5

    return CQ500DiagnosisResult(
        predictions=predictions,
        confidence=confidence,
        reasoning=reasoning,
        parse_success=True,
        raw=data,
    )


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------

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
