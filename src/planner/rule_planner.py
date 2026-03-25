# ============================================================================
# 模块职责: 基于规则的 Planner — 第一版 MVP 使用硬编码规则
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — rule-based planning
# ============================================================================
from __future__ import annotations

from typing import Any

from src.degradations.types import DegradationReport, DegradationType, Severity
from src.planner.base import BasePlanner, Plan, ToolCall


# 默认规则：(退化类型, 严重度) -> 工具列表
# 策略: artifact 优先处理极端值, 再 denoise, 最后 sharpen
DEFAULT_RULES: dict[tuple[DegradationType, Severity], list] = {
    # ---- Metal artifact (clip → DnCNN as primary restorer) ----
    (DegradationType.ARTIFACT_METAL, Severity.MILD): [
        "clip_extreme",
        "denoise_dncnn",
    ],
    (DegradationType.ARTIFACT_METAL, Severity.MODERATE): [
        "clip_extreme",
        "denoise_dncnn",
    ],
    (DegradationType.ARTIFACT_METAL, Severity.SEVERE): [
        "clip_extreme",
        "denoise_dncnn",
    ],
    # ---- Noise (DnCNN for all severities) ----
    (DegradationType.NOISE, Severity.MILD): ["denoise_dncnn"],
    (DegradationType.NOISE, Severity.MODERATE): ["denoise_dncnn"],
    (DegradationType.NOISE, Severity.SEVERE): ["denoise_dncnn"],
    # ---- Blur ----
    (DegradationType.BLUR, Severity.MILD): [
        "denoise_dncnn",
        "sharpen_usm",
    ],
    (DegradationType.BLUR, Severity.MODERATE): [
        "denoise_dncnn",
        ("deblur_richardson_lucy", {"iterations": 15}),
    ],
    (DegradationType.BLUR, Severity.SEVERE): [
        "denoise_dncnn",
        ("deblur_richardson_lucy", {"iterations": 30, "psf_sigma": 1.5}),
    ],
    # ---- Low resolution ----
    (DegradationType.LOW_RESOLUTION, Severity.MILD): [
        "denoise_dncnn",
    ],
    (DegradationType.LOW_RESOLUTION, Severity.MODERATE): [
        "denoise_dncnn",
        "sharpen_usm",
    ],
    (DegradationType.LOW_RESOLUTION, Severity.SEVERE): [
        "denoise_dncnn",
        ("sharpen_usm", {"amount": 2.0}),
    ],
}

# artifact > noise > blur > lowres
_DEGRADATION_PRIORITY = {
    DegradationType.ARTIFACT_METAL: 0,
    DegradationType.NOISE: 1,
    DegradationType.BLUR: 2,
    DegradationType.LOW_RESOLUTION: 3,
}


class RuleBasedPlanner(BasePlanner):
    """基于规则映射的 Planner。"""

    def __init__(self, rules: dict | None = None, max_chain: int = 4) -> None:
        self.rules = rules or DEFAULT_RULES
        self.max_chain = max_chain

    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        steps: list[ToolCall] = []
        reasons: list[str] = []
        seen_tools: set[str] = set()

        sorted_degs = sorted(
            report.degradations,
            key=lambda ds: _DEGRADATION_PRIORITY.get(ds[0], 99),
        )

        for deg_type, severity in sorted_degs:
            tool_entries = self.rules.get((deg_type, severity), [])
            for entry in tool_entries:
                if len(steps) >= self.max_chain:
                    break
                if isinstance(entry, tuple):
                    name, params = entry
                else:
                    name, params = entry, {}
                if name in seen_tools:
                    continue
                seen_tools.add(name)
                steps.append(ToolCall(tool_name=name, params=params))
                reasons.append(f"{deg_type.value}({severity.value}) -> {name}")

        return Plan(
            steps=steps,
            reasoning="; ".join(reasons) if reasons else "No degradation detected, no tools needed.",
        )
