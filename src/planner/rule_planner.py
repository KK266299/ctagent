# ============================================================================
# 模块职责: 基于规则的 Planner — 第一版 MVP 使用硬编码规则
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — rule-based planning
# ============================================================================
from __future__ import annotations

from typing import Any

from src.degradations.types import DegradationReport, DegradationType, Severity
from src.planner.base import BasePlanner, Plan, ToolCall


# 默认规则：(退化类型, 严重度) -> 工具列表
DEFAULT_RULES: dict[tuple[DegradationType, Severity], list[str]] = {
    (DegradationType.NOISE, Severity.MILD): ["denoise_nlm"],
    (DegradationType.NOISE, Severity.MODERATE): ["denoise_nlm"],
    (DegradationType.NOISE, Severity.SEVERE): ["ldct_denoiser"],
    (DegradationType.BLUR, Severity.MILD): ["sharpen_usm"],
    (DegradationType.BLUR, Severity.SEVERE): ["sr_ct"],
    (DegradationType.ARTIFACT_METAL, Severity.MILD): ["mar_rise"],
    (DegradationType.ARTIFACT_METAL, Severity.MODERATE): ["mar_rise"],
    (DegradationType.ARTIFACT_METAL, Severity.SEVERE): ["mar_rise"],
    (DegradationType.LOW_RESOLUTION, Severity.MILD): ["sr_ct"],
    (DegradationType.LOW_RESOLUTION, Severity.MODERATE): ["sr_ct"],
    (DegradationType.LOW_RESOLUTION, Severity.SEVERE): ["sr_ct"],
}


class RuleBasedPlanner(BasePlanner):
    """基于规则映射的 Planner。"""

    def __init__(self, rules: dict | None = None, max_chain: int = 3) -> None:
        self.rules = rules or DEFAULT_RULES
        self.max_chain = max_chain

    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        steps: list[ToolCall] = []
        reasons: list[str] = []

        for deg_type, severity in report.degradations:
            tool_names = self.rules.get((deg_type, severity), [])
            for t in tool_names:
                if len(steps) >= self.max_chain:
                    break
                steps.append(ToolCall(tool_name=t))
                reasons.append(f"{deg_type.value}({severity.value}) -> {t}")

        return Plan(
            steps=steps,
            reasoning="; ".join(reasons) if reasons else "No degradation detected, no tools needed.",
        )
