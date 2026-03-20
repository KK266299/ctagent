# ============================================================================
# 模块职责: 退化类型定义 — 枚举与数据结构
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR)
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DegradationType(Enum):
    """CT 图像退化类型枚举。"""
    NOISE = "noise"
    BLUR = "blur"
    ARTIFACT_METAL = "artifact_metal"
    ARTIFACT_STREAK = "artifact_streak"
    ARTIFACT_RING = "artifact_ring"
    LOW_RESOLUTION = "low_resolution"
    LOW_DOSE = "low_dose"
    UNKNOWN = "unknown"


class Severity(Enum):
    """退化严重程度。"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class DegradationReport:
    """退化检测报告。"""
    degradations: list[tuple[DegradationType, Severity]] = field(default_factory=list)
    iqa_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_degradation(self) -> DegradationType | None:
        """返回最严重的退化类型。"""
        if not self.degradations:
            return None
        severity_order = {Severity.SEVERE: 0, Severity.MODERATE: 1, Severity.MILD: 2}
        sorted_deg = sorted(self.degradations, key=lambda x: severity_order.get(x[1], 99))
        return sorted_deg[0][0]
