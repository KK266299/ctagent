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
    # ---- Ring artifact (polar ring removal → denoise) ----
    (DegradationType.ARTIFACT_RING, Severity.MILD): [
        ("ring_removal_polar", {"sigma": 1.0}),
    ],
    (DegradationType.ARTIFACT_RING, Severity.MODERATE): [
        ("ring_removal_polar", {"sigma": 2.0}),
        "denoise_dncnn",
    ],
    (DegradationType.ARTIFACT_RING, Severity.SEVERE): [
        ("ring_removal_wavelet", {"level": 4, "sigma": 3.0}),
        "denoise_dncnn",
    ],
    # ---- Motion artifact (TV / Wiener → denoise) ----
    (DegradationType.ARTIFACT_MOTION, Severity.MILD): [
        ("motion_correction_tv", {"weight": 0.08}),
    ],
    (DegradationType.ARTIFACT_MOTION, Severity.MODERATE): [
        ("motion_correction_tv", {"weight": 0.15}),
        "denoise_dncnn",
    ],
    (DegradationType.ARTIFACT_MOTION, Severity.SEVERE): [
        ("motion_correction_tv", {"weight": 0.25}),
        "denoise_dncnn",
    ],
    # ---- Beam hardening (flatfield/polynomial → denoise) ----
    (DegradationType.ARTIFACT_BEAM_HARDENING, Severity.MILD): [
        ("bhc_flatfield", {"strength": 0.3}),
    ],
    (DegradationType.ARTIFACT_BEAM_HARDENING, Severity.MODERATE): [
        ("bhc_flatfield", {"strength": 0.5}),
        "denoise_dncnn",
    ],
    (DegradationType.ARTIFACT_BEAM_HARDENING, Severity.SEVERE): [
        ("bhc_flatfield", {"strength": 0.8}),
        ("bhc_polynomial", {"strength": 1.5}),
        "denoise_dncnn",
    ],
    # ---- Scatter artifact (detrend → contrast restore) ----
    (DegradationType.ARTIFACT_SCATTER, Severity.MILD): [
        ("scatter_correction_detrend", {"scatter_fraction": 0.15}),
    ],
    (DegradationType.ARTIFACT_SCATTER, Severity.MODERATE): [
        ("scatter_correction_detrend", {"scatter_fraction": 0.3}),
        "denoise_dncnn",
    ],
    (DegradationType.ARTIFACT_SCATTER, Severity.SEVERE): [
        ("scatter_correction_detrend", {"scatter_fraction": 0.5}),
        "scatter_correction_clahe",
        "denoise_dncnn",
    ],
    # ---- Truncation artifact (extrapolate → TV → denoise) ----
    (DegradationType.ARTIFACT_TRUNCATION, Severity.MILD): [
        ("truncation_correction_extrapolate", {"margin": 10}),
    ],
    (DegradationType.ARTIFACT_TRUNCATION, Severity.MODERATE): [
        ("truncation_correction_extrapolate", {"margin": 20}),
        "denoise_dncnn",
    ],
    (DegradationType.ARTIFACT_TRUNCATION, Severity.SEVERE): [
        ("truncation_correction_extrapolate", {"margin": 30}),
        ("truncation_correction_tv", {"weight": 0.15}),
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

# artifact types first (by specificity), then generic degradations
_DEGRADATION_PRIORITY = {
    DegradationType.ARTIFACT_METAL: 0,
    DegradationType.ARTIFACT_RING: 1,
    DegradationType.ARTIFACT_MOTION: 2,
    DegradationType.ARTIFACT_BEAM_HARDENING: 3,
    DegradationType.ARTIFACT_SCATTER: 4,
    DegradationType.ARTIFACT_TRUNCATION: 5,
    DegradationType.NOISE: 6,
    DegradationType.BLUR: 7,
    DegradationType.LOW_RESOLUTION: 8,
}


_GENERIC_TYPES = {
    DegradationType.NOISE,
    DegradationType.BLUR,
    DegradationType.LOW_RESOLUTION,
}

_ARTIFACT_TYPES = {
    DegradationType.ARTIFACT_METAL,
    DegradationType.ARTIFACT_RING,
    DegradationType.ARTIFACT_MOTION,
    DegradationType.ARTIFACT_BEAM_HARDENING,
    DegradationType.ARTIFACT_SCATTER,
    DegradationType.ARTIFACT_TRUNCATION,
}


class RuleBasedPlanner(BasePlanner):
    """基于规则映射的 Planner。

    增加 do-no-harm 门控：当只检测到 mild 级别的通用退化
    （noise/blur/lowres）且没有任何特定伪影时，跳过修复以避免
    在近乎干净的图像上引入新的失真。
    """

    def __init__(
        self,
        rules: dict | None = None,
        max_chain: int = 4,
        skip_mild_generic: bool = True,
    ) -> None:
        self.rules = rules or DEFAULT_RULES
        self.max_chain = max_chain
        self.skip_mild_generic = skip_mild_generic

    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        if not report.degradations:
            return Plan(reasoning="No degradation detected, skipping restoration.")

        if self.skip_mild_generic and self._only_mild_generic(report):
            return Plan(
                reasoning=(
                    "Only mild generic degradation detected "
                    f"({self._format_degs(report.degradations)}), "
                    "skipping to avoid harm."
                ),
            )

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
            reasoning="; ".join(reasons) if reasons else "No degradation detected, skipping restoration.",
        )

    @staticmethod
    def _only_mild_generic(report: DegradationReport) -> bool:
        """True if all detected degradations are mild AND generic (no artifacts)."""
        for deg_type, severity in report.degradations:
            if deg_type in _ARTIFACT_TYPES:
                return False
            if severity != Severity.MILD:
                return False
        return True

    @staticmethod
    def _format_degs(degs: list[tuple[DegradationType, Severity]]) -> str:
        return ", ".join(f"{d.value}/{s.value}" for d, s in degs)
