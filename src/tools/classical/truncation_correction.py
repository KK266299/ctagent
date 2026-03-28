# ============================================================================
# 模块职责: 截断伪影校正工具
#   truncation_correction_extrapolate: 边缘外推 + 平滑过渡
#   truncation_correction_tv:          TV 正则化抑制截断亮边
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


def _safe_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


@ToolRegistry.register
class TruncationCorrectionExtrapolate(BaseTool):
    """截断校正 — 边缘外推与平滑衰减。

    截断伪影源于 FOV 边缘的投影数据不完整。
    在图像域用余弦窗将边缘值平滑衰减到背景值，
    减轻亮边效应。
    """

    @property
    def name(self) -> str:
        return "truncation_correction_extrapolate"

    @property
    def description(self) -> str:
        return "Correct truncation artifacts by smooth edge extrapolation with cosine tapering."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_truncation"],
            expected_cost="cheap",
            expected_safety="safe",
            params_schema={
                "margin": {"type": "int", "default": 20, "range": [5, 50]},
                "background": {"type": "float", "default": 0.0, "range": [0.0, 0.05]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        margin = _safe_int(kwargs.get("margin", 20), 20)
        background = _safe_float(kwargs.get("background", 0.0), 0.0)

        arr = image.astype(np.float64)
        h, w = arr.shape
        result = arr.copy()

        taper_y = np.ones(h, dtype=np.float64)
        taper_x = np.ones(w, dtype=np.float64)

        m = min(margin, h // 4, w // 4)
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, m)))

        taper_y[:m] = ramp
        taper_y[-m:] = ramp[::-1]
        taper_x[:m] = ramp
        taper_x[-m:] = ramp[::-1]

        taper_2d = taper_y[:, None] * taper_x[None, :]

        result = result * taper_2d + background * (1.0 - taper_2d)

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"Truncation correction (extrapolate, margin={m})",
            metadata={"margin": m, "background": background},
        )


@ToolRegistry.register
class TruncationCorrectionTV(BaseTool):
    """截断校正 — TV 正则化抑制亮边。

    截断伪影在图像域表现为 FOV 边缘的亮带。
    TV 正则化可有效平滑这些突变结构。
    """

    @property
    def name(self) -> str:
        return "truncation_correction_tv"

    @property
    def description(self) -> str:
        return "Suppress truncation artifact bright edges using Total Variation regularization."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_truncation"],
            expected_cost="medium",
            expected_safety="safe",
            params_schema={
                "weight": {"type": "float", "default": 0.12, "range": [0.01, 0.5]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import denoise_tv_chambolle

        weight = _safe_float(kwargs.get("weight", 0.12), 0.12)
        arr = image.astype(np.float64)
        result = denoise_tv_chambolle(arr, weight=weight)

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"Truncation correction (TV, weight={weight})",
            metadata={"weight": weight},
        )
