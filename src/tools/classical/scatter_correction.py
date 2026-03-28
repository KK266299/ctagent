# ============================================================================
# 模块职责: 散射伪影校正工具
#   scatter_correction_detrend: 低频散射分量估计与减除
#   scatter_correction_clahe:   自适应对比度增强恢复散射损失的对比度
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


@ToolRegistry.register
class ScatterCorrectionDetrend(BaseTool):
    """散射校正 — 低频散射分量估计与减除。

    散射导致一个平滑的低频偏移叠加在真实投影上。
    用大核高斯估计散射分量，按比例减除。
    """

    @property
    def name(self) -> str:
        return "scatter_correction_detrend"

    @property
    def description(self) -> str:
        return "Estimate and subtract low-frequency scatter component to restore contrast."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_scatter"],
            expected_cost="cheap",
            expected_safety="safe",
            params_schema={
                "kernel_sigma": {"type": "float", "default": 40.0, "range": [15.0, 80.0]},
                "scatter_fraction": {"type": "float", "default": 0.3, "range": [0.05, 0.8]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.ndimage import gaussian_filter

        kernel_sigma = _safe_float(kwargs.get("kernel_sigma", 40.0), 40.0)
        scatter_fraction = _safe_float(kwargs.get("scatter_fraction", 0.3), 0.3)

        arr = image.astype(np.float64)
        scatter_est = gaussian_filter(arr, sigma=kernel_sigma)

        result = arr - scatter_fraction * scatter_est
        result = np.clip(result, 0.0, arr.max() * 1.5)

        scale = np.percentile(arr[arr > 0.02], 95) if np.any(arr > 0.02) else 1.0
        result_scale = np.percentile(result[result > 0.02], 95) if np.any(result > 0.02) else 1.0
        if result_scale > 1e-10:
            result = result * (scale / result_scale)

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"Scatter correction (detrend, sigma={kernel_sigma}, frac={scatter_fraction})",
            metadata={"kernel_sigma": kernel_sigma, "scatter_fraction": scatter_fraction},
        )


@ToolRegistry.register
class ScatterCorrectionCLAHE(BaseTool):
    """散射校正 — CLAHE 自适应对比度增强。

    散射导致全局对比度降低，用 CLAHE 在局部恢复对比度。
    """

    @property
    def name(self) -> str:
        return "scatter_correction_clahe"

    @property
    def description(self) -> str:
        return "Restore contrast lost to scatter artifacts using adaptive histogram equalization."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_scatter"],
            expected_cost="cheap",
            expected_safety="moderate",
            params_schema={
                "clip_limit": {"type": "float", "default": 0.01, "range": [0.005, 0.05]},
                "tile_size": {"type": "int", "default": 8, "range": [4, 16]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.exposure import equalize_adapthist

        clip_limit = _safe_float(kwargs.get("clip_limit", 0.01), 0.01)

        arr = image.astype(np.float64)
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin < 1e-10:
            return ToolResult(image=arr.astype(np.float32), tool_name=self.name,
                              success=False, message="Image has no dynamic range")

        norm = (arr - vmin) / (vmax - vmin)
        enhanced = equalize_adapthist(norm, clip_limit=clip_limit)
        result = enhanced * (vmax - vmin) + vmin

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"Scatter correction (CLAHE, clip={clip_limit})",
            metadata={"clip_limit": clip_limit},
        )
