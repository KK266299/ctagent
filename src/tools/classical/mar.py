# ============================================================================
# 模块职责: 经典 Metal Artifact Reduction
#   mar_threshold_replace: 检测金属像素 → 周围正常组织均值替换
# 参考: RISE-MAR (https://github.com/Masaaki-75/rise-mar) — threshold + replace 思路
#       AI-MAR-CT (https://github.com/harshitAgr/AI-MAR-CT) — classical MAR survey
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class MARThresholdReplace(BaseTool):
    """金属阈值替换 — 检测超出正常 μ 范围的金属像素，用周围均值替换。

    适用: metal artifact（金属植入物产生的 bright/dark streak）
    原理:
      1. 阈值检测 metal trace: μ > metal_threshold
      2. 膨胀 mask 覆盖 streak 邻域
      3. 用局部均值（排除 metal）填充
    优势: 比 inpaint_biharmonic 快很多，且对 metal 特征有针对性
    局限: 大面积 metal 或复杂 streak 效果有限
    """

    @property
    def name(self) -> str:
        return "mar_threshold_replace"

    @property
    def description(self) -> str:
        return (
            "Classical MAR: detect metal pixels by μ threshold, "
            "replace with local mean from surrounding tissue. "
            "Fast and targeted for metal artifact reduction."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="mar",
            suitable_for=["artifact"],
            expected_cost="cheap",
            expected_safety="safe",
            params_schema={
                "metal_threshold": {"type": "float", "default": 0.55, "range": [0.5, 0.8]},
                "dilate_radius": {"type": "int", "default": 2, "range": [1, 5]},
                "filter_size": {"type": "int", "default": 15, "range": [5, 31]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        try:
            metal_threshold = float(kwargs.get("metal_threshold", 0.4))
        except (TypeError, ValueError):
            metal_threshold = 0.4
        try:
            dilate_radius = int(kwargs.get("dilate_radius", 3))
        except (TypeError, ValueError):
            dilate_radius = 3
        try:
            filter_size = int(kwargs.get("filter_size", 15))
        except (TypeError, ValueError):
            filter_size = 15

        metal_mask = image > metal_threshold

        n_metal = int(metal_mask.sum())
        if n_metal == 0:
            return ToolResult(
                image=image.copy(),
                tool_name=self.name,
                success=True,
                message="No metal pixels detected.",
                metadata={"metal_pixels": 0},
            )

        struct = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), dtype=bool)
        expanded_mask = binary_dilation(metal_mask, structure=struct, iterations=1)

        safe_image = image.copy()
        safe_image[expanded_mask] = np.nan

        local_mean = uniform_filter(
            np.where(np.isnan(safe_image), 0.0, safe_image),
            size=filter_size,
        )
        count_map = uniform_filter(
            (~np.isnan(safe_image)).astype(np.float64),
            size=filter_size,
        )
        count_map = np.maximum(count_map, 1e-10)
        local_mean = local_mean / count_map

        result = image.copy().astype(np.float64)
        result[expanded_mask] = local_mean[expanded_mask]
        result = np.clip(result, 0.0, metal_threshold)

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            metadata={
                "metal_pixels": n_metal,
                "replaced_pixels": int(expanded_mask.sum()),
                "metal_threshold": metal_threshold,
            },
        )
