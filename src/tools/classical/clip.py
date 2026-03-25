# ============================================================================
# 模块职责: 极值裁剪 — 将 μ 值图像裁剪到合理范围
#   clip_extreme: 最轻量的 metal artifact 预处理
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class ClipExtreme(BaseTool):
    """极值裁剪 — 将超出合理 μ 值范围的像素截断。

    适用: metal artifact 产生的极端像素（μ >> 0.5 或 μ < 0）
    原理: 直接 clip 到 [low, high]，消除极端离群值
    优势: 零计算代价，无副作用，应作为所有 artifact chain 的第一步
    """

    @property
    def name(self) -> str:
        return "clip_extreme"

    @property
    def description(self) -> str:
        return (
            "Clip extreme pixel values to a valid μ range [0, max_mu]. "
            "Essential first step for metal artifact images. Zero cost, zero risk."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="preprocess",
            suitable_for=["artifact", "noise"],
            expected_cost="cheap",
            expected_safety="safe",
            params_schema={
                "low": {"type": "float", "default": 0.0, "range": [0.0, 0.0]},
                "high": {"type": "float", "default": 0.55, "range": [0.3, 1.0]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        try:
            low = float(kwargs.get("low", 0.0))
        except (TypeError, ValueError):
            low = 0.0
        try:
            high = float(kwargs.get("high", 0.55))
        except (TypeError, ValueError):
            high = 0.55

        n_clipped = int(np.sum((image < low) | (image > high)))
        clipped = np.clip(image, low, high).astype(np.float32)
        return ToolResult(
            image=clipped,
            tool_name=self.name,
            metadata={"low": low, "high": high, "pixels_clipped": n_clipped},
        )
