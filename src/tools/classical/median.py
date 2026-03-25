# ============================================================================
# 模块职责: 中值滤波 — 对脉冲噪声/条纹伪影特效
#   denoise_median: 标准中值滤波
# 参考: scipy.ndimage.median_filter
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class MedianDenoise(BaseTool):
    """中值滤波 — 对 salt-and-pepper / impulse 噪声和条纹伪影最有效。

    适用: metal artifact 条纹、impulse noise
    原理: 滑动窗口内取中值替代中心像素，天然抑制极端值
    优势: 不产生新值，对条纹伪影的亮/暗线特别有效
    局限: kernel 过大会丢失细节
    """

    @property
    def name(self) -> str:
        return "denoise_median"

    @property
    def description(self) -> str:
        return (
            "Median filter: highly effective against impulse noise and "
            "streak artifacts in CT images."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise", "artifact"],
            expected_cost="cheap",
            expected_safety="safe",
            params_schema={
                "size": {"type": "int", "default": 3, "range": [3, 9]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.ndimage import median_filter

        try:
            size = int(kwargs.get("size", 3))
        except (TypeError, ValueError):
            size = 3
        size = max(3, min(size, 9))

        denoised = median_filter(image.astype(np.float64), size=size)
        return ToolResult(
            image=denoised.astype(np.float32),
            tool_name=self.name,
            metadata={"size": size},
        )
