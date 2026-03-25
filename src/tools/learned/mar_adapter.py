# ============================================================================
# 模块职责: MAR Learned Adapter — 深度学习 Metal Artifact Reduction 占位
#   统一接口，支持挂载 RISE-MAR / FIND-Net / ADN 等模型
#   当前为 placeholder: 降级为 classical MAR pipeline
# 参考: RISE-MAR (https://github.com/Masaaki-75/rise-mar)
#       FIND-Net (https://github.com/Farid-Tasharofi/FIND-Net)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class MARLearnedAdapter(BaseTool):
    """MAR Learned Adapter — 深度学习 MAR 的统一接口。

    当前: placeholder，使用 clip + threshold_replace + TV 组合降级
    未来: 挂载 RISE-MAR / FIND-Net / ADN 等预训练模型
    """

    @property
    def name(self) -> str:
        return "mar_learned"

    @property
    def description(self) -> str:
        return (
            "Deep learning MAR adapter (placeholder: uses classical MAR pipeline). "
            "Will support RISE-MAR/FIND-Net when model weights are available."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="mar",
            suitable_for=["artifact"],
            expected_cost="expensive",
            expected_safety="moderate",
            params_schema={
                "model": {"type": "str", "default": "placeholder", "options": ["placeholder", "rise_mar", "find_net"]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        model = str(kwargs.get("model", "placeholder"))

        if model != "placeholder":
            return ToolResult(
                image=image.copy(),
                tool_name=self.name,
                success=False,
                message=f"Model '{model}' not yet available. Use placeholder or classical MAR tools.",
            )

        clipped = np.clip(image, 0.0, 0.5).astype(np.float64)

        metal_mask = image > 0.4
        if metal_mask.sum() > 0:
            from scipy.ndimage import binary_dilation, uniform_filter
            struct = np.ones((7, 7), dtype=bool)
            expanded = binary_dilation(metal_mask, structure=struct)
            safe = clipped.copy()
            safe[expanded] = np.nan
            local_mean = uniform_filter(np.where(np.isnan(safe), 0, safe), size=15)
            count = uniform_filter((~np.isnan(safe)).astype(np.float64), size=15)
            count = np.maximum(count, 1e-10)
            local_mean = local_mean / count
            clipped[expanded] = local_mean[expanded]

        from skimage.restoration import denoise_tv_chambolle
        result = denoise_tv_chambolle(clipped, weight=0.03)

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            metadata={"model": model, "metal_pixels": int(metal_mask.sum())},
        )
