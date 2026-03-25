# ============================================================================
# 模块职责: SR Learned Adapter — CT 超分辨率/稀疏视图重建占位
#   统一接口，支持挂载 PromptCT / ProCT 等模型
#   当前为 placeholder
# 参考: PromptCT (https://github.com/shibaoshun/PromptCT)
#       ProCT (https://github.com/Masaaki-75/proct)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class SRLearnedAdapter(BaseTool):
    """SR Learned Adapter — CT 超分辨率的统一接口。

    当前: placeholder，使用 USM + Laplacian 组合降级
    未来: 挂载 PromptCT / ProCT 预训练模型
    """

    @property
    def name(self) -> str:
        return "sr_learned"

    @property
    def description(self) -> str:
        return (
            "Deep learning CT super-resolution adapter (placeholder: uses USM+Laplacian). "
            "Will support PromptCT/ProCT when model weights are available."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="sr",
            suitable_for=["low_resolution", "blur"],
            expected_cost="expensive",
            expected_safety="moderate",
            params_schema={
                "model": {"type": "str", "default": "placeholder", "options": ["placeholder", "prompt_ct", "pro_ct"]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        model = str(kwargs.get("model", "placeholder"))

        if model != "placeholder":
            return ToolResult(
                image=image.copy(),
                tool_name=self.name,
                success=False,
                message=f"Model '{model}' not yet available.",
            )

        from skimage.filters import unsharp_mask
        from scipy.ndimage import laplace

        sharpened = unsharp_mask(image.astype(np.float64), radius=2.0, amount=1.5)
        lap = laplace(sharpened)
        enhanced = sharpened - 0.15 * lap
        enhanced = np.clip(enhanced, 0.0, max(image.max(), 1.0))

        return ToolResult(
            image=enhanced.astype(np.float32),
            tool_name=self.name,
            metadata={"model": model},
        )
