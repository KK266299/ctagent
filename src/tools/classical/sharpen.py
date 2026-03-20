# ============================================================================
# 模块职责: 锐化工具 — Unsharp Mask 等
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class UnsharpMask(BaseTool):
    """Unsharp Mask 锐化。"""

    @property
    def name(self) -> str:
        return "sharpen_usm"

    @property
    def description(self) -> str:
        return "Unsharp mask sharpening for mildly blurred images."

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.filters import unsharp_mask

        radius = kwargs.get("radius", 2.0)
        amount = kwargs.get("amount", 1.5)
        sharpened = unsharp_mask(image, radius=radius, amount=amount)
        return ToolResult(image=sharpened, tool_name=self.name)
