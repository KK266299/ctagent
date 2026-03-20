# ============================================================================
# 模块职责: 直方图工具 — CLAHE 等对比度增强
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class CLAHE(BaseTool):
    """CLAHE 自适应直方图均衡化。"""

    @property
    def name(self) -> str:
        return "histogram_clahe"

    @property
    def description(self) -> str:
        return "Contrast Limited Adaptive Histogram Equalization for low-contrast images."

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.exposure import equalize_adapthist

        clip_limit = kwargs.get("clip_limit", 0.02)
        enhanced = equalize_adapthist(image, clip_limit=clip_limit)
        return ToolResult(image=enhanced.astype(np.float32), tool_name=self.name)
