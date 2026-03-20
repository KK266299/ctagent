# ============================================================================
# 模块职责: CT 超分辨率工具
# 参考: ProCT (https://github.com/Masaaki-75/proct) — CT 超分辨率
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class CTSuperResolutionTool(BaseTool):
    """CT 图像超分辨率工具（占位，待接入预训练模型）。"""

    def __init__(self, checkpoint: str | None = None, device: str = "cuda", scale_factor: int = 2) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.scale_factor = scale_factor
        self.model = None

    @property
    def name(self) -> str:
        return "sr_ct"

    @property
    def description(self) -> str:
        return f"CT Super Resolution (x{self.scale_factor}) for low-resolution CT images."

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        # TODO: 加载并运行预训练超分模型
        return ToolResult(
            image=image,
            tool_name=self.name,
            message="CT SR placeholder — model not loaded",
        )
