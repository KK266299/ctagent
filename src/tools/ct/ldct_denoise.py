# ============================================================================
# 模块职责: 低剂量 CT 去噪工具
# 参考: ProCT (https://github.com/Masaaki-75/proct) — CT 去噪
#       PromptCT (https://github.com/shibaoshun/PromptCT) — prompt 驱动增强
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class LDCTDenoiseTool(BaseTool):
    """低剂量 CT 去噪工具（占位，待接入预训练模型）。"""

    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.model = None

    @property
    def name(self) -> str:
        return "ldct_denoiser"

    @property
    def description(self) -> str:
        return "Deep learning based Low-Dose CT denoising."

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        # TODO: 加载并运行预训练去噪模型
        return ToolResult(
            image=image,
            tool_name=self.name,
            message="LDCT denoiser placeholder — model not loaded",
        )
