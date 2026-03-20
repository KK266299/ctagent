# ============================================================================
# 模块职责: 金属伪影去除 (MAR) 工具
# 参考: RISE-MAR (https://github.com/Masaaki-75/rise-mar) — MAR 算法
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class MARTool(BaseTool):
    """金属伪影去除工具（占位，待接入预训练模型）。"""

    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.model = None

    @property
    def name(self) -> str:
        return "mar_rise"

    @property
    def description(self) -> str:
        return "Metal Artifact Reduction for CT images with metal implants."

    def _load_model(self) -> None:
        """延迟加载模型权重。"""
        if self.checkpoint is None:
            return
        # TODO: 加载 RISE-MAR 预训练权重
        pass

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        if self.model is None and self.checkpoint is not None:
            self._load_model()

        # TODO: 实现实际推理逻辑
        # 当前为 passthrough
        return ToolResult(
            image=image,
            tool_name=self.name,
            message="MAR placeholder — model not loaded",
            metadata={"checkpoint": self.checkpoint},
        )
