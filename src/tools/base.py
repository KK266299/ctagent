# ============================================================================
# 模块职责: 工具基类 — 所有修复工具的统一抽象接口
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — tool base class
#       MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — tool interface
# ============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ToolResult:
    """工具执行结果。"""
    image: np.ndarray
    tool_name: str
    success: bool = True
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """所有修复工具的基类。

    子类需实现:
    - name: 工具名称
    - description: 工具描述 (供 planner 参考)
    - run(): 执行修复
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一标识名。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能描述，用于 planner 选择。"""
        ...

    @abstractmethod
    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        """执行图像修复。

        Args:
            image: 输入图像
            **kwargs: 工具特定参数

        Returns:
            ToolResult 包含修复后图像和元信息
        """
        ...

    def validate_input(self, image: np.ndarray) -> bool:
        """校验输入图像是否合法。"""
        if image is None or image.size == 0:
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
