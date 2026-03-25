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
class ToolMeta:
    """工具元信息 — 供 planner 筛选和排序。"""
    category: str = "other"
    suitable_for: list[str] = field(default_factory=list)
    expected_cost: str = "cheap"
    expected_safety: str = "safe"
    params_schema: dict[str, Any] = field(default_factory=dict)

    def matches_degradation(self, degradation_type: str) -> bool:
        return degradation_type in self.suitable_for


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

    可选重写:
    - meta: 工具元信息 (供 planner 筛选)
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

    @property
    def meta(self) -> ToolMeta:
        """工具元信息，子类可重写以提供详细信息。"""
        return ToolMeta()

    @abstractmethod
    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        """执行图像修复。"""
        ...

    def validate_input(self, image: np.ndarray) -> bool:
        """校验输入图像是否合法。"""
        if image is None or image.size == 0:
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
