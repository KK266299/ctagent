# ============================================================================
# 模块职责: Planner 基类 — 统一的规划接口
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — planner interface
#       MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
# ============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.degradations.types import DegradationReport


@dataclass
class ToolCall:
    """单个工具调用描述。"""
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """工具调用计划。"""
    steps: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""

    def __len__(self) -> int:
        return len(self.steps)

    def tool_names(self) -> list[str]:
        return [s.tool_name for s in self.steps]


class BasePlanner(ABC):
    """Planner 基类。"""

    @abstractmethod
    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        """根据退化报告生成工具调用计划。

        Args:
            report: 退化检测报告
            **kwargs: 额外上下文

        Returns:
            Plan 包含有序的工具调用列表
        """
        ...
