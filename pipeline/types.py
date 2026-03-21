# ============================================================================
# 模块职责: Pipeline 共享类型 — Plan / ToolCall / PipelineState 等
#   Plan, ToolCall: 从 src/planner/base.py re-export (保持兼容)
#   PipelineState: 新增，闭环 agent workflow 的状态容器
# 参考: 4KAgent — pipeline state management
#       AgenticIR — exploration state
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.planner.base import Plan, ToolCall  # re-export

__all__ = ["Plan", "ToolCall", "PipelineState"]


@dataclass
class PipelineState:
    """闭环 agent pipeline 的运行时状态。

    在 agent_loop 的每个迭代中，此对象在各模块之间传递:
    planner → executor → judge → (replan or downstream)

    Attributes:
        image_original: 原始输入图像 (不可变)
        image_current: 当前处理后图像
        plan: 当前 iteration 的工具调用计划
        trace_ids: 所有 ExecutionTrace 的 ID 列表
        judge_verdicts: 各 judge 的评判结果
        diagnosis_result: 下游诊断结果 (闭环结束后填入)
        iteration: 当前迭代轮次 (0-based)
        max_iterations: 最大允许迭代次数
        done: 是否完成 (judge 通过 or 达到 max_iterations)
        metadata: 自由扩展字段
    """
    image_original: np.ndarray | None = None
    image_current: np.ndarray | None = None
    plan: Plan | None = None
    trace_ids: list[str] = field(default_factory=list)
    judge_verdicts: list[dict[str, Any]] = field(default_factory=list)
    diagnosis_result: Any = None
    iteration: int = 0
    max_iterations: int = 3
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_replan(self) -> bool:
        """根据最近一次 judge verdict 判断是否需要 replan。"""
        if not self.judge_verdicts:
            return False
        latest = self.judge_verdicts[-1]
        return not latest.get("passed", True)

    @property
    def can_continue(self) -> bool:
        """是否还可以继续迭代。"""
        return not self.done and self.iteration < self.max_iterations
