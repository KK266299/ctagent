# ============================================================================
# 模块职责: Judge 基类 — 所有评判器的统一抽象接口
#   JudgeVerdict: 从 src/judge/quality_judge.py 提升为公共类型
#   BaseJudge:    统一 judge 接口，quality 和 safety 共享
# 参考: MedAgent-Pro — multi-criteria judge pattern
#       4KAgent — reflection module interface
# ============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class JudgeVerdict:
    """评判结果 (所有 judge 共享的输出格式)。

    Attributes:
        passed: 是否通过评判
        score: 综合评分 (0-1 标准化)
        reason: 判定理由 (人类可读)
        judge_type: 评判类型标识 ("quality" / "safety" / ...)
        details: 详细指标/子项
    """
    passed: bool
    score: float
    reason: str
    judge_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "judge_type": self.judge_type,
            "details": self.details,
        }


class BaseJudge(ABC):
    """Judge 基类。

    所有 judge 实现此接口:
    - judge(): 主评判方法，输入修复前后图像，输出 JudgeVerdict
    """

    @property
    @abstractmethod
    def judge_type(self) -> str:
        """评判类型标识。"""
        ...

    @abstractmethod
    def judge(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray,
        reference: np.ndarray | None = None,
        **kwargs: Any,
    ) -> JudgeVerdict:
        """执行评判。

        Args:
            image_before: 修复前图像 (退化图像)
            image_after:  修复后图像
            reference:    参考图像 (可选，有则用全参考指标)
            **kwargs:     额外上下文 (如 execution_trace, plan 等)

        Returns:
            JudgeVerdict 评判结果
        """
        ...


def aggregate_verdicts(verdicts: list[JudgeVerdict]) -> JudgeVerdict:
    """聚合多个 judge 的 verdict 为一个综合判定。

    规则: 任一 judge 不通过 → 综合不通过。综合分数取最低分。
    """
    if not verdicts:
        return JudgeVerdict(passed=True, score=1.0, reason="No judges applied", judge_type="aggregate")

    all_passed = all(v.passed for v in verdicts)
    min_score = min(v.score for v in verdicts)
    reasons = [f"[{v.judge_type}] {v.reason}" for v in verdicts if not v.passed]
    reason = "; ".join(reasons) if reasons else "All judges passed"

    return JudgeVerdict(
        passed=all_passed,
        score=min_score,
        reason=reason,
        judge_type="aggregate",
        details={v.judge_type: v.to_dict() for v in verdicts},
    )
