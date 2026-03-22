# ============================================================================
# 模块职责: Replan 决策器 — judge FAIL 时决定下一步行动
#   当前实现:
#     RuleBasedReplanner  — 顺序尝试固定策略列表
#     ScoreAwareReplanner — 根据 quality/safety 瓶颈自适应选择策略
#   后续扩展: AgentReplanner (基于 LLM 的 reflection-driven replan)
#   决策类型:
#     retry    — 用新 Plan 重新执行
#     stop     — 接受当前最佳结果
#     abstain  — 放弃修复, 返回原始退化图
#
#   策略预设:
#     OLD_STRATEGIES      — 仅 gaussian + sharpen (原始工具集)
#     EXPANDED_STRATEGIES — 顺序尝试 NLM → wiener+CLAHE → gentle TV
# 参考: 4KAgent — reflection → replan loop
#       AgenticIR — rollback & retry mechanism
# ============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.planner.base import Plan, ToolCall


@dataclass
class ReplanFeedback:
    """传递给 replanner 的上下文信息。"""
    iteration: int
    quality_passed: bool
    quality_score: float
    safety_passed: bool
    safety_score: float
    quality_details: dict[str, Any] = field(default_factory=dict)
    safety_details: dict[str, Any] = field(default_factory=dict)
    previous_plans: list[list[str]] = field(default_factory=list)
    previous_scores: list[float] = field(default_factory=list)


@dataclass
class ReplanDecision:
    """Replan 决策结果。"""
    action: str  # "retry" | "stop" | "abstain"
    plan: Plan | None = None
    reason: str = ""


class BaseReplanner(ABC):
    @abstractmethod
    def replan(self, feedback: ReplanFeedback) -> ReplanDecision:
        ...


# ---- 策略预设 ----

OLD_STRATEGIES: list[dict[str, Any]] = [
    {
        "tools": [
            ToolCall("denoise_gaussian", params={"sigma": 1.0}),
            ToolCall("sharpen_usm", params={"radius": 1.5, "amount": 0.5}),
        ],
        "reason": "same denoise + mild sharpen to recover structure",
    },
    {
        "tools": [
            ToolCall("denoise_gaussian", params={"sigma": 0.7}),
            ToolCall("sharpen_usm", params={"radius": 1.5, "amount": 0.8}),
        ],
        "reason": "gentler denoise + stronger sharpen for better safety",
    },
]


EXPANDED_STRATEGIES: list[dict[str, Any]] = [
    {
        "tools": [ToolCall("denoise_nlm")],
        "reason": "NLM: structure-aware via self-similarity, may yield better SSIM",
    },
    {
        "tools": [ToolCall("denoise_tv", params={"weight": 0.03})],
        "reason": "very gentle TV: safety-first minimal intervention",
    },
    {
        "tools": [ToolCall("denoise_bilateral", params={"sigma_color": 0.03, "sigma_spatial": 3})],
        "reason": "gentle bilateral: alternative edge-preserving approach",
    },
]


class RuleBasedReplanner(BaseReplanner):
    """基于规则的 Replanner — 顺序尝试策略列表。"""

    def __init__(self, strategies: list[dict[str, Any]] | None = None) -> None:
        self.strategies = strategies if strategies is not None else OLD_STRATEGIES

    def replan(self, feedback: ReplanFeedback) -> ReplanDecision:
        if feedback.quality_passed and feedback.safety_passed:
            return ReplanDecision(action="stop", reason="Both judges passed")

        replan_idx = len(feedback.previous_plans) - 1
        if replan_idx >= len(self.strategies):
            return ReplanDecision(
                action="stop",
                reason=f"Exhausted {len(self.strategies)} replan strategies",
            )

        if len(feedback.previous_scores) >= 2:
            if feedback.previous_scores[-1] <= feedback.previous_scores[-2]:
                return ReplanDecision(
                    action="stop",
                    reason="Score not improving, accept best so far",
                )

        strategy = self.strategies[replan_idx]
        q_label = "PASS" if feedback.quality_passed else "FAIL"
        s_label = "PASS" if feedback.safety_passed else "FAIL"

        return ReplanDecision(
            action="retry",
            plan=Plan(
                steps=list(strategy["tools"]),
                reasoning=f"Replan #{replan_idx + 1}: quality={q_label} safety={s_label} → {strategy['reason']}",
            ),
            reason=strategy["reason"],
        )


class ScoreAwareReplanner(BaseReplanner):
    """自适应 Replanner — 根据 quality/safety 哪个是瓶颈选择不同策略方向。

    当 safety 是瓶颈 (safety < quality): 选择更温和的工具, 减少图像修改幅度
    当 quality 是瓶颈 (quality < safety): 选择更强的去噪工具
    当两者都低: 尝试完全不同的方法
    """

    SAFETY_STRATEGIES: list[dict[str, Any]] = [
        {
            "tools": [ToolCall("denoise_tv", params={"weight": 0.07})],
            "reason": "moderate TV: balance quality-safety for better aggregate",
        },
        {
            "tools": [ToolCall("denoise_tv", params={"weight": 0.05})],
            "reason": "gentle TV: prioritize safety over quality",
        },
        {
            "tools": [ToolCall("denoise_bilateral", params={"sigma_color": 0.04, "sigma_spatial": 4})],
            "reason": "bilateral: different algorithm for safety-first approach",
        },
    ]

    QUALITY_STRATEGIES: list[dict[str, Any]] = [
        {
            "tools": [ToolCall("denoise_tv", params={"weight": 0.15})],
            "reason": "stronger TV for better quality",
        },
        {
            "tools": [ToolCall("denoise_nlm")],
            "reason": "NLM: strong structure-aware denoising",
        },
        {
            "tools": [
                ToolCall("denoise_tv", params={"weight": 0.10}),
                ToolCall("sharpen_usm", params={"radius": 1.5, "amount": 0.3}),
            ],
            "reason": "moderate TV + mild sharpen",
        },
    ]

    def replan(self, feedback: ReplanFeedback) -> ReplanDecision:
        if feedback.quality_passed and feedback.safety_passed:
            return ReplanDecision(action="stop", reason="Both judges passed")

        replan_idx = len(feedback.previous_plans) - 1

        if len(feedback.previous_scores) >= 2:
            if feedback.previous_scores[-1] <= feedback.previous_scores[-2]:
                return ReplanDecision(
                    action="stop",
                    reason="Score not improving, accept best so far",
                )

        if feedback.safety_score < feedback.quality_score:
            strategies = self.SAFETY_STRATEGIES
            bottleneck = "safety"
        else:
            strategies = self.QUALITY_STRATEGIES
            bottleneck = "quality"

        if replan_idx >= len(strategies):
            return ReplanDecision(
                action="stop",
                reason=f"Exhausted {len(strategies)} {bottleneck}-focused strategies",
            )

        strategy = strategies[replan_idx]
        q_label = "PASS" if feedback.quality_passed else "FAIL"
        s_label = "PASS" if feedback.safety_passed else "FAIL"

        return ReplanDecision(
            action="retry",
            plan=Plan(
                steps=list(strategy["tools"]),
                reasoning=(
                    f"Replan #{replan_idx + 1} ({bottleneck} bottleneck): "
                    f"quality={q_label}({feedback.quality_score:.3f}) "
                    f"safety={s_label}({feedback.safety_score:.3f}) "
                    f"→ {strategy['reason']}"
                ),
            ),
            reason=f"{bottleneck} bottleneck → {strategy['reason']}",
        )
