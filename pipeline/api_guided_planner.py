# ============================================================================
# 模块职责: API-Guided Planner — 通过闭源 API (GPT-4o/Claude) 进行
#   tool selection, parameter selection, retry/stop/abstain 决策
#
#   双角色:
#     1. BasePlanner    — iter#0 首轮规划 (vision call, 传入退化图)
#     2. BaseReplanner  — iter#1+ judge FAIL 后 replan (text call, 传 scores)
#
#   三种运行模式 (mode):
#     "full"         — plan() + replan() 都走 API
#     "replan_only"  — plan() 走 rule-based, 仅 replan() 走 API
#     "rule_based"   — 完全不走 API, 等同于 ScoreAwareReplanner (debug 用)
#
#   内置 fallback: API 失败时自动降级到 rule-based
#
# 参考: 4KAgent — LLM-guided planning + reflection loop
#       AgenticIR — planner with LLM delegation + fallback
#       JarvisIR — VLM as controller dispatching tools
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from llm.api_client import BaseLLMClient, LLMConfig, create_client
from llm.guided_caller import CallRecord, GuidedCaller, GuidedPlanRequest
from llm.response_parser import GuidedDecision
from src.degradations.types import DegradationReport
from src.planner.base import BasePlanner, Plan
from pipeline.replan import (
    BaseReplanner,
    ReplanDecision,
    ReplanFeedback,
    ScoreAwareReplanner,
)

logger = logging.getLogger(__name__)


class APIGuidedPlanner(BasePlanner, BaseReplanner):
    """API-Guided Planner — 通过 LLM API 进行工具选择和参数决策。

    Usage:
        # Full mode — plan + replan 都走 API
        planner = APIGuidedPlanner(llm_config=LLMConfig(model="gpt-4o"))
        pipeline = ClosedLoopPipeline(planner=planner, replanner=planner)

        # Replan-only mode — 首轮用规则, replan 走 API
        planner = APIGuidedPlanner(llm_config=..., mode="replan_only")
        pipeline = ClosedLoopPipeline(planner=rule_planner, replanner=planner)

        # Mock mode — 用自定义 client 做测试
        planner = APIGuidedPlanner(llm_client=mock_client)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        llm_config: LLMConfig | None = None,
        mode: str = "full",
        use_vision: bool = True,
        fallback_planner: BasePlanner | None = None,
        fallback_replanner: BaseReplanner | None = None,
        tool_catalog: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            llm_client: 直接传入 LLM client (优先于 llm_config)
            llm_config: LLM 配置 (provider, model, api_key, ...)
            mode: "full" | "replan_only" | "rule_based"
            use_vision: iter#0 是否传图 (True=vision-first, False=text-only)
            fallback_planner: API 失败时的 plan() 降级
            fallback_replanner: API 失败时的 replan() 降级
            tool_catalog: 自定义工具目录 (默认使用 TOOL_CATALOG)
        """
        self.mode = mode
        self.use_vision = use_vision

        if llm_client is not None:
            client = llm_client
        elif llm_config is not None:
            client = create_client(llm_config)
        else:
            client = None

        self.caller: GuidedCaller | None = None
        if client is not None:
            self.caller = GuidedCaller(
                llm_client=client,
                tool_catalog=tool_catalog,
            )

        self._fallback_planner = fallback_planner
        self._fallback_replanner = fallback_replanner or ScoreAwareReplanner()

        self._last_degradation_report: dict[str, Any] = {}
        self._current_image: np.ndarray | None = None

        self.call_records: list[CallRecord] = []
        self.fallback_events: list[dict[str, Any]] = []

    def reset_records(self) -> None:
        """在 case 之间清空调用记录和降级事件。"""
        self.call_records.clear()
        self.fallback_events.clear()

    # ------------------------------------------------------------------
    # BasePlanner interface — iter#0 首轮规划
    # ------------------------------------------------------------------

    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        """首轮规划。mode='full' 时走 API (vision), 否则走 fallback。"""
        image = kwargs.get("image")
        self._current_image = image

        self._last_degradation_report = {
            "degradations": [
                {"type": d.value, "severity": s.value}
                for d, s in report.degradations
            ],
            "iqa_scores": report.iqa_scores,
        }

        if self.mode == "full" and self.caller is not None:
            try:
                request = GuidedPlanRequest(
                    iteration=0,
                    degradation_summary=self._last_degradation_report,
                    current_scores=None,
                    history=[],
                    image=image if self.use_vision else None,
                )
                decision = self.caller.call(request)
                self._capture_record()
                logger.info("API plan(): decision=%s reason=%s", decision.decision, decision.reason)

                if decision.decision == "retry" and decision.plan:
                    return decision.plan
                if decision.decision == "abstain":
                    return Plan(reasoning=f"API abstain: {decision.reason}")
                return Plan(reasoning=f"API stop at iter#0: {decision.reason}")

            except Exception as e:
                logger.warning("API plan() failed, falling back to rule-based: %s", e)
                self.fallback_events.append({"iteration": 0, "phase": "plan", "reason": str(e)})

        return self._fallback_plan(report, **kwargs)

    def _fallback_plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        if self._fallback_planner is not None:
            return self._fallback_planner.plan(report, **kwargs)
        from src.planner.rule_planner import RuleBasedPlanner
        return RuleBasedPlanner().plan(report, **kwargs)

    # ------------------------------------------------------------------
    # BaseReplanner interface — iter#1+ judge FAIL 后 replan
    # ------------------------------------------------------------------

    def replan(self, feedback: ReplanFeedback) -> ReplanDecision:
        """Replan 决策。mode='full'|'replan_only' 时走 API, 否则走 fallback。"""
        if self.mode == "rule_based" or self.caller is None:
            return self._fallback_replanner.replan(feedback)

        try:
            history = self._build_history(feedback)
            bottleneck = "safety" if feedback.safety_score < feedback.quality_score else "quality"

            request = GuidedPlanRequest(
                iteration=feedback.iteration,
                degradation_summary=self._last_degradation_report,
                current_scores={
                    "quality_score": round(feedback.quality_score, 4),
                    "quality_passed": feedback.quality_passed,
                    "safety_score": round(feedback.safety_score, 4),
                    "safety_passed": feedback.safety_passed,
                    "aggregate_score": round(min(feedback.quality_score, feedback.safety_score), 4),
                    "aggregate_passed": feedback.quality_passed and feedback.safety_passed,
                    "bottleneck": bottleneck,
                },
                history=history,
                image=None,
            )
            decision = self.caller.call(request)
            self._capture_record()
            logger.info(
                "API replan() iter#%d: decision=%s reason=%s",
                feedback.iteration, decision.decision, decision.reason,
            )
            return self._guided_to_replan(decision)

        except Exception as e:
            logger.warning("API replan() failed, falling back: %s", e)
            self.fallback_events.append({"iteration": feedback.iteration, "phase": "replan", "reason": str(e)})
            return self._fallback_replanner.replan(feedback)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _capture_record(self) -> None:
        """从 caller.last_record 捕获到 call_records 列表。"""
        if self.caller is not None and self.caller.last_record is not None:
            self.call_records.append(self.caller.last_record)

    @staticmethod
    def _build_history(feedback: ReplanFeedback) -> list[dict[str, Any]]:
        """从 ReplanFeedback 构建 history 列表。"""
        history = []
        for i, (plan_tools, score) in enumerate(
            zip(feedback.previous_plans, feedback.previous_scores)
        ):
            history.append({
                "iteration": i,
                "tools": plan_tools,
                "aggregate_score": round(score, 4),
                "passed": False,
            })
        return history

    @staticmethod
    def _guided_to_replan(decision: GuidedDecision) -> ReplanDecision:
        """将 GuidedDecision 转换为 ReplanDecision。"""
        if decision.decision == "retry" and decision.plan:
            return ReplanDecision(
                action="retry",
                plan=decision.plan,
                reason=decision.reason,
            )
        return ReplanDecision(
            action=decision.decision,
            reason=decision.reason,
        )
