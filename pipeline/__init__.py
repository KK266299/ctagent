# ============================================================================
# 模块职责: 端到端流程编排 — agent 系统的"主控"层
#   包含: planner 子模块、单次线性 pipeline、闭环 agent workflow
#   当前阶段: 定义共享类型 + planner bridge
#   后续阶段: 实现 single_pass.py / agent_loop.py / runner.py
# 参考: AgenticIR — pipeline orchestration
#       4KAgent — perception-planning-execution-reflection loop
#       JarvisIR — pipeline runner
# ============================================================================

from pipeline.types import PipelineState
from pipeline.single_pass import SinglePassPipeline, SinglePassResult
from pipeline.agent_loop import ClosedLoopPipeline, ClosedLoopResult
from pipeline.replan import (
    ReplanDecision,
    RuleBasedReplanner,
    ScoreAwareReplanner,
    OLD_STRATEGIES,
    EXPANDED_STRATEGIES,
)
from pipeline.api_guided_planner import APIGuidedPlanner

__all__ = [
    "PipelineState",
    "SinglePassPipeline",
    "SinglePassResult",
    "ClosedLoopPipeline",
    "ClosedLoopResult",
    "ReplanDecision",
    "RuleBasedReplanner",
    "ScoreAwareReplanner",
    "OLD_STRATEGIES",
    "EXPANDED_STRATEGIES",
    "APIGuidedPlanner",
]
