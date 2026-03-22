# ============================================================================
# 模块职责: 闭环 Agent Pipeline — 多轮 plan → execute → judge → replan
#   当 judge FAIL 时触发 replan, 直到 PASS 或达到 max_iterations
#   每轮始终从原始退化图重新执行 (避免累积伪影)
#   最终选取所有迭代中 aggregate score 最高的结果做诊断
#   完整 trajectory 记录每轮的 plan/trace/verdict/score
# 参考: 4KAgent — closed-loop reflection pipeline
#       AgenticIR — exploration loop with rollback
#       JarvisIR — iterative refinement
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.degradations.detector import DegradationDetector
from src.planner.rule_planner import RuleBasedPlanner
from src.executor.engine import Executor
from src.judge.quality_judge import QualityJudge
from judge.safety_judge import SafetyJudge
from judge.base import aggregate_verdicts, JudgeVerdict
from downstream.mock_diagnosis import MockDiagnosis
from executor.trace import ExecutionTrace
from pipeline.replan import (
    BaseReplanner, RuleBasedReplanner, ReplanFeedback, ReplanDecision,
)

logger = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """单轮迭代的完整记录。"""
    iteration: int = 0
    plan_tools: list[str] = field(default_factory=list)
    plan_reasoning: str = ""
    trace: ExecutionTrace | None = None
    quality_verdict: dict[str, Any] = field(default_factory=dict)
    safety_verdict: dict[str, Any] = field(default_factory=dict)
    aggregate_passed: bool = False
    aggregate_score: float = 0.0
    replan_decision: str = ""
    replan_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "plan_tools": self.plan_tools,
            "plan_reasoning": self.plan_reasoning,
            "trace": self.trace.to_dict() if self.trace else {},
            "quality_verdict": self.quality_verdict,
            "safety_verdict": self.safety_verdict,
            "aggregate_passed": self.aggregate_passed,
            "aggregate_score": self.aggregate_score,
            "replan_decision": self.replan_decision,
            "replan_reason": self.replan_reason,
        }


@dataclass
class ClosedLoopResult:
    """闭环 pipeline 的完整结果。"""
    case_id: str = ""
    mode: str = "closed_loop"
    total_iterations: int = 0
    replan_count: int = 0
    best_iteration: int = 0
    best_score: float = 0.0
    exit_reason: str = ""
    iterations: list[IterationRecord] = field(default_factory=list)
    degradation_report: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    label: dict[str, Any] = field(default_factory=dict)
    diagnosis_correct: bool | None = None

    @property
    def final_quality(self) -> dict[str, Any]:
        if self.iterations:
            return self.iterations[self.best_iteration].quality_verdict
        return {}

    @property
    def final_safety(self) -> dict[str, Any]:
        if self.iterations:
            return self.iterations[self.best_iteration].safety_verdict
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "total_iterations": self.total_iterations,
            "replan_count": self.replan_count,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "exit_reason": self.exit_reason,
            "degradation_report": self.degradation_report,
            "iterations": [it.to_dict() for it in self.iterations],
            "diagnosis": self.diagnosis,
            "label": self.label,
            "diagnosis_correct": self.diagnosis_correct,
        }

    def summary(self) -> str:
        lines = [
            f"Case: {self.case_id} [closed-loop]",
            f"  Iterations: {self.total_iterations} (replan={self.replan_count}), "
            f"best=iter#{self.best_iteration} (score={self.best_score:.3f})",
            f"  Exit: {self.exit_reason}",
        ]
        for it in self.iterations:
            q = it.quality_verdict.get("score", "?")
            s = it.safety_verdict.get("score", "?")
            p = "PASS" if it.aggregate_passed else "FAIL"
            tools = " → ".join(it.plan_tools) or "none"
            lines.append(f"    iter#{it.iteration}: {tools} → {p} (q={q}, s={s})")
            if it.replan_decision:
                lines.append(f"      → replan: {it.replan_decision} ({it.replan_reason})")
        lines.append(f"  Diagnosis: {self.diagnosis.get('prediction', 'N/A')} "
                      f"(conf={self.diagnosis.get('confidence', 'N/A')})")
        if self.diagnosis_correct is not None:
            lines.append(f"  Correct: {self.diagnosis_correct}")
        return "\n".join(lines)


class ClosedLoopPipeline:
    """闭环 Agent Pipeline。

    Usage:
        pipeline = ClosedLoopPipeline(max_iterations=3)
        result = pipeline.run(degraded_image)
        result = pipeline.run_toy_case(case)
    """

    def __init__(
        self,
        detector: DegradationDetector | None = None,
        planner: Any | None = None,
        executor: Executor | None = None,
        quality_judge: QualityJudge | None = None,
        safety_judge: SafetyJudge | None = None,
        replanner: BaseReplanner | None = None,
        diagnosis_adapter: Any | None = None,
        max_iterations: int = 3,
        pass_image_to_planner: bool = False,
    ) -> None:
        self.detector = detector or DegradationDetector()
        self.planner = planner or RuleBasedPlanner()
        self.executor = executor or Executor()
        self.quality_judge = quality_judge or QualityJudge()
        self.safety_judge = safety_judge or SafetyJudge()
        self.replanner = replanner or RuleBasedReplanner()
        self.diagnosis_adapter = diagnosis_adapter or MockDiagnosis()
        self.max_iterations = max_iterations
        self.pass_image_to_planner = pass_image_to_planner
        self._ensure_tools_registered()

    def run(self, image: np.ndarray, case_id: str = "") -> ClosedLoopResult:
        result = ClosedLoopResult(case_id=case_id, mode="closed_loop")

        # 1. Assessment (once)
        report = self.detector.detect(image)
        result.degradation_report = {
            "degradations": [
                {"type": d.value, "severity": s.value}
                for d, s in report.degradations
            ],
            "iqa_scores": report.iqa_scores,
        }
        logger.info("[%s] Degradation: %s", case_id, result.degradation_report["degradations"])

        best_image = image
        best_score = -1.0
        best_iter = 0
        previous_plans: list[list[str]] = []
        previous_agg_scores: list[float] = []
        previous_eff_scores: list[float] = []
        restored_images: list[np.ndarray] = []

        for i in range(self.max_iterations):
            iter_rec = IterationRecord(iteration=i)

            # 2. Plan
            if i == 0:
                plan_kwargs: dict[str, Any] = {}
                if self.pass_image_to_planner:
                    plan_kwargs["image"] = image
                plan = self.planner.plan(report, **plan_kwargs)
            else:
                last_iter = result.iterations[-1]
                feedback = ReplanFeedback(
                    iteration=i,
                    quality_passed=last_iter.quality_verdict.get("passed", False),
                    quality_score=last_iter.quality_verdict.get("score", 0),
                    quality_details=last_iter.quality_verdict,
                    safety_passed=last_iter.safety_verdict.get("passed", False),
                    safety_score=last_iter.safety_verdict.get("score", 0),
                    safety_details=last_iter.safety_verdict,
                    previous_plans=previous_plans,
                    previous_scores=previous_agg_scores,
                )
                decision = self.replanner.replan(feedback)
                iter_rec.replan_decision = decision.action
                iter_rec.replan_reason = decision.reason
                logger.info("[%s] Replan iter#%d: %s — %s", case_id, i, decision.action, decision.reason)

                if decision.action in ("stop", "abstain"):
                    result.iterations.append(iter_rec)
                    result.exit_reason = f"{decision.action}: {decision.reason}"
                    break

                plan = decision.plan
                result.replan_count += 1

            iter_rec.plan_tools = plan.tool_names()
            iter_rec.plan_reasoning = plan.reasoning
            logger.info("[%s] iter#%d Plan: %s", case_id, i, iter_rec.plan_tools)

            # 3. Execute (always from original degraded)
            if plan.steps:
                tool_results = self.executor.execute(plan, image)
                iter_rec.trace = self.executor.last_trace
                restored = (
                    tool_results[-1].image
                    if tool_results and tool_results[-1].success
                    else image
                )
            else:
                restored = image
                logger.info("[%s] iter#%d: No tools to execute", case_id, i)

            restored_images.append(restored)

            # 4. Judge
            qv = self.quality_judge.judge_no_reference(image, restored)
            sv = self.safety_judge.judge(image, restored)
            agg = aggregate_verdicts([
                JudgeVerdict(passed=qv.passed, score=qv.score, reason=qv.reason, judge_type="quality"),
                sv,
            ])

            iter_rec.quality_verdict = {"passed": qv.passed, "score": qv.score, "reason": qv.reason}
            iter_rec.safety_verdict = sv.to_dict()
            iter_rec.aggregate_passed = agg.passed
            iter_rec.aggregate_score = agg.score

            previous_plans.append(iter_rec.plan_tools)
            effective_score = (qv.score + sv.score) / 2.0
            previous_agg_scores.append(agg.score)
            previous_eff_scores.append(effective_score)
            if agg.score > best_score:
                best_score = agg.score
                best_image = restored
                best_iter = i

            result.iterations.append(iter_rec)
            result.total_iterations = i + 1

            logger.info("[%s] iter#%d: quality=%.3f(%s) safety=%.3f(%s) agg=%.3f(%s)",
                        case_id, i, qv.score, "P" if qv.passed else "F",
                        sv.score, "P" if sv.passed else "F",
                        agg.score, "P" if agg.passed else "F")

            if agg.passed:
                result.exit_reason = f"Passed at iteration #{i}"
                break
        else:
            result.exit_reason = f"Reached max_iterations ({self.max_iterations})"

        result.best_iteration = best_iter
        result.best_score = best_score

        # 5. Diagnosis on best image
        diag = self.diagnosis_adapter.predict(best_image)
        result.diagnosis = {
            "prediction": diag.prediction,
            "confidence": diag.confidence,
            "metadata": diag.metadata,
        }
        logger.info("[%s] Final diagnosis (from iter#%d): %s (conf=%.2f)",
                    case_id, best_iter, diag.prediction, diag.confidence)

        return result

    def run_toy_case(self, case: dict[str, Any]) -> ClosedLoopResult:
        result = self.run(
            image=case["degraded"],
            case_id=case.get("case_id", "unknown"),
        )

        label = case.get("label")
        if label is not None:
            label_dict = label.to_dict() if hasattr(label, "to_dict") else label
            result.label = label_dict
            gt_present = label_dict.get("lesion_present", False)
            pred_present = result.diagnosis.get("metadata", {}).get("lesion_present", False)
            result.diagnosis_correct = bool(gt_present) == bool(pred_present)

        return result

    @staticmethod
    def _ensure_tools_registered() -> None:
        import src.tools.classical.denoise
        import src.tools.classical.sharpen
        import src.tools.classical.histogram
