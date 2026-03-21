# ============================================================================
# 模块职责: 单次线性 Pipeline — 串联完整的一遍处理流程 (无 replan)
#   input → assessment → planner → executor → judge → diagnosis → report
#   新增 diagnose_only(): 纯诊断 baseline, 用于三模式对比
#   所有模块通过构造器注入，默认使用 rule-based + mock 配置
# 参考: 4KAgent — single-pass evaluation pipeline
#       JarvisIR — pipeline runner
#       AgenticIR — pipeline/run.py
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
from pipeline.types import PipelineState

logger = logging.getLogger(__name__)


@dataclass
class SinglePassResult:
    """单次 pipeline 的完整结果。"""
    case_id: str = ""
    mode: str = "restored"
    degradation_report: dict[str, Any] = field(default_factory=dict)
    plan_reasoning: str = ""
    plan_tools: list[str] = field(default_factory=list)
    trace: ExecutionTrace | None = None
    quality_verdict: dict[str, Any] = field(default_factory=dict)
    safety_verdict: dict[str, Any] = field(default_factory=dict)
    aggregate_verdict: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    label: dict[str, Any] = field(default_factory=dict)
    diagnosis_correct: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "degradation_report": self.degradation_report,
            "plan": {"reasoning": self.plan_reasoning, "tools": self.plan_tools},
            "trace": self.trace.to_dict() if self.trace else {},
            "quality_verdict": self.quality_verdict,
            "safety_verdict": self.safety_verdict,
            "aggregate_verdict": self.aggregate_verdict,
            "diagnosis": self.diagnosis,
            "label": self.label,
            "diagnosis_correct": self.diagnosis_correct,
        }

    def summary(self) -> str:
        lines = [
            f"Case: {self.case_id} [{self.mode}]",
        ]
        if self.plan_tools:
            lines.append(f"  Plan: {' -> '.join(self.plan_tools)} ({self.plan_reasoning})")
        if self.trace:
            lines.append(f"  Trace: {self.trace.num_success}ok/{self.trace.num_failed}fail, {self.trace.total_elapsed_ms:.0f}ms")
        if self.quality_verdict:
            lines.append(f"  Quality: {'PASS' if self.quality_verdict.get('passed') else 'FAIL'} (score={self.quality_verdict.get('score', 'N/A')})")
        if self.safety_verdict:
            lines.append(f"  Safety:  {'PASS' if self.safety_verdict.get('passed') else 'FAIL'} (score={self.safety_verdict.get('score', 'N/A')})")
        lines.append(f"  Diagnosis: {self.diagnosis.get('prediction', 'N/A')} (conf={self.diagnosis.get('confidence', 'N/A')})")
        if self.diagnosis_correct is not None:
            lines.append(f"  Correct: {self.diagnosis_correct}")
        return "\n".join(lines)


class SinglePassPipeline:
    """单次线性 Pipeline。

    Usage:
        pipeline = SinglePassPipeline()
        result = pipeline.run(degraded_image)           # full pipeline
        result = pipeline.run_toy_case(case)             # full + label comparison
        result = pipeline.diagnose_only(image, ...)      # baseline: diagnosis only
    """

    def __init__(
        self,
        detector: DegradationDetector | None = None,
        planner: Any | None = None,
        executor: Executor | None = None,
        quality_judge: QualityJudge | None = None,
        safety_judge: SafetyJudge | None = None,
        diagnosis_adapter: Any | None = None,
    ) -> None:
        self.detector = detector or DegradationDetector()
        self.planner = planner or RuleBasedPlanner()
        self.executor = executor or Executor()
        self.quality_judge = quality_judge or QualityJudge()
        self.safety_judge = safety_judge or SafetyJudge()
        self.diagnosis_adapter = diagnosis_adapter or MockDiagnosis()
        self._ensure_tools_registered()

    def diagnose_only(
        self,
        image: np.ndarray,
        case_id: str = "",
        mode: str = "clean",
        label: Any | None = None,
    ) -> SinglePassResult:
        """纯诊断 (不做检测/规划/执行/评判), 用于 baseline 对比。"""
        result = SinglePassResult(case_id=case_id, mode=mode)

        diag = self.diagnosis_adapter.predict(image)
        result.diagnosis = {
            "prediction": diag.prediction,
            "confidence": diag.confidence,
            "metadata": diag.metadata,
        }

        if label is not None:
            label_dict = label.to_dict() if hasattr(label, "to_dict") else label
            result.label = label_dict
            gt_present = label_dict.get("lesion_present", False)
            pred_present = result.diagnosis.get("metadata", {}).get("lesion_present", False)
            result.diagnosis_correct = bool(gt_present) == bool(pred_present)

        return result

    def run(
        self,
        image: np.ndarray,
        case_id: str = "",
        reference: np.ndarray | None = None,
    ) -> SinglePassResult:
        """执行单次完整 pipeline。"""
        result = SinglePassResult(case_id=case_id, mode="restored")

        # 1. Assessment
        report = self.detector.detect(image)
        result.degradation_report = {
            "degradations": [
                {"type": d.value, "severity": s.value}
                for d, s in report.degradations
            ],
            "iqa_scores": report.iqa_scores,
        }
        logger.info("[%s] Degradation: %s", case_id, result.degradation_report["degradations"])

        # 2. Plan
        plan = self.planner.plan(report)
        result.plan_reasoning = plan.reasoning
        result.plan_tools = plan.tool_names()
        logger.info("[%s] Plan: %s", case_id, result.plan_tools)

        # 3. Execute
        if plan.steps:
            tool_results = self.executor.execute(plan, image)
            result.trace = self.executor.last_trace
            restored = tool_results[-1].image if tool_results and tool_results[-1].success else image
        else:
            restored = image
            result.trace = None
            logger.info("[%s] No tools to execute", case_id)

        # 4. Judge
        qv = self.quality_judge.judge_no_reference(image, restored)
        sv = self.safety_judge.judge(image, restored)
        agg = aggregate_verdicts([
            JudgeVerdict(passed=qv.passed, score=qv.score, reason=qv.reason, judge_type="quality"),
            sv,
        ])
        result.quality_verdict = {"passed": qv.passed, "score": qv.score, "reason": qv.reason}
        result.safety_verdict = sv.to_dict()
        result.aggregate_verdict = agg.to_dict()

        # 5. Diagnosis
        diag = self.diagnosis_adapter.predict(restored)
        result.diagnosis = {
            "prediction": diag.prediction,
            "confidence": diag.confidence,
            "metadata": diag.metadata,
        }
        logger.info("[%s] Diagnosis: %s (conf=%.2f)", case_id, diag.prediction, diag.confidence)

        return result

    def run_toy_case(self, case: dict[str, Any]) -> SinglePassResult:
        """运行一个 toy case (含 label 比对)。"""
        result = self.run(
            image=case["degraded"],
            case_id=case.get("case_id", "unknown"),
            reference=case.get("clean"),
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
