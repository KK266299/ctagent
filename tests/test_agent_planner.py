# ============================================================================
# 模块职责: Agent-based planner 单元测试
# ============================================================================
import numpy as np

from src.degradations.types import DegradationReport, DegradationType, Severity
from src.planner.agent_based import AgentBasedPlanner
from src.planner.policy_rl_placeholder import RLPolicyPlanner


def _make_test_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((64, 64)).astype(np.float32)


class TestAgentBasedPlanner:
    def test_fallback_without_api(self):
        """无 api_caller 时应回退到规则。"""
        planner = AgentBasedPlanner(api_caller=None)
        report = DegradationReport(
            degradations=[(DegradationType.NOISE, Severity.MODERATE)]
        )
        plan = planner.plan(report)
        assert len(plan) > 0

    def test_collect_perceptions(self):
        """感知步骤应返回 3 种工具结果。"""
        planner = AgentBasedPlanner()
        perceptions = planner.collect_perceptions(_make_test_image())
        assert "analysis" in perceptions
        assert "perception" in perceptions
        assert "statistics" in perceptions

    def test_plan_with_mock_api(self):
        """使用 mock API 验证 agent 规划流程。"""
        import json
        mock_response = json.dumps({
            "reasoning": "Image has moderate noise",
            "steps": [{"tool_name": "denoise_nlm", "params": {}}],
        })
        mock_caller = lambda messages: mock_response  # noqa: E731
        planner = AgentBasedPlanner(api_caller=mock_caller)
        plan = planner.plan_with_perception(_make_test_image())
        assert len(plan) == 1
        assert plan.tool_names() == ["denoise_nlm"]
        assert "noise" in plan.reasoning.lower()

    def test_plan_with_bad_api_response(self):
        """API 返回无法解析时应回退。"""
        mock_caller = lambda messages: "I cannot parse this"  # noqa: E731
        planner = AgentBasedPlanner(api_caller=mock_caller)
        plan = planner.plan_with_perception(_make_test_image())
        # 应回退到 analysis-based plan (可能为空或有工具)
        assert isinstance(plan.reasoning, str)


class TestRLPolicyPlanner:
    def test_fallback_without_policy(self):
        """无 policy 时应回退到规则。"""
        planner = RLPolicyPlanner()
        report = DegradationReport(
            degradations=[(DegradationType.NOISE, Severity.SEVERE)]
        )
        plan = planner.plan(report)
        assert len(plan) > 0

    def test_extract_state(self):
        """状态提取应返回必需字段。"""
        planner = RLPolicyPlanner()
        report = DegradationReport(
            degradations=[(DegradationType.NOISE, Severity.MILD)],
            iqa_scores={"noise_level": 20.0},
        )
        state = planner.extract_state(_make_test_image(), report)
        assert "image_shape" in state
        assert "degradations" in state
        assert "iqa_scores" in state
