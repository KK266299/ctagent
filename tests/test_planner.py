# ============================================================================
# 模块职责: Planner 模块单元测试
# ============================================================================
from src.degradations.types import DegradationReport, DegradationType, Severity
from src.planner.rule_planner import RuleBasedPlanner


def test_rule_planner_noise():
    """噪声退化应产生去噪工具计划。"""
    planner = RuleBasedPlanner()
    report = DegradationReport(degradations=[(DegradationType.NOISE, Severity.MILD)])
    plan = planner.plan(report)
    assert len(plan) > 0
    assert "denoise_nlm" in plan.tool_names()


def test_rule_planner_no_degradation():
    """无退化应产生空计划。"""
    planner = RuleBasedPlanner()
    report = DegradationReport()
    plan = planner.plan(report)
    assert len(plan) == 0


def test_rule_planner_max_chain():
    """超过 max_chain 应截断。"""
    planner = RuleBasedPlanner(max_chain=1)
    report = DegradationReport(degradations=[
        (DegradationType.NOISE, Severity.MILD),
        (DegradationType.BLUR, Severity.MILD),
    ])
    plan = planner.plan(report)
    assert len(plan) <= 1
