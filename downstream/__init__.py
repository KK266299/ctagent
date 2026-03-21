# ============================================================================
# 模块职责: 下游诊断任务 — 顶层 bridge + MockDiagnosis
#   re-export src/downstream 已有模块 + 暴露新增 MockDiagnosis
# 参考: src/downstream/ — BaseDownstreamTask / ClosedAPIAdapter
# ============================================================================

from src.downstream.base import BaseDownstreamTask, DiagnosisResult
from src.downstream.closed_api_adapter import ClosedAPIAdapter, APIConfig
from downstream.mock_diagnosis import MockDiagnosis

__all__ = [
    "BaseDownstreamTask",
    "DiagnosisResult",
    "ClosedAPIAdapter",
    "APIConfig",
    "MockDiagnosis",
]
