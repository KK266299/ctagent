# ============================================================================
# 模块职责: 下游诊断任务 — 修复后图像的分类/检测/分割
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — diagnosis
#       MedQ-Bench (https://github.com/liujiyaoFDU/MedQ-Bench)
# ============================================================================

from src.downstream.base import BaseDownstreamTask, DiagnosisResult
from src.downstream.classifier import CTClassifier
from src.downstream.closed_api_adapter import ClosedAPIAdapter, APIConfig
from src.downstream.prompt_builder import PromptBuilder
from src.downstream.response_parser import ResponseParser

__all__ = [
    "BaseDownstreamTask",
    "DiagnosisResult",
    "CTClassifier",
    "ClosedAPIAdapter",
    "APIConfig",
    "PromptBuilder",
    "ResponseParser",
]
