# ============================================================================
# 模块职责: 下游诊断任务 — 修复后图像的分类/检测/分割
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — diagnosis
#       MedQ-Bench (https://github.com/liujiyaoFDU/MedQ-Bench)
# ============================================================================

from src.downstream.base import BaseDownstreamTask
from src.downstream.classifier import CTClassifier

__all__ = ["BaseDownstreamTask", "CTClassifier"]
