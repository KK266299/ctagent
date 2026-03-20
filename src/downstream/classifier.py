# ============================================================================
# 模块职责: CT 图像分类器 — 下游分类诊断任务
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
#       MedQ-Bench (https://github.com/liujiyaoFDU/MedQ-Bench)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.downstream.base import BaseDownstreamTask, DiagnosisResult


class CTClassifier(BaseDownstreamTask):
    """CT 图像分类器（占位）。"""

    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.model = None

    @property
    def task_name(self) -> str:
        return "ct_classification"

    def predict(self, image: np.ndarray, **kwargs: Any) -> DiagnosisResult:
        # TODO: 加载并运行分类模型
        return DiagnosisResult(
            prediction=None,
            confidence=0.0,
            metadata={"message": "Classifier placeholder — model not loaded"},
        )
