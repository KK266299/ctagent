# ============================================================================
# 模块职责: 下游任务基类 — 统一的诊断任务接口
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
# ============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DiagnosisResult:
    """诊断结果。"""
    prediction: Any = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDownstreamTask(ABC):
    """下游诊断任务基类。"""

    @property
    @abstractmethod
    def task_name(self) -> str:
        ...

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs: Any) -> DiagnosisResult:
        """对图像执行诊断预测。"""
        ...
