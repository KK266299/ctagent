# ============================================================================
# 模块职责: 无参考 IQA — 不需要 ground truth 的质量评估
# 参考: IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch) — NR-IQA
#       CAPIQA (https://github.com/aaz-imran/capiqa) — CT 无参考 IQA
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np


class NoReferenceIQA:
    """无参考图像质量评估。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def compute_sharpness(self, image: np.ndarray) -> float:
        """基于 Laplacian 方差的清晰度。"""
        from scipy.ndimage import laplace

        lap = laplace(image.astype(np.float64))
        return float(np.var(lap))

    def compute_noise_estimate(self, image: np.ndarray) -> float:
        """MAD-based 噪声估计。"""
        from scipy.ndimage import laplace

        lap = laplace(image.astype(np.float64))
        return float(np.median(np.abs(lap)) * 1.4826)

    def evaluate(self, image: np.ndarray) -> dict[str, float]:
        """计算所有无参考指标。"""
        return {
            "sharpness": self.compute_sharpness(image),
            "noise_estimate": self.compute_noise_estimate(image),
        }
