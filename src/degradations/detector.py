# ============================================================================
# 模块职责: 退化检测器 — 基于 IQA 指标判断退化类型和严重程度
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — degradation perception
#       CAPIQA (https://github.com/aaz-imran/capiqa) — CT 感知 IQA
#       IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.degradations.types import DegradationReport, DegradationType, Severity


class DegradationDetector:
    """基于 IQA 阈值的退化检测器（第一版 MVP）。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        # 默认噪声阈值 (基于标准差估计)
        self.noise_thresholds = self.config.get(
            "noise_thresholds", {"mild": 15, "moderate": 30, "severe": 50}
        )

    def detect(self, image: np.ndarray) -> DegradationReport:
        """分析图像退化情况。

        Args:
            image: 输入 CT 图像 (HU 或归一化值)

        Returns:
            DegradationReport 包含检测到的退化列表
        """
        report = DegradationReport()
        # 噪声估计
        noise_level = self._estimate_noise(image)
        report.iqa_scores["noise_level"] = noise_level
        if noise_level > self.noise_thresholds["severe"]:
            report.degradations.append((DegradationType.NOISE, Severity.SEVERE))
        elif noise_level > self.noise_thresholds["moderate"]:
            report.degradations.append((DegradationType.NOISE, Severity.MODERATE))
        elif noise_level > self.noise_thresholds["mild"]:
            report.degradations.append((DegradationType.NOISE, Severity.MILD))

        # TODO: 添加更多退化检测 (blur, artifact, resolution)
        return report

    def _estimate_noise(self, image: np.ndarray) -> float:
        """基于 MAD 的噪声水平估计。"""
        from scipy.ndimage import laplace

        laplacian = laplace(image.astype(np.float64))
        sigma = np.median(np.abs(laplacian)) * 1.4826 / np.sqrt(2)
        return float(sigma)
