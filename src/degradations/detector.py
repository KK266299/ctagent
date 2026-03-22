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
        # 金属伪影检测阈值
        self.metal_thresholds = self.config.get(
            "metal_thresholds",
            {
                "hu_threshold": 3000,       # 金属区域 HU 阈值
                "streak_intensity": 0.15,   # 条纹强度阈值
                "metal_area_ratio": 0.001,  # 最小金属面积比
            },
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

        # 金属伪影检测
        metal_result = self._detect_metal_artifact(image)
        report.iqa_scores["metal_area_ratio"] = metal_result["area_ratio"]
        report.iqa_scores["streak_score"] = metal_result["streak_score"]
        if metal_result["detected"]:
            report.degradations.append(
                (DegradationType.ARTIFACT_METAL, metal_result["severity"])
            )

        # TODO: 添加更多退化检测 (blur, resolution)
        return report

    def _estimate_noise(self, image: np.ndarray) -> float:
        """基于 MAD 的噪声水平估计。"""
        from scipy.ndimage import laplace

        laplacian = laplace(image.astype(np.float64))
        sigma = np.median(np.abs(laplacian)) * 1.4826 / np.sqrt(2)
        return float(sigma)

    def _detect_metal_artifact(self, image: np.ndarray) -> dict[str, Any]:
        """检测金属伪影。

        通过以下特征判断:
        1. 高亮区域（疑似金属）面积比
        2. 从高亮区域辐射出的条纹模式强度

        Returns:
            包含 detected, severity, area_ratio, streak_score 的字典
        """
        img = image.astype(np.float64)
        hu_thresh = self.metal_thresholds["hu_threshold"]
        min_area = self.metal_thresholds["metal_area_ratio"]
        streak_thresh = self.metal_thresholds["streak_intensity"]

        # 1. 检测高亮区域（可能的金属）
        metal_mask = img > hu_thresh
        area_ratio = float(np.sum(metal_mask)) / max(img.size, 1)

        # 2. 条纹检测：金属伪影在角度方向产生辐射状条纹
        #    用 Laplacian 在非金属区域检测异常高频能量
        from scipy.ndimage import laplace

        non_metal = img.copy()
        non_metal[metal_mask] = np.median(img[~metal_mask]) if np.any(~metal_mask) else 0
        lap = laplace(non_metal)
        # 条纹分数：非金属区域的 Laplacian 标准差（归一化）
        img_range = np.ptp(img) if np.ptp(img) > 0 else 1.0
        streak_score = float(np.std(lap)) / img_range

        # 3. 判定
        detected = area_ratio >= min_area and streak_score >= streak_thresh
        if detected:
            if area_ratio > 0.01 or streak_score > 0.5:
                severity = Severity.SEVERE
            elif area_ratio > 0.005 or streak_score > 0.3:
                severity = Severity.MODERATE
            else:
                severity = Severity.MILD
        else:
            severity = Severity.MILD

        return {
            "detected": detected,
            "severity": severity,
            "area_ratio": area_ratio,
            "streak_score": streak_score,
        }
