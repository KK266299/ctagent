# ============================================================================
# 模块职责: 退化检测器 — 基于 IQA 指标判断退化类型和严重程度
#   支持检测: noise / blur / metal artifact / low resolution
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — degradation perception
#       CAPIQA (https://github.com/aaz-imran/capiqa) — CT 感知 IQA
#       IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.degradations.types import DegradationReport, DegradationType, Severity

_DEFAULT_CONFIG: dict[str, Any] = {
    "noise_thresholds": {"mild": 0.005, "moderate": 0.015, "severe": 0.030},
    "blur_thresholds": {"mild": 0.00015, "moderate": 0.0001, "severe": 0.00005},
    "metal_thresholds": {"mild": 0.0005, "moderate": 0.001, "severe": 0.002},
    "lowres_thresholds": {"mild": 0.00020, "moderate": 0.00010, "severe": 0.00005},
    "mu_water": 0.192,
}


class DegradationDetector:
    """基于 IQA 阈值的退化检测器。

    检测维度:
    - noise:   MAD-based 噪声水平估计
    - blur:    Laplacian 方差 (越低越模糊)
    - metal:   HU 极端值 (>3000 HU) 像素占比
    - lowres:  FFT 高频能量占比 (越低分辨率越差)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(_DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg
        self.noise_thresholds = cfg["noise_thresholds"]
        self.blur_thresholds = cfg["blur_thresholds"]
        self.metal_thresholds = cfg["metal_thresholds"]
        self.lowres_thresholds = cfg["lowres_thresholds"]
        self.mu_water: float = cfg["mu_water"]

    def detect(self, image: np.ndarray) -> DegradationReport:
        """分析图像退化情况。

        Args:
            image: 输入 CT 图像 (μ 值空间或 HU 空间均可)

        Returns:
            DegradationReport 包含检测到的退化列表和 IQA 分数
        """
        report = DegradationReport()

        noise_level = self._estimate_noise(image)
        report.iqa_scores["noise_level"] = noise_level
        self._classify(
            report, noise_level, self.noise_thresholds,
            DegradationType.NOISE, higher_is_worse=True,
        )

        sharpness = self._estimate_sharpness(image)
        report.iqa_scores["sharpness"] = sharpness
        self._classify(
            report, sharpness, self.blur_thresholds,
            DegradationType.BLUR, higher_is_worse=False,
        )

        metal_ratio = self._estimate_metal_artifact(image)
        report.iqa_scores["metal_ratio"] = metal_ratio
        self._classify(
            report, metal_ratio, self.metal_thresholds,
            DegradationType.ARTIFACT_METAL, higher_is_worse=True,
        )

        hf_ratio = self._estimate_resolution(image)
        report.iqa_scores["hf_energy_ratio"] = hf_ratio
        self._classify(
            report, hf_ratio, self.lowres_thresholds,
            DegradationType.LOW_RESOLUTION, higher_is_worse=False,
        )

        return report

    @staticmethod
    def _classify(
        report: DegradationReport,
        value: float,
        thresholds: dict[str, float],
        deg_type: DegradationType,
        higher_is_worse: bool,
    ) -> None:
        mild = thresholds["mild"]
        moderate = thresholds["moderate"]
        severe = thresholds["severe"]

        if higher_is_worse:
            if value > severe:
                report.degradations.append((deg_type, Severity.SEVERE))
            elif value > moderate:
                report.degradations.append((deg_type, Severity.MODERATE))
            elif value > mild:
                report.degradations.append((deg_type, Severity.MILD))
        else:
            if value < severe:
                report.degradations.append((deg_type, Severity.SEVERE))
            elif value < moderate:
                report.degradations.append((deg_type, Severity.MODERATE))
            elif value < mild:
                report.degradations.append((deg_type, Severity.MILD))

    def _estimate_noise(self, image: np.ndarray) -> float:
        """MAD-based 噪声水平估计。"""
        from scipy.ndimage import laplace

        laplacian = laplace(image.astype(np.float64))
        sigma = np.median(np.abs(laplacian)) * 1.4826 / np.sqrt(2)
        return float(sigma)

    @staticmethod
    def _estimate_sharpness(image: np.ndarray) -> float:
        """Laplacian variance — 值越大越清晰。"""
        from scipy.ndimage import laplace

        lap = laplace(image.astype(np.float64))
        return float(np.var(lap))

    def _estimate_metal_artifact(self, image: np.ndarray) -> float:
        """HU 极端值像素占比 — 检测金属植入物。

        将 μ 转为 HU, 统计 |HU| > 3000 的像素比例。
        若输入已经是 HU 空间 (值域 [-1000, 4000+]), 自动适配。
        """
        arr = image.astype(np.float64)
        if arr.max() < 10:
            hu = (arr / self.mu_water - 1.0) * 1000.0
        else:
            hu = arr

        extreme_mask = np.abs(hu) > 3000
        total_pixels = hu.size
        if total_pixels == 0:
            return 0.0
        return float(np.sum(extreme_mask) / total_pixels)

    @staticmethod
    def _estimate_resolution(image: np.ndarray) -> float:
        """FFT 高频能量占比 — 值越低分辨率越差。"""
        f = np.fft.fft2(image.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4

        total_energy = np.sum(magnitude ** 2)
        if total_energy < 1e-10:
            return 0.0

        center_mask = np.zeros_like(magnitude, dtype=bool)
        Y, X = np.ogrid[:h, :w]
        center_mask[(Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2] = True

        lf_energy = np.sum(magnitude[center_mask] ** 2)
        hf_energy = total_energy - lf_energy
        return float(hf_energy / total_energy)
