# ============================================================================
# 模块职责: 退化检测器 — 基于 IQA 指标判断退化类型和严重程度
#   支持检测: noise / blur / metal artifact / low resolution
#             ring / motion / beam_hardening / scatter / truncation artifact
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
    "lowres_thresholds": {"mild": 0.00010, "moderate": 0.00005, "severe": 0.00002},
    # --- CT artifact detection thresholds (recalibrated v2) ---
    # Raised motion/scatter/lowres mild thresholds to reduce false positives
    # on near-clean images. Previous thresholds triggered on almost everything.
    "ring_thresholds": {"mild": 0.45, "moderate": 0.80, "severe": 1.05},
    "motion_thresholds": {"mild": 0.80, "moderate": 0.95, "severe": 1.10},
    "beam_hardening_thresholds": {"mild": 1.10, "moderate": 1.17, "severe": 1.22},
    "scatter_thresholds": {"mild": 0.12, "moderate": 0.18, "severe": 0.25},
    "truncation_thresholds": {"mild": 0.10, "moderate": 0.20, "severe": 0.35},
    "mu_water": 0.192,
}


class DegradationDetector:
    """基于 IQA 阈值的退化检测器。

    检测维度:
    - noise:            MAD-based 噪声水平估计
    - blur:             Laplacian 方差 (越低越模糊)
    - metal:            HU 极端值 (>3000 HU) 像素占比
    - lowres:           FFT 高频能量占比 (越低分辨率越差)
    - ring:             极坐标域角度方向频谱峰值 (同心环纹)
    - motion:           方向梯度各向异性 (运动模糊/鬼影)
    - beam_hardening:   中心-边缘亮度差异比 (杯状效应)
    - scatter:          全局对比度损失指标
    - truncation:       边缘亮度异常比
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
        self.ring_thresholds = cfg["ring_thresholds"]
        self.motion_thresholds = cfg["motion_thresholds"]
        self.beam_hardening_thresholds = cfg["beam_hardening_thresholds"]
        self.scatter_thresholds = cfg["scatter_thresholds"]
        self.truncation_thresholds = cfg["truncation_thresholds"]
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

        # --- CT artifact detectors ---
        ring_score = self._estimate_ring_artifact(image)
        report.iqa_scores["ring_score"] = ring_score
        self._classify(
            report, ring_score, self.ring_thresholds,
            DegradationType.ARTIFACT_RING, higher_is_worse=True,
        )

        motion_score = self._estimate_motion_artifact(image)
        report.iqa_scores["motion_score"] = motion_score
        self._classify(
            report, motion_score, self.motion_thresholds,
            DegradationType.ARTIFACT_MOTION, higher_is_worse=True,
        )

        bh_score = self._estimate_beam_hardening(image)
        report.iqa_scores["beam_hardening_score"] = bh_score
        self._classify(
            report, bh_score, self.beam_hardening_thresholds,
            DegradationType.ARTIFACT_BEAM_HARDENING, higher_is_worse=True,
        )

        scatter_score = self._estimate_scatter(image)
        report.iqa_scores["scatter_score"] = scatter_score
        self._classify(
            report, scatter_score, self.scatter_thresholds,
            DegradationType.ARTIFACT_SCATTER, higher_is_worse=True,
        )

        truncation_score = self._estimate_truncation(image)
        report.iqa_scores["truncation_score"] = truncation_score
        self._classify(
            report, truncation_score, self.truncation_thresholds,
            DegradationType.ARTIFACT_TRUNCATION, higher_is_worse=True,
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
        """HU 极端值像素占比 — 检测金属植入物。"""
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

    # ================================================================
    # CT artifact detectors
    # ================================================================

    @staticmethod
    def _estimate_ring_artifact(image: np.ndarray) -> float:
        """检测环形伪影 — 极坐标径向梯度的角度一致性。

        Ring artifact 在极坐标中表现为：特定半径处，所有角度的值
        几乎相同（因为是完整的环）。检测方法：
        1. 变换到极坐标
        2. 计算径向差分（相邻半径之差）
        3. 对每个半径，计算角度方向的标准差/均值比
        4. 如果某些半径的角度一致性极高（std/mean 极低），说明有环
        """
        from scipy.ndimage import map_coordinates

        arr = image.astype(np.float64)
        h, w = arr.shape
        cy, cx = h / 2.0, w / 2.0
        max_r = min(cy, cx) * 0.7

        n_angles, n_radii = 360, int(max_r)
        if n_radii < 20:
            return 0.0

        theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii = np.linspace(10, max_r, n_radii)
        r_grid, t_grid = np.meshgrid(radii, theta)
        y_coords = cy + r_grid * np.sin(t_grid)
        x_coords = cx + r_grid * np.cos(t_grid)

        polar = map_coordinates(arr, [y_coords.ravel(), x_coords.ravel()],
                                order=1, mode="constant", cval=0.0)
        polar = polar.reshape(n_angles, n_radii)

        radial_diff = np.diff(polar, axis=1)

        angular_mean = np.mean(radial_diff, axis=0)
        angular_std = np.std(radial_diff, axis=0)
        abs_mean = np.abs(angular_mean)

        valid = abs_mean > 1e-8
        if np.sum(valid) < 5:
            return 0.0

        consistency = np.zeros_like(abs_mean)
        consistency[valid] = abs_mean[valid] / (angular_std[valid] + 1e-10)

        top_k = min(10, len(consistency) // 4)
        sorted_c = np.sort(consistency)[::-1]
        score = np.mean(sorted_c[:top_k])
        baseline = np.median(consistency)

        return float(max(0.0, score - baseline))

    @staticmethod
    def _estimate_motion_artifact(image: np.ndarray) -> float:
        """检测运动伪影 — FFT 方向性分析。

        运动伪影在 Fourier 域中造成特定方向的能量集中（条纹），
        通过计算 FFT 幅度的方向性展开来检测。
        """
        arr = image.astype(np.float64)
        h, w = arr.shape
        cy, cx = h // 2, w // 2

        f = np.fft.fft2(arr)
        fshift = np.abs(np.fft.fftshift(f))
        fshift[cy, cx] = 0

        n_dirs = 36
        dir_energies = np.zeros(n_dirs)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        angle_map = np.arctan2(Y - cy, X - cx)

        band_mask = (dist > 5) & (dist < min(cy, cx) * 0.8)

        for di in range(n_dirs):
            a0 = -np.pi + di * (2 * np.pi / n_dirs)
            a1 = a0 + (2 * np.pi / n_dirs)
            ang_mask = (angle_map >= a0) & (angle_map < a1)
            m = band_mask & ang_mask
            if np.sum(m) > 0:
                dir_energies[di] = np.sum(fshift[m] ** 2)

        total = np.sum(dir_energies)
        if total < 1e-10:
            return 0.0

        normalized = dir_energies / total
        uniformity = 1.0 / n_dirs

        anisotropy = np.max(normalized) / max(uniformity, 1e-10) - 1.0
        return float(max(0.0, anisotropy))

    def _estimate_beam_hardening(self, image: np.ndarray) -> float:
        """检测射束硬化伪影 — 杯状效应（cupping）幅度。

        将 body 区域分成同心环带，拟合径向亮度变化的二次多项式，
        二次项系数反映杯状效应强度。
        """
        arr = image.astype(np.float64)
        h, w = arr.shape
        cy, cx = h // 2, w // 2

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        max_r = min(cy, cx) * 0.85

        body_mask = arr > (self.mu_water * 0.3)
        if np.sum(body_mask) < 100:
            return 0.0

        n_rings = 10
        ring_means = []
        ring_radii = []
        for i in range(n_rings):
            r_lo = max_r * i / n_rings
            r_hi = max_r * (i + 1) / n_rings
            ring_mask = body_mask & (dist >= r_lo) & (dist < r_hi)
            if np.sum(ring_mask) < 10:
                continue
            ring_means.append(np.mean(arr[ring_mask]))
            ring_radii.append((r_lo + r_hi) / 2.0)

        if len(ring_radii) < 4:
            return 0.0

        r_norm = np.array(ring_radii) / max_r
        means = np.array(ring_means)

        mean_val = np.mean(means)
        if mean_val < 1e-10:
            return 0.0

        means_norm = means / mean_val
        coeffs = np.polyfit(r_norm, means_norm, 2)
        curvature = abs(coeffs[0])

        return float(curvature)

    def _estimate_scatter(self, image: np.ndarray) -> float:
        """检测散射伪影 — 软组织区域对比度/均匀性分析。

        Scatter 使整体值升高、对比度降低。在 body 区域内，
        scatter 使低密度区（脑实质）和高密度区（骨）的比值降低。
        """
        arr = image.astype(np.float64)

        mu_air_thresh = self.mu_water * 0.1
        mu_soft = self.mu_water
        mu_bone_thresh = self.mu_water * 3.0

        body = arr[arr > mu_air_thresh]
        if body.size < 100:
            return 0.0

        soft_mask = (arr > mu_soft * 0.5) & (arr < mu_soft * 2.0)
        bone_mask = arr > mu_bone_thresh

        if np.sum(soft_mask) < 50 or np.sum(bone_mask) < 10:
            p10 = np.percentile(body, 10)
            p90 = np.percentile(body, 90)
            if p90 < 1e-10:
                return 0.0
            dynamic_range = (p90 - p10) / p90
            loss = max(0.0, 1.0 - dynamic_range / 0.6)
            return float(loss)

        soft_mean = np.mean(arr[soft_mask])
        bone_mean = np.mean(arr[bone_mask])

        if bone_mean < 1e-10:
            return 0.0

        contrast_ratio = (bone_mean - soft_mean) / bone_mean
        expected_ratio = 0.7
        loss = max(0.0, 1.0 - contrast_ratio / expected_ratio)
        return float(loss)

    @staticmethod
    def _estimate_truncation(image: np.ndarray) -> float:
        """检测截断伪影 — FOV 圆形边界处亮度异常。

        Truncation artifact 在 FBP 重建 FOV 的圆形边界处产生亮带。
        检测方法：分析距图像中心特定半径范围内的亮度异常。
        """
        arr = image.astype(np.float64)
        h, w = arr.shape
        cy, cx = h / 2.0, w / 2.0

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        max_r = min(cy, cx)

        boundary_mask = (dist > max_r * 0.75) & (dist < max_r * 0.95)
        inner_mask = (dist > max_r * 0.2) & (dist < max_r * 0.6)

        if np.sum(boundary_mask) < 50 or np.sum(inner_mask) < 50:
            return 0.0

        boundary_vals = arr[boundary_mask]
        inner_vals = arr[inner_mask]

        boundary_high = np.percentile(boundary_vals, 90)
        inner_high = np.percentile(inner_vals, 90)

        if inner_high < 1e-10:
            return 0.0

        ratio = boundary_high / inner_high
        score = max(0.0, ratio - 0.8)
        return float(score)
