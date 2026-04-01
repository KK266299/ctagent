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
    # --- New artifact type thresholds (v3) ---
    "low_dose_thresholds": {"mild": 0.002, "moderate": 0.006, "severe": 0.015},
    "sparse_view_thresholds": {"mild": 2.5, "moderate": 5.0, "severe": 10.0},
    "limited_angle_thresholds": {"mild": 0.3, "moderate": 0.5, "severe": 0.7},
    "focal_spot_blur_thresholds": {"mild": 2.5, "moderate": 3.5, "severe": 5.0},
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
        self.low_dose_thresholds = cfg["low_dose_thresholds"]
        self.sparse_view_thresholds = cfg["sparse_view_thresholds"]
        self.limited_angle_thresholds = cfg["limited_angle_thresholds"]
        self.focal_spot_blur_thresholds = cfg["focal_spot_blur_thresholds"]
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

        # --- New artifact detectors (v3) ---
        low_dose_score = self._estimate_low_dose(image)
        report.iqa_scores["low_dose_score"] = low_dose_score
        self._classify(
            report, low_dose_score, self.low_dose_thresholds,
            DegradationType.LOW_DOSE, higher_is_worse=True,
        )

        sparse_view_score = self._estimate_sparse_view(image)
        report.iqa_scores["sparse_view_score"] = sparse_view_score
        self._classify(
            report, sparse_view_score, self.sparse_view_thresholds,
            DegradationType.ARTIFACT_SPARSE_VIEW, higher_is_worse=True,
        )

        limited_angle_score = self._estimate_limited_angle(image)
        report.iqa_scores["limited_angle_score"] = limited_angle_score
        self._classify(
            report, limited_angle_score, self.limited_angle_thresholds,
            DegradationType.ARTIFACT_LIMITED_ANGLE, higher_is_worse=True,
        )

        focal_spot_blur_score = self._estimate_focal_spot_blur(image)
        report.iqa_scores["focal_spot_blur_score"] = focal_spot_blur_score
        self._classify(
            report, focal_spot_blur_score, self.focal_spot_blur_thresholds,
            DegradationType.ARTIFACT_FOCAL_SPOT_BLUR, higher_is_worse=True,
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

    def _estimate_low_dose(self, image: np.ndarray) -> float:
        """检测低剂量噪声 — 信号依赖噪声分析 (Poisson 特征)。

        Poisson 噪声的方差与信号均值成正比 (var ≈ α·mean)。
        在多个不同信号水平的 ROI 中测量 (mean, variance)，
        拟合线性模型得到斜率 α，α 越大说明 Poisson 噪声越强。
        """
        from scipy.ndimage import uniform_filter

        arr = image.astype(np.float64)
        h, w = arr.shape

        # 局部均值和局部方差
        patch = 16
        local_mean = uniform_filter(arr, size=patch)
        local_sq_mean = uniform_filter(arr ** 2, size=patch)
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

        # 采样 body 区域内的点 (排除空气和极端值)
        body_mask = arr > (self.mu_water * 0.3)
        # 排除边缘
        margin = patch
        body_mask[:margin, :] = False
        body_mask[-margin:, :] = False
        body_mask[:, :margin] = False
        body_mask[:, -margin:] = False

        if np.sum(body_mask) < 200:
            return 0.0

        means = local_mean[body_mask]
        variances = local_var[body_mask]

        # 按信号水平分 bin 取中位数，减少异常值干扰
        n_bins = 10
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_means = []
        bin_vars = []
        for i in range(n_bins):
            lo = np.percentile(means, percentiles[i])
            hi = np.percentile(means, percentiles[i + 1])
            mask = (means >= lo) & (means < hi + 1e-10)
            if np.sum(mask) < 10:
                continue
            bin_means.append(np.median(means[mask]))
            bin_vars.append(np.median(variances[mask]))

        if len(bin_means) < 4:
            return 0.0

        bin_means = np.array(bin_means)
        bin_vars = np.array(bin_vars)

        # 线性拟合 var = α·mean + β
        if np.std(bin_means) < 1e-10:
            return 0.0
        coeffs = np.polyfit(bin_means, bin_vars, 1)
        alpha = max(0.0, coeffs[0])  # 斜率

        return float(alpha)

    @staticmethod
    def _estimate_sparse_view(image: np.ndarray) -> float:
        """检测稀疏视角伪影 — FFT 角向能量周期性。

        稀疏视角在 FFT 中产生角向周期性的能量分布 (view aliasing)，
        通过计算角向能量轮廓的 FFT 频谱峰值来检测。
        区别于 ring (角度一致性高) 和 motion (单方向能量集中)。
        """
        arr = image.astype(np.float64)
        h, w = arr.shape
        cy, cx = h // 2, w // 2

        f = np.fft.fft2(arr)
        fshift = np.abs(np.fft.fftshift(f))
        fshift[cy, cx] = 0

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        angle_map = np.arctan2(Y - cy, X - cx)

        # 中频带 (排除 DC 和极高频)
        r_min = min(cy, cx) * 0.1
        r_max = min(cy, cx) * 0.7
        band_mask = (dist > r_min) & (dist < r_max)

        n_angles = 180
        angular_energy = np.zeros(n_angles)
        for i in range(n_angles):
            a0 = -np.pi + i * (2 * np.pi / n_angles)
            a1 = a0 + (2 * np.pi / n_angles)
            ang_mask = (angle_map >= a0) & (angle_map < a1) & band_mask
            count = np.sum(ang_mask)
            if count > 0:
                angular_energy[i] = np.sum(fshift[ang_mask] ** 2) / count

        # 对角向能量轮廓做 FFT，检测周期性
        ae_centered = angular_energy - np.mean(angular_energy)
        ae_fft = np.abs(np.fft.rfft(ae_centered))
        if len(ae_fft) < 3:
            return 0.0

        # 跳过 DC (index 0)，取峰值与中位数的比值
        ae_fft_no_dc = ae_fft[1:]
        if np.median(ae_fft_no_dc) < 1e-10:
            return 0.0

        peak_ratio = np.max(ae_fft_no_dc) / np.median(ae_fft_no_dc)
        return float(max(0.0, peak_ratio - 1.0))

    @staticmethod
    def _estimate_limited_angle(image: np.ndarray) -> float:
        """检测有限角伪影 — FFT 缺失楔检测。

        有限角在 FFT 中形成一个连续的低能量扇区 (missing wedge)。
        将 FFT 分成角度扇区，寻找连续低能量区域。
        """
        arr = image.astype(np.float64)
        h, w = arr.shape
        cy, cx = h // 2, w // 2

        f = np.fft.fft2(arr)
        fshift = np.abs(np.fft.fftshift(f))
        fshift[cy, cx] = 0

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        angle_map = np.arctan2(Y - cy, X - cx)

        r_min = min(cy, cx) * 0.1
        r_max = min(cy, cx) * 0.7
        band_mask = (dist > r_min) & (dist < r_max)

        n_sectors = 36
        sector_energy = np.zeros(n_sectors)
        for i in range(n_sectors):
            a0 = -np.pi + i * (2 * np.pi / n_sectors)
            a1 = a0 + (2 * np.pi / n_sectors)
            sec_mask = (angle_map >= a0) & (angle_map < a1) & band_mask
            count = np.sum(sec_mask)
            if count > 0:
                sector_energy[i] = np.sum(fshift[sec_mask] ** 2) / count

        if np.median(sector_energy) < 1e-10:
            return 0.0

        # 归一化
        normed = sector_energy / np.median(sector_energy)

        # 寻找最大连续低能量区域 (阈值: 中位数的 50%)
        low_thresh = 0.5
        is_low = normed < low_thresh

        # 环形搜索 (首尾相连)
        doubled = np.concatenate([is_low, is_low])
        max_run = 0
        current_run = 0
        for v in doubled:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        max_run = min(max_run, n_sectors)  # 不超过一圈

        # 缺失楔占比
        missing_fraction = max_run / n_sectors
        return float(missing_fraction)

    def _estimate_focal_spot_blur(self, image: np.ndarray) -> float:
        """检测焦点模糊 — 各向同性边缘展宽。

        焦点/探测器模糊使所有方向的边缘均匀变宽 (各向同性)。
        通过测量多个边缘的剖面宽度来估计。
        区别于 motion blur (各向异性, 特定方向更宽)。
        """
        from scipy.ndimage import sobel, gaussian_filter1d

        arr = image.astype(np.float64)
        h, w = arr.shape

        # body 区域
        body_mask = arr > (self.mu_water * 0.3)
        if np.sum(body_mask) < 100:
            return 0.0

        # Sobel 梯度
        gx = sobel(arr, axis=1)
        gy = sobel(arr, axis=0)
        gmag = np.sqrt(gx ** 2 + gy ** 2)

        # 只在 body 内部取边缘 (排除图像边框)
        margin = 10
        inner_mask = np.zeros_like(body_mask)
        inner_mask[margin:-margin, margin:-margin] = True
        edge_mask = inner_mask & body_mask

        # 取强边缘点
        gmag_body = gmag[edge_mask]
        if len(gmag_body) < 100:
            return 0.0
        edge_thresh = np.percentile(gmag_body, 90)
        strong_edges = edge_mask & (gmag >= edge_thresh)

        edge_coords = np.argwhere(strong_edges)
        if len(edge_coords) < 20:
            return 0.0

        # 随机采样边缘点，测量法向剖面宽度
        rng = np.random.default_rng(42)
        n_samples = min(200, len(edge_coords))
        sample_idx = rng.choice(len(edge_coords), n_samples, replace=False)

        widths = []
        profile_half = 5  # 半径 5 像素

        for idx in sample_idx:
            py, px = edge_coords[idx]
            gx_val = gx[py, px]
            gy_val = gy[py, px]
            g_norm = np.sqrt(gx_val ** 2 + gy_val ** 2)
            if g_norm < 1e-8:
                continue

            # 法向方向 (垂直于边缘)
            nx = gx_val / g_norm
            ny = gy_val / g_norm

            # 沿法向采样剖面
            profile = []
            valid = True
            for t in range(-profile_half, profile_half + 1):
                sy = int(round(py + t * ny))
                sx = int(round(px + t * nx))
                if 0 <= sy < h and 0 <= sx < w:
                    profile.append(arr[sy, sx])
                else:
                    valid = False
                    break

            if not valid or len(profile) < 2 * profile_half + 1:
                continue

            profile = np.array(profile, dtype=np.float64)
            # 用梯度幅度估计边缘宽度 (梯度的半高全宽)
            grad_profile = np.abs(np.diff(profile))
            if np.max(grad_profile) < 1e-10:
                continue

            # 半高全宽近似
            half_max = np.max(grad_profile) / 2.0
            above_half = grad_profile >= half_max
            width = float(np.sum(above_half))
            widths.append(width)

        if len(widths) < 10:
            return 0.0

        avg_width = float(np.median(widths))
        return avg_width

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
