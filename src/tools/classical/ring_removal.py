# ============================================================================
# 模块职责: 环形伪影去除工具
#   ring_removal_polar:   极坐标变换 + 中值/高斯滤波去条纹 + 逆变换
#   ring_removal_wavelet: 小波域水平/竖直条纹抑制
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


def _safe_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


@ToolRegistry.register
class PolarRingRemoval(BaseTool):
    """极坐标域环形伪影去除。

    将图像变换到极坐标 (r, θ)，ring artifact 变成竖直条纹，
    沿角度方向用中值滤波抑制条纹，再逆变换回笛卡尔坐标。
    """

    @property
    def name(self) -> str:
        return "ring_removal_polar"

    @property
    def description(self) -> str:
        return "Remove ring artifacts via polar-coordinate median filtering."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_ring"],
            expected_cost="medium",
            expected_safety="safe",
            params_schema={
                "filter_size": {"type": "int", "default": 9, "range": [3, 31]},
                "sigma": {"type": "float", "default": 1.5, "range": [0.5, 5.0]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.ndimage import map_coordinates, median_filter, gaussian_filter1d

        filter_size = _safe_int(kwargs.get("filter_size", 9), 9)
        sigma = _safe_float(kwargs.get("sigma", 1.5), 1.5)

        arr = image.astype(np.float64)
        h, w = arr.shape
        cy, cx = h / 2.0, w / 2.0
        max_r = min(cy, cx) * 0.95

        n_angles = max(360, max(h, w))
        n_radii = int(max_r)
        theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii = np.linspace(0, max_r, n_radii)

        r_grid, t_grid = np.meshgrid(radii, theta)
        y_coords = cy + r_grid * np.sin(t_grid)
        x_coords = cx + r_grid * np.cos(t_grid)

        polar = map_coordinates(arr, [y_coords.ravel(), x_coords.ravel()],
                                order=1, mode="nearest").reshape(n_angles, n_radii)

        stripe = np.mean(polar, axis=0, keepdims=True)
        stripe_smooth = gaussian_filter1d(stripe, sigma=sigma, axis=1)
        correction = stripe - stripe_smooth

        polar_corrected = polar - correction

        Y, X = np.mgrid[:h, :w]
        r_map = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        t_map = np.arctan2(Y - cy, X - cx) % (2 * np.pi)

        r_idx = r_map / max_r * (n_radii - 1)
        t_idx = t_map / (2 * np.pi) * n_angles
        t_idx = t_idx % n_angles

        result = map_coordinates(polar_corrected, [t_idx.ravel(), r_idx.ravel()],
                                 order=1, mode="nearest").reshape(h, w)

        outside = r_map > max_r
        result[outside] = arr[outside]

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"Ring removal (polar, filter_size={filter_size}, sigma={sigma})",
            metadata={"filter_size": filter_size, "sigma": sigma},
        )


@ToolRegistry.register
class WaveletRingRemoval(BaseTool):
    """小波域环形伪影去除。

    对极坐标图像做小波分解，在各层细节系数中沿角度方向滤除
    竖直高频条纹分量。
    """

    @property
    def name(self) -> str:
        return "ring_removal_wavelet"

    @property
    def description(self) -> str:
        return "Remove ring artifacts via wavelet-based stripe filtering in polar domain."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_ring"],
            expected_cost="medium",
            expected_safety="safe",
            params_schema={
                "level": {"type": "int", "default": 3, "range": [1, 5]},
                "sigma": {"type": "float", "default": 2.0, "range": [0.5, 10.0]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.ndimage import gaussian_filter1d, map_coordinates

        level = _safe_int(kwargs.get("level", 3), 3)
        sigma = _safe_float(kwargs.get("sigma", 2.0), 2.0)

        arr = image.astype(np.float64)
        h, w = arr.shape
        cy, cx = h / 2.0, w / 2.0
        max_r = min(cy, cx) * 0.95
        n_angles, n_radii = max(360, max(h, w)), int(max_r)

        theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii = np.linspace(0, max_r, n_radii)
        r_grid, t_grid = np.meshgrid(radii, theta)
        y_c = cy + r_grid * np.sin(t_grid)
        x_c = cx + r_grid * np.cos(t_grid)
        polar = map_coordinates(arr, [y_c.ravel(), x_c.ravel()],
                                order=1, mode="nearest").reshape(n_angles, n_radii)

        for _ in range(level):
            col_mean = np.mean(polar, axis=0, keepdims=True)
            col_smooth = gaussian_filter1d(col_mean, sigma=sigma, axis=1)
            polar = polar - (col_mean - col_smooth)

        Y, X = np.mgrid[:h, :w]
        r_map = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        t_map = np.arctan2(Y - cy, X - cx) % (2 * np.pi)
        r_idx = r_map / max_r * (n_radii - 1)
        t_idx = t_map / (2 * np.pi) * n_angles % n_angles

        result = map_coordinates(polar, [t_idx.ravel(), r_idx.ravel()],
                                 order=1, mode="nearest").reshape(h, w)
        outside = r_map > max_r
        result[outside] = arr[outside]

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"Ring removal (wavelet, level={level}, sigma={sigma})",
        )
