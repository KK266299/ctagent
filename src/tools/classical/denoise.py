# ============================================================================
# 模块职责: 经典去噪工具 — 从快速粗糙到高质量边缘保持
#   denoise_gaussian:  Gaussian 滤波, 快但模糊边缘
#   denoise_bilateral: 双边滤波, 保边去噪, 适合 mild-moderate noise
#   denoise_tv:        Total Variation, 保边去噪, 适合 moderate-severe noise
#   denoise_nlm:       Non-Local Means, 结构感知, 适合纹理丰富区域
#   denoise_wiener:    Wiener 滤波, 频域去噪, 适合均匀高斯噪声
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR)
#       scikit-image restoration module
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
class GaussianDenoise(BaseTool):
    """Gaussian 滤波去噪 — 最快但最模糊边缘。

    适用: 快速预处理, 对边缘质量无要求的场景
    局限: structure_preservation 很低, 不适合诊断级修复
    """

    @property
    def name(self) -> str:
        return "denoise_gaussian"

    @property
    def description(self) -> str:
        return "Gaussian filter denoising, fast but may blur edges."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise"],
            expected_cost="cheap",
            expected_safety="moderate",
            params_schema={"sigma": {"type": "float", "default": 1.0, "range": [0.5, 5.0]}},
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.ndimage import gaussian_filter

        sigma = _safe_float(kwargs.get("sigma", 1.0), 1.0)
        denoised = gaussian_filter(image, sigma=sigma)
        return ToolResult(image=denoised.astype(np.float32), tool_name=self.name)


@ToolRegistry.register
class BilateralDenoise(BaseTool):
    """双边滤波去噪 — 边缘保持型去噪的首选。

    适用: mild-moderate Gaussian noise, 需要保持边缘和结构
    原理: 空间域 + 值域双重加权, 平滑区域去噪但保持强边缘
    优势: structure_preservation 显著优于 Gaussian
    """

    @property
    def name(self) -> str:
        return "denoise_bilateral"

    @property
    def description(self) -> str:
        return "Bilateral filter: edge-preserving denoising for mild-to-moderate noise."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise", "artifact"],
            expected_cost="medium",
            expected_safety="safe",
            params_schema={
                "sigma_color": {"type": "float", "default": 0.05, "range": [0.01, 0.2]},
                "sigma_spatial": {"type": "float", "default": 5, "range": [1, 15]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import denoise_bilateral

        sigma_color = _safe_float(kwargs.get("sigma_color", 0.05), 0.05)
        sigma_spatial = _safe_float(kwargs.get("sigma_spatial", 5), 5.0)
        denoised = denoise_bilateral(
            image.astype(np.float64),
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            channel_axis=None,
        )
        return ToolResult(
            image=denoised.astype(np.float32),
            tool_name=self.name,
            metadata={"sigma_color": sigma_color, "sigma_spatial": sigma_spatial},
        )


@ToolRegistry.register
class TVDenoise(BaseTool):
    """Total Variation 去噪 — 强边缘保持, 适合中重度噪声。

    适用: moderate-severe noise, 需要强保边
    原理: 最小化 total variation 正则, 趋向 piecewise-constant
    优势: 比 bilateral 更强的去噪能力, 仍保持清晰边缘
    局限: 可能产生 "staircasing" 效应 (对 CT 阶梯感可接受)
    """

    @property
    def name(self) -> str:
        return "denoise_tv"

    @property
    def description(self) -> str:
        return "Total Variation denoising: strong edge-preserving for moderate-severe noise."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise", "artifact"],
            expected_cost="medium",
            expected_safety="safe",
            params_schema={"weight": {"type": "float", "default": 0.1, "range": [0.01, 0.5]}},
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import denoise_tv_chambolle

        weight = _safe_float(kwargs.get("weight", 0.1), 0.1)
        denoised = denoise_tv_chambolle(image.astype(np.float64), weight=weight)
        return ToolResult(
            image=denoised.astype(np.float32),
            tool_name=self.name,
            metadata={"weight": weight},
        )


@ToolRegistry.register
class NLMDenoise(BaseTool):
    """Non-Local Means 去噪 — 结构感知, 利用图像自相似性。

    适用: mild-moderate noise, 纹理丰富区域
    原理: 利用非局部块匹配做加权平均, 保持重复纹理
    优势: 对周期性结构保持最好
    局限: 较慢, 对高噪声效果有限
    """

    @property
    def name(self) -> str:
        return "denoise_nlm"

    @property
    def description(self) -> str:
        return "Non-Local Means denoising: structure-aware via self-similarity."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise"],
            expected_cost="expensive",
            expected_safety="safe",
            params_schema={
                "h": {"type": "float", "default": "auto", "range": [0.005, 0.2]},
                "patch_size": {"type": "int", "default": 5, "range": [3, 9]},
                "patch_distance": {"type": "int", "default": 6, "range": [3, 11]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import denoise_nl_means, estimate_sigma

        sigma_est = float(estimate_sigma(image))
        h_default = max(0.01, 1.15 * sigma_est)
        h = _safe_float(kwargs.get("h", h_default), h_default)
        patch_size = _safe_int(kwargs.get("patch_size", 5), 5)
        patch_distance = _safe_int(kwargs.get("patch_distance", 6), 6)

        denoised = denoise_nl_means(
            image.astype(np.float64),
            h=h,
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=True,
        )
        return ToolResult(
            image=denoised.astype(np.float32),
            tool_name=self.name,
            metadata={"sigma_est": sigma_est, "h": float(h)},
        )


@ToolRegistry.register
class WienerDenoise(BaseTool):
    """Wiener 滤波去噪 — 频域最优线性滤波。

    适用: 均匀高斯白噪声, 已知或可估计噪声功率谱
    原理: 频域最小均方误差估计
    优势: 对白噪声理论最优
    局限: 假设平稳噪声, 不适合空间变化的退化
    """

    @property
    def name(self) -> str:
        return "denoise_wiener"

    @property
    def description(self) -> str:
        return "Wiener filter: frequency-domain denoising for uniform Gaussian noise."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise"],
            expected_cost="cheap",
            expected_safety="moderate",
            params_schema={"mysize": {"type": "int", "default": 5, "range": [3, 15]}},
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.signal import wiener

        mysize = _safe_int(kwargs.get("mysize", 5), 5)
        denoised = wiener(image.astype(np.float64), mysize=mysize)
        denoised = np.clip(denoised, 0.0, 1.0)
        return ToolResult(
            image=denoised.astype(np.float32),
            tool_name=self.name,
            metadata={"mysize": mysize},
        )
