# ============================================================================
# 模块职责: 经典去噪工具 — NLM, Gaussian 等传统去噪算法
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class NLMDenoise(BaseTool):
    """Non-Local Means 去噪。"""

    @property
    def name(self) -> str:
        return "denoise_nlm"

    @property
    def description(self) -> str:
        return "Non-Local Means denoising for mild to moderate Gaussian noise."

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import denoise_nl_means, estimate_sigma

        sigma_est = estimate_sigma(image)
        h = kwargs.get("h", 1.15 * sigma_est)
        patch_size = kwargs.get("patch_size", 5)
        patch_distance = kwargs.get("patch_distance", 6)

        denoised = denoise_nl_means(
            image, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True
        )
        return ToolResult(image=denoised, tool_name=self.name, metadata={"sigma_est": float(sigma_est)})


@ToolRegistry.register
class GaussianDenoise(BaseTool):
    """Gaussian 滤波去噪。"""

    @property
    def name(self) -> str:
        return "denoise_gaussian"

    @property
    def description(self) -> str:
        return "Gaussian filter denoising, fast but may blur edges."

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.ndimage import gaussian_filter

        sigma = kwargs.get("sigma", 1.0)
        denoised = gaussian_filter(image, sigma=sigma)
        return ToolResult(image=denoised, tool_name=self.name)
