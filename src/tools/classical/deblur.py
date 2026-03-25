# ============================================================================
# 模块职责: 去模糊工具 — Richardson-Lucy 迭代反卷积
#   deblur_richardson_lucy: 经典迭代去卷积，假设 Poisson 噪声模型
# 参考: scikit-image restoration.richardson_lucy
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


def _make_gaussian_psf(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """生成 2D 高斯 PSF。"""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return psf / psf.sum()


@ToolRegistry.register
class RichardsonLucy(BaseTool):
    """Richardson-Lucy 迭代反卷积 — CT 去模糊的经典选择。

    适用: Gaussian blur, motion blur, 系统 PSF 模糊
    原理: 基于 Poisson 噪声模型的 EM 迭代估计
    优势: 不需要显式正则化即可收敛，结果自然保正
    局限: 迭代过多会放大噪声，需控制 iterations
    """

    @property
    def name(self) -> str:
        return "deblur_richardson_lucy"

    @property
    def description(self) -> str:
        return (
            "Richardson-Lucy iterative deconvolution: restores images "
            "blurred by a known or estimated PSF."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="deblur",
            suitable_for=["blur"],
            expected_cost="medium",
            expected_safety="moderate",
            params_schema={
                "iterations": {"type": "int", "default": 15, "range": [5, 50]},
                "psf_size": {"type": "int", "default": 5, "range": [3, 11]},
                "psf_sigma": {"type": "float", "default": 1.0, "range": [0.5, 3.0]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import richardson_lucy

        try:
            iterations = int(kwargs.get("iterations", 15))
        except (TypeError, ValueError):
            iterations = 15
        try:
            psf_size = int(kwargs.get("psf_size", 5))
        except (TypeError, ValueError):
            psf_size = 5
        try:
            psf_sigma = float(kwargs.get("psf_sigma", 1.0))
        except (TypeError, ValueError):
            psf_sigma = 1.0

        psf = _make_gaussian_psf(size=psf_size, sigma=psf_sigma)

        img = image.astype(np.float64)
        offset = img.min()
        img_shifted = img - offset + 1e-12

        restored = richardson_lucy(img_shifted, psf, num_iter=iterations, clip=False)
        restored = restored + offset
        restored = np.clip(restored, 0.0, image.max())

        return ToolResult(
            image=restored.astype(np.float32),
            tool_name=self.name,
            metadata={"iterations": iterations, "psf_size": psf_size, "psf_sigma": psf_sigma},
        )
