# ============================================================================
# 模块职责: 增强工具 — 直方图匹配 + Laplacian 边缘增强
#   histogram_match:   将退化图像强度分布对齐到参考图像
#   enhance_laplacian: Laplacian 高通增强边缘，与 USM 互补
# 参考: scikit-image exposure.match_histograms, scipy.ndimage
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class HistogramMatch(BaseTool):
    """直方图匹配 — 将退化图像的强度分布对齐到参考图像。

    适用: 对比度偏移、域漂移、修复后校准
    原理: CDF 匹配，将源图像的累积分布函数映射到目标
    优势: 简单有效的域对齐方法
    注意: 需要 reference 参数
    """

    @property
    def name(self) -> str:
        return "histogram_match"

    @property
    def description(self) -> str:
        return (
            "Histogram matching: aligns intensity distribution of degraded "
            "image to a reference (e.g. clean) image."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="contrast",
            suitable_for=["contrast", "artifact"],
            expected_cost="cheap",
            expected_safety="safe",
            params_schema={
                "reference": {"type": "ndarray", "required": True, "description": "target distribution image"},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.exposure import match_histograms

        reference = kwargs.get("reference", None)
        if reference is None:
            return ToolResult(
                image=image,
                tool_name=self.name,
                success=False,
                message="histogram_match requires a 'reference' image parameter.",
            )

        matched = match_histograms(image.astype(np.float64), reference.astype(np.float64))
        return ToolResult(
            image=matched.astype(np.float32),
            tool_name=self.name,
        )


@ToolRegistry.register
class LaplacianEnhance(BaseTool):
    """Laplacian 边缘增强 — 高通滤波增强边缘和细节。

    适用: 轻度模糊、细节丢失、低分辨率感
    原理: image + alpha * Laplacian(image)
    优势: 比 USM 更精确的边缘增强，不依赖 Gaussian 差分
    局限: 会放大噪声，建议先去噪再增强
    """

    @property
    def name(self) -> str:
        return "enhance_laplacian"

    @property
    def description(self) -> str:
        return (
            "Laplacian edge enhancement: sharpens edges and fine details, "
            "complementary to USM."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="sharpen",
            suitable_for=["blur", "low_resolution"],
            expected_cost="cheap",
            expected_safety="risky",
            params_schema={
                "alpha": {"type": "float", "default": 0.3, "range": [0.1, 1.0]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.ndimage import laplace

        try:
            alpha = float(kwargs.get("alpha", 0.3))
        except (TypeError, ValueError):
            alpha = 0.3
        alpha = max(0.05, min(alpha, 1.0))

        lap = laplace(image.astype(np.float64))
        enhanced = image.astype(np.float64) - alpha * lap
        enhanced = np.clip(enhanced, 0.0, max(image.max(), 1.0))
        return ToolResult(
            image=enhanced.astype(np.float32),
            tool_name=self.name,
            metadata={"alpha": alpha},
        )
