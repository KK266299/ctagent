# ============================================================================
# 模块职责: 小波去噪 — 多尺度保细节去噪，CT 去噪标配
#   denoise_wavelet: BayesShrink / VisuShrink 小波阈值去噪
# 参考: scikit-image restoration.denoise_wavelet
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class WaveletDenoise(BaseTool):
    """小波阈值去噪 — 多尺度分解，保持精细结构的同时有效去噪。

    适用: mild-severe noise，对 CT 纹理保持优于 TV
    原理: DWT 分解 → 各尺度系数 soft-thresholding → IDWT 重构
    优势: 自适应（BayesShrink），对不同频段噪声区别对待
    """

    @property
    def name(self) -> str:
        return "denoise_wavelet"

    @property
    def description(self) -> str:
        return (
            "Wavelet thresholding denoising (BayesShrink): multi-scale, "
            "preserves fine structures better than TV for CT images."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise", "artifact"],
            expected_cost="medium",
            expected_safety="safe",
            params_schema={
                "wavelet": {"type": "str", "default": "db4", "options": ["db1", "db4", "db8", "sym4", "coif1"]},
                "method": {"type": "str", "default": "BayesShrink", "options": ["BayesShrink", "VisuShrink"]},
                "mode": {"type": "str", "default": "soft", "options": ["soft", "hard"]},
                "wavelet_levels": {"type": "int", "default": "auto", "range": [1, 6]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import denoise_wavelet, estimate_sigma

        wavelet = str(kwargs.get("wavelet", "db4"))
        method = str(kwargs.get("method", "BayesShrink"))
        mode = str(kwargs.get("mode", "soft"))
        wavelet_levels = kwargs.get("wavelet_levels", None)
        if wavelet_levels is not None:
            try:
                wavelet_levels = int(wavelet_levels)
            except (TypeError, ValueError):
                wavelet_levels = None

        sigma_est = float(estimate_sigma(image))

        denoised = denoise_wavelet(
            image.astype(np.float64),
            wavelet=wavelet,
            method=method,
            mode=mode,
            wavelet_levels=wavelet_levels,
            rescale_sigma=True,
        )
        return ToolResult(
            image=denoised.astype(np.float32),
            tool_name=self.name,
            metadata={"sigma_est": sigma_est, "wavelet": wavelet, "method": method, "mode": mode},
        )
