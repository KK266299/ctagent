# ============================================================================
# 模块职责: BM3D 去噪 — SOTA 经典去噪，带 fallback
#   若 bm3d 未安装，自动降级为 wavelet + NLM 组合
# 参考: bm3d (https://pypi.org/project/bm3d/)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry

try:
    import bm3d as _bm3d
    _HAS_BM3D = True
except ImportError:
    _HAS_BM3D = False


@ToolRegistry.register
class BM3DDenoise(BaseTool):
    """BM3D 去噪 — 经典 SOTA 去噪算法。

    适用: Gaussian noise, mixed noise
    原理: Block-Matching and 3D filtering
    优势: 在经典方法中 PSNR 通常最高
    Fallback: 若 bm3d 未安装，自动使用 wavelet + bilateral 组合
    """

    @property
    def name(self) -> str:
        return "denoise_bm3d"

    @property
    def description(self) -> str:
        if _HAS_BM3D:
            return "BM3D denoising: state-of-the-art classical denoiser, best PSNR among classical methods."
        return (
            "BM3D denoising (fallback mode: wavelet+bilateral combo). "
            "Install bm3d package for full performance."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise"],
            expected_cost="expensive",
            expected_safety="safe",
            params_schema={
                "sigma_psd": {"type": "float", "default": "auto", "range": [0.001, 0.1]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        if _HAS_BM3D:
            return self._run_bm3d(image, **kwargs)
        return self._run_fallback(image, **kwargs)

    def _run_bm3d(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import estimate_sigma

        sigma_psd = kwargs.get("sigma_psd", None)
        if sigma_psd is None or sigma_psd == "auto":
            sigma_psd = float(estimate_sigma(image))

        denoised = _bm3d.bm3d(image.astype(np.float64), sigma_psd=sigma_psd)
        return ToolResult(
            image=denoised.astype(np.float32),
            tool_name=self.name,
            metadata={"method": "bm3d", "sigma_psd": sigma_psd},
        )

    def _run_fallback(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import denoise_wavelet, denoise_bilateral

        stage1 = denoise_wavelet(
            image.astype(np.float64),
            wavelet="db4",
            method="BayesShrink",
            mode="soft",
            rescale_sigma=True,
        )
        stage2 = denoise_bilateral(
            stage1,
            sigma_color=0.03,
            sigma_spatial=3,
            channel_axis=None,
        )
        return ToolResult(
            image=stage2.astype(np.float32),
            tool_name=self.name,
            metadata={"method": "fallback_wavelet_bilateral"},
        )
