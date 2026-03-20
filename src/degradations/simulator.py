# ============================================================================
# 模块职责: 退化模拟器 — 人工合成退化用于训练和评估
# 参考: ProCT (https://github.com/Masaaki-75/proct) — CT 退化建模
#       PromptCT (https://github.com/shibaoshun/PromptCT)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.degradations.types import DegradationType


class DegradationSimulator:
    """CT 图像退化模拟器。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def apply(
        self, image: np.ndarray, degradation_type: DegradationType, **kwargs: Any
    ) -> np.ndarray:
        """对图像施加指定退化。"""
        handlers = {
            DegradationType.NOISE: self._add_noise,
            DegradationType.BLUR: self._add_blur,
            DegradationType.LOW_RESOLUTION: self._downsample,
        }
        handler = handlers.get(degradation_type)
        if handler is None:
            raise NotImplementedError(f"Degradation not implemented: {degradation_type}")
        return handler(image, **kwargs)

    def _add_noise(self, image: np.ndarray, sigma: float = 25.0, **_: Any) -> np.ndarray:
        """添加高斯噪声。"""
        rng = np.random.default_rng()
        noise = rng.normal(0, sigma, image.shape).astype(image.dtype)
        return image + noise

    def _add_blur(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 1.5, **_: Any) -> np.ndarray:
        """添加高斯模糊。"""
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(image, sigma=sigma)

    def _downsample(self, image: np.ndarray, scale: int = 2, **_: Any) -> np.ndarray:
        """降采样。"""
        from skimage.transform import resize

        h, w = image.shape[:2]
        small = resize(image, (h // scale, w // scale), anti_aliasing=True)
        return resize(small, (h, w))
