# ============================================================================
# 模块职责: 统计工具 — CT 图像 HU 分布 / ROI 统计 / 直方图特征
# 参考: Earth-Agent (https://github.com/opendatalab/Earth-Agent/tree/main/agent/tools)
#       CAPIQA (https://github.com/aaz-imran/capiqa) — CT 统计特征
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np


class StatisticsTool:
    """CT 图像统计工具 — 提取 HU 分布与 ROI 统计信息。"""

    name = "ct_image_statistics"
    description = (
        "Compute statistical features of a CT image: HU distribution, "
        "percentiles, histogram features, and optional ROI-based statistics."
    )

    parameters_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "Image identifier"},
            "roi_center": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "ROI center [y, x] (optional)",
            },
            "roi_size": {"type": "integer", "description": "ROI half-size in pixels"},
        },
        "required": ["image"],
    }

    def __call__(
        self,
        image: np.ndarray,
        roi_center: tuple[int, int] | None = None,
        roi_size: int = 32,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """计算统计信息，返回结构化结果。"""
        result: dict[str, Any] = {
            "tool": self.name,
            "global": self._compute_stats(image),
        }

        # ROI 统计
        if roi_center is not None:
            roi = self._extract_roi(image, roi_center, roi_size)
            if roi is not None:
                result["roi"] = self._compute_stats(roi)
                result["roi_center"] = list(roi_center)
                result["roi_size"] = roi_size

        # 直方图特征
        result["histogram_features"] = self._histogram_features(image)
        return result

    def _compute_stats(self, arr: np.ndarray) -> dict[str, float]:
        """计算基本统计量。"""
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }

    def _extract_roi(
        self, image: np.ndarray, center: tuple[int, int], half_size: int
    ) -> np.ndarray | None:
        """提取 ROI 区域。"""
        y, x = center
        h, w = image.shape[:2]
        y0 = max(0, y - half_size)
        y1 = min(h, y + half_size)
        x0 = max(0, x - half_size)
        x1 = min(w, x + half_size)
        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        return roi

    def _histogram_features(self, image: np.ndarray, bins: int = 64) -> dict[str, float]:
        """计算直方图特征。"""
        hist, bin_edges = np.histogram(image.ravel(), bins=bins)
        hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
        # 熵
        nonzero = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        # 峰度 / 偏度
        from scipy.stats import kurtosis, skew

        flat = image.ravel()
        return {
            "entropy": entropy,
            "kurtosis": float(kurtosis(flat)),
            "skewness": float(skew(flat)),
        }
