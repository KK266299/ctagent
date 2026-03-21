# ============================================================================
# 模块职责: Mock 诊断 Adapter — 基于局部对比度的可解释 mock 诊断
#   不调用任何 LLM / API，纯本地计算，用于 toy workflow 验证
#   诊断逻辑:
#     1. 计算局部均值和局部标准差
#     2. 用 z-score 检测显著高于局部背景的区域 (模拟 lesion detection)
#     3. 对噪声敏感: 噪声降低局部 CNR → 漏检 → restoration 有价值
#   与 ToyLabel 的 ground truth 可直接比对
# 参考: MedAgent-Pro — downstream task adapter pattern
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.downstream.base import BaseDownstreamTask, DiagnosisResult

logger = logging.getLogger(__name__)


class MockDiagnosis(BaseDownstreamTask):
    """基于局部对比度的 Mock 诊断器。

    核心指标: local CNR (contrast-to-noise ratio)
    lesion 只有在 CNR > threshold 时才能被检出,
    噪声越大 → local_std 越大 → CNR 越低 → 漏检率越高。
    """

    def __init__(
        self,
        cnr_threshold: float = 2.5,
        min_abs_contrast: float = 0.04,
        noise_floor: float = 0.025,
        min_region_size: int = 8,
        local_window: int = 21,
    ) -> None:
        self.cnr_threshold = cnr_threshold
        self.min_abs_contrast = min_abs_contrast
        self.noise_floor = noise_floor
        self.min_region_size = min_region_size
        self.local_window = local_window

    @property
    def task_name(self) -> str:
        return "mock_lesion_detection"

    def predict(self, image: np.ndarray, **kwargs: Any) -> DiagnosisResult:
        """基于局部对比度检测 lesion。"""
        from scipy.ndimage import uniform_filter, label as ndlabel

        img = image.astype(np.float64)
        w = self.local_window

        local_mean = uniform_filter(img, size=w)
        local_sq_mean = uniform_filter(img ** 2, size=w)
        raw_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0) + 1e-10)
        local_std = np.maximum(raw_std, self.noise_floor)

        z_score = (img - local_mean) / local_std
        abs_contrast = img - local_mean

        candidate_mask = (z_score > self.cnr_threshold) & (abs_contrast > self.min_abs_contrast)
        labeled, num_features = ndlabel(candidate_mask)

        regions = []
        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) < self.min_region_size:
                continue
            cy = int(coords[:, 0].mean())
            cx = int(coords[:, 1].mean())
            peak_z = float(z_score[labeled == i].max())
            regions.append({"y": cy, "x": cx, "size": len(coords), "peak_cnr": round(peak_z, 2)})

        lesion_count = len(regions)
        lesion_present = lesion_count > 0

        h, w_img = image.shape[:2]
        cx_mid = w_img // 2
        positions = []
        for r in regions:
            side = "left" if r["x"] < cx_mid else "right"
            positions.append({**r, "side": side})

        if lesion_present:
            sides = [p["side"] for p in positions]
            if all(s == "left" for s in sides):
                laterality = "left"
            elif all(s == "right" for s in sides):
                laterality = "right"
            else:
                laterality = "bilateral"
        else:
            laterality = "none"

        confidence = self._compute_confidence(regions)

        if lesion_present:
            prediction = f"lesion_detected ({lesion_count})"
            severity = "mild" if lesion_count == 1 else "moderate"
        else:
            prediction = "normal"
            severity = "normal"

        return DiagnosisResult(
            prediction=prediction,
            confidence=confidence,
            metadata={
                "lesion_present": lesion_present,
                "lesion_count": lesion_count,
                "lesion_positions": positions,
                "laterality": laterality,
                "severity": severity,
                "method": "mock_local_cnr",
            },
        )

    @staticmethod
    def _compute_confidence(regions: list[dict]) -> float:
        if not regions:
            return 0.6
        max_cnr = max(r.get("peak_cnr", 0) for r in regions)
        return min(0.98, 0.5 + max_cnr / 20.0)
