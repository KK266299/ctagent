# ============================================================================
# 模块职责: 质量评判器 — 基于 IQA 指标判断修复效果
#   有参考: PSNR + SSIM 阈值
#   无参考: noise_reduction × structure_preservation 复合评分 [0, 1]
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
#       IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch)
#       CAPIQA (https://github.com/aaz-imran/capiqa)
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.iqa.metrics import compute_metrics
from src.iqa.no_reference import NoReferenceIQA


@dataclass
class JudgeVerdict:
    """评判结果。"""
    passed: bool
    score: float
    reason: str
    details: dict[str, Any]


class QualityJudge:
    """基于 IQA 的修复质量评判器。

    judge_with_reference: PSNR/SSIM 阈值
    judge_no_reference:   noise_reduction + structure_preservation 复合分
    """

    def __init__(
        self,
        psnr_threshold: float = 30.0,
        ssim_threshold: float = 0.85,
        quality_threshold: float = 0.35,
    ) -> None:
        self.psnr_threshold = psnr_threshold
        self.ssim_threshold = ssim_threshold
        self.quality_threshold = quality_threshold
        self.nr_iqa = NoReferenceIQA()

    def judge_with_reference(
        self,
        restored: np.ndarray,
        reference: np.ndarray,
    ) -> JudgeVerdict:
        """有参考图像时的评判。"""
        metrics = compute_metrics(restored, reference, ["psnr", "ssim"])
        passed = (
            metrics.get("psnr", 0) >= self.psnr_threshold
            and metrics.get("ssim", 0) >= self.ssim_threshold
        )
        score = metrics.get("ssim", 0)
        reason = "PASS" if passed else f"Below threshold: PSNR={metrics.get('psnr', 0):.2f}, SSIM={metrics.get('ssim', 0):.4f}"
        return JudgeVerdict(passed=passed, score=score, reason=reason, details=metrics)

    def judge_no_reference(
        self,
        degraded: np.ndarray,
        restored: np.ndarray,
    ) -> JudgeVerdict:
        """无参考评判: noise_reduction + cleanliness + structure 复合分。

        三分量:
          noise_reduction: 噪声降低比例 (0=无降低, 1=完全去除)
          cleanliness:     绝对残余噪声水平 (区分轻/重退化修复后的绝对质量)
          structure:       结构保持率 (防止过度平滑)

        Score 语义:
          0.0-0.2: 修复无效或有害
          0.2-0.4: 轻微改善
          0.4-0.6: 有效降噪但可能过度平滑
          0.6-0.8: 良好修复
          0.8-1.0: 优秀修复
        """
        before = self.nr_iqa.evaluate(degraded)
        after = self.nr_iqa.evaluate(restored)

        noise_before = before["noise_estimate"]
        noise_after = after["noise_estimate"]
        sharp_before = before["sharpness"]
        sharp_after = after["sharpness"]

        noise_reduction = max(0.0, (noise_before - noise_after)) / (noise_before + 1e-8)

        cleanliness = max(0.0, min(1.0, 1.0 - noise_after / 0.03))

        sharp_ratio = sharp_after / (sharp_before + 1e-8)
        structure_score = min(1.0, sharp_ratio ** 0.5)

        score = round(0.35 * noise_reduction + 0.35 * cleanliness + 0.30 * structure_score, 4)
        passed = score >= self.quality_threshold

        label = "PASS" if passed else "FAIL"
        reason = (f"{label} (score={score:.3f}, "
                  f"nr={noise_reduction:.3f}, clean={cleanliness:.3f}, struct={structure_score:.3f})")

        return JudgeVerdict(
            passed=passed,
            score=score,
            reason=reason,
            details={
                "noise_reduction": round(noise_reduction, 4),
                "cleanliness": round(cleanliness, 4),
                "structure_preservation": round(structure_score, 4),
                "noise_before": round(noise_before, 6),
                "noise_after": round(noise_after, 6),
                "sharpness_before": round(sharp_before, 6),
                "sharpness_after": round(sharp_after, 6),
            },
        )
