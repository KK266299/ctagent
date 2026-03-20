# ============================================================================
# 模块职责: 质量评判器 — 基于 IQA 指标判断修复效果
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
    """基于 IQA 阈值的修复质量评判器。"""

    def __init__(
        self,
        psnr_threshold: float = 30.0,
        ssim_threshold: float = 0.85,
    ) -> None:
        self.psnr_threshold = psnr_threshold
        self.ssim_threshold = ssim_threshold
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
        """无参考图像时的评判（比较修复前后质量）。"""
        before = self.nr_iqa.evaluate(degraded)
        after = self.nr_iqa.evaluate(restored)
        improved = after["sharpness"] > before["sharpness"]
        return JudgeVerdict(
            passed=improved,
            score=after["sharpness"],
            reason="Improved" if improved else "No improvement detected",
            details={"before": before, "after": after},
        )
