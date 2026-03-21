# ============================================================================
# 模块职责: Safety Judge — 检查修复是否引入诊断有害变化
#   校准为 [0, 1] 值域图像 (toy phantom / 归一化 CT)
#   检查维度:
#     1. 全局统计稳定性: 均值偏移、标准差变化
#     2. 最大局部变化: 是否有区域被极端修改
#     3. 结构相似度 (SSIM): 修复前后整体结构一致性
#     4. 数值合理性: NaN/Inf/越界检查
#   每项检查产出连续子分数 [0,1], 综合加权为最终 score
# 参考: MedAgent-Pro — safety-aware medical AI
#       4KAgent — quality & safety reflection
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from judge.base import BaseJudge, JudgeVerdict

logger = logging.getLogger(__name__)


class SafetyJudge(BaseJudge):
    """安全性评判器 — 检查修复是否引入诊断有害变化。

    所有阈值校准为 [0, 1] 值域图像。
    每项检查输出连续分数而非二值 pass/fail, 避免 score 常量化。
    """

    def __init__(
        self,
        max_mean_shift: float = 0.03,
        max_std_change: float = 0.5,
        max_local_change: float = 0.3,
        ssim_floor: float = 0.15,
        pass_threshold: float = 0.6,
    ) -> None:
        self.max_mean_shift = max_mean_shift
        self.max_std_change = max_std_change
        self.max_local_change = max_local_change
        self.ssim_floor = ssim_floor
        self.pass_threshold = pass_threshold

    @property
    def judge_type(self) -> str:
        return "safety"

    def judge(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray,
        reference: np.ndarray | None = None,
        **kwargs: Any,
    ) -> JudgeVerdict:
        checks: dict[str, dict[str, Any]] = {}
        sub_scores: list[float] = []

        before_f = image_before.astype(np.float64)
        after_f = image_after.astype(np.float64)

        # ---- Check 1: 全局均值稳定性 ----
        mean_shift = abs(float(np.mean(after_f)) - float(np.mean(before_f)))
        mean_score = max(0.0, 1.0 - mean_shift / self.max_mean_shift)
        checks["mean_stability"] = {
            "shift": round(mean_shift, 5),
            "threshold": self.max_mean_shift,
            "score": round(mean_score, 4),
        }
        sub_scores.append(mean_score)

        # ---- Check 2: 标准差变化 ----
        std_before = float(np.std(before_f)) + 1e-8
        std_after = float(np.std(after_f)) + 1e-8
        std_ratio = abs(std_after - std_before) / std_before
        std_score = max(0.0, 1.0 - std_ratio / self.max_std_change)
        checks["std_stability"] = {
            "std_before": round(std_before, 5),
            "std_after": round(std_after, 5),
            "change_ratio": round(std_ratio, 4),
            "score": round(std_score, 4),
        }
        sub_scores.append(std_score)

        # ---- Check 3: 最大局部变化 (patch-wise) ----
        diff = np.abs(after_f - before_f)
        p95_diff = float(np.percentile(diff, 95))
        max_diff = float(np.max(diff))
        local_score = max(0.0, 1.0 - p95_diff / self.max_local_change)
        checks["local_change"] = {
            "p95_diff": round(p95_diff, 5),
            "max_diff": round(max_diff, 5),
            "threshold": self.max_local_change,
            "score": round(local_score, 4),
        }
        sub_scores.append(local_score)

        # ---- Check 4: 结构相似度 (SSIM) ----
        ssim_val = self._compute_ssim(before_f, after_f)
        ssim_score = max(0.0, min(1.0, (ssim_val - self.ssim_floor) / (1.0 - self.ssim_floor)))
        checks["structural_similarity"] = {
            "ssim": round(ssim_val, 4),
            "floor": self.ssim_floor,
            "score": round(ssim_score, 4),
        }
        sub_scores.append(ssim_score)

        # ---- Check 5: 数值合理性 ----
        has_nan = bool(np.any(np.isnan(after_f)))
        has_inf = bool(np.any(np.isinf(after_f)))
        out_of_range = bool(np.any(after_f < -0.1) or np.any(after_f > 1.1))
        validity_score = 0.0 if (has_nan or has_inf) else (0.5 if out_of_range else 1.0)
        checks["value_validity"] = {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "out_of_range": out_of_range,
            "min": round(float(np.min(after_f)), 5),
            "max": round(float(np.max(after_f)), 5),
            "score": validity_score,
        }
        sub_scores.append(validity_score)

        # ---- 综合: 加权平均 ----
        weights = [0.15, 0.15, 0.25, 0.30, 0.15]
        score = sum(w * s for w, s in zip(weights, sub_scores))
        score = round(score, 4)
        passed = score >= self.pass_threshold

        failed_items = [k for k, v in checks.items() if v.get("score", 1.0) < 0.5]
        if passed:
            reason = f"PASS (score={score:.3f})"
        else:
            reason = f"FAIL (score={score:.3f}, weak: {', '.join(failed_items)})"

        return JudgeVerdict(
            passed=passed,
            score=score,
            reason=reason,
            judge_type=self.judge_type,
            details=checks,
        )

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img1, img2, data_range=1.0))
