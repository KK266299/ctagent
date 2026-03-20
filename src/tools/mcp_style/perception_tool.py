# ============================================================================
# 模块职责: 感知工具 — 计算 CT 图像的 IQA 指标，返回质量评估报告
# 参考: 4KAgent (https://github.com/taco-group/4KAgent) — perception module
#       IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch)
#       CAPIQA (https://github.com/aaz-imran/capiqa) — CT IQA
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.iqa.metrics import compute_metrics
from src.iqa.no_reference import NoReferenceIQA


class PerceptionTool:
    """IQA 感知工具 — 计算图像质量指标。

    MCP-style: 支持有参考和无参考两种模式。
    """

    name = "ct_quality_perception"
    description = (
        "Compute image quality assessment metrics for a CT image. "
        "Supports no-reference metrics (sharpness, noise level) and "
        "full-reference metrics (PSNR, SSIM) when a reference is provided."
    )

    parameters_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "Image identifier"},
            "reference": {"type": "string", "description": "Reference image (optional)"},
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metric names to compute",
            },
        },
        "required": ["image"],
    }

    def __init__(self) -> None:
        self.nr_iqa = NoReferenceIQA()

    def __call__(
        self,
        image: np.ndarray,
        reference: np.ndarray | None = None,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """计算 IQA 指标，返回结构化结果。"""
        result: dict[str, Any] = {"tool": self.name}

        # 无参考指标 (总是计算)
        nr_scores = self.nr_iqa.evaluate(image)
        result["no_reference"] = nr_scores

        # 有参考指标
        if reference is not None:
            metric_names = metrics or ["psnr", "ssim"]
            fr_scores = compute_metrics(image, reference, metric_names)
            result["full_reference"] = fr_scores

        # 质量等级判定
        result["quality_grade"] = self._grade_quality(nr_scores)
        return result

    def _grade_quality(self, nr_scores: dict[str, float]) -> str:
        """基于无参考指标给出质量等级。"""
        sharpness = nr_scores.get("sharpness", 0)
        noise = nr_scores.get("noise_estimate", 0)
        if sharpness > 100 and noise < 10:
            return "good"
        elif sharpness > 50 and noise < 30:
            return "moderate"
        else:
            return "poor"
