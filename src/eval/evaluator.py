# ============================================================================
# 模块职责: Pipeline 评估器 — 端到端评估并生成报告
# 参考: MedQ-Bench (https://github.com/liujiyaoFDU/MedQ-Bench) — evaluation
#       IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch)
# ============================================================================
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.iqa.metrics import compute_metrics

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """端到端 Pipeline 评估器。"""

    def __init__(
        self,
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.metric_names = metric_names or ["psnr", "ssim"]
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: list[dict[str, Any]] = []

    def evaluate_single(
        self,
        restored: np.ndarray,
        reference: np.ndarray,
        sample_id: str = "",
    ) -> dict[str, float]:
        """评估单个样本。"""
        metrics = compute_metrics(restored, reference, self.metric_names)
        record = {"sample_id": sample_id, **metrics}
        self.results.append(record)
        return metrics

    def summarize(self) -> dict[str, float]:
        """汇总所有样本的评估结果。"""
        if not self.results:
            return {}
        summary = {}
        for key in self.metric_names:
            values = [r[key] for r in self.results if key in r]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
        return summary

    def save_report(self, path: str | Path | None = None) -> None:
        """保存评估报告为 JSON。"""
        if path is None:
            if self.output_dir is None:
                logger.warning("No output path specified, skipping report save.")
                return
            path = self.output_dir / "eval_report.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "summary": self.summarize(),
            "per_sample": self.results,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved to %s", path)
