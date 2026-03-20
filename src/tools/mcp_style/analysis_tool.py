# ============================================================================
# 模块职责: 分析工具 — 对 CT 图像做退化类型判断，返回结构化报告
#   供 VLM 做增强诊断时使用，输出 JSON 而非图像
# 参考: Earth-Agent (https://github.com/opendatalab/Earth-Agent/tree/main/agent/tools)
#       4KAgent (https://github.com/taco-group/4KAgent) — degradation analysis
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.degradations.detector import DegradationDetector


class AnalysisTool:
    """退化分析工具 — 检测 CT 图像退化类型和严重程度。

    MCP-style: 接收图像，返回 JSON 结构化分析结果。
    """

    name = "ct_degradation_analysis"
    description = (
        "Analyze a CT image to detect degradation types (noise, blur, artifact, "
        "low-resolution) and their severity levels. Returns a structured report."
    )

    # JSON Schema for tool parameters (MCP 兼容)
    parameters_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "Image identifier or path"},
        },
        "required": ["image"],
    }

    def __init__(self, detector: DegradationDetector | None = None) -> None:
        self.detector = detector or DegradationDetector()

    def __call__(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """执行退化分析，返回结构化结果。"""
        report = self.detector.detect(image)
        return {
            "tool": self.name,
            "degradations": [
                {"type": d.value, "severity": s.value}
                for d, s in report.degradations
            ],
            "iqa_scores": report.iqa_scores,
            "primary_degradation": (
                report.primary_degradation.value
                if report.primary_degradation
                else "none"
            ),
            "num_degradations": len(report.degradations),
        }
