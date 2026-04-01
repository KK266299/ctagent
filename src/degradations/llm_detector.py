# ============================================================================
# 模块职责: LLM-based CT 伪影检测器 — 通过 VLM 视觉模型判断伪影类型和严重程度
#   将 CT 图像转为窗位显示后发送给 VLM, 获取结构化伪影分析结果
#   可替代或增强 threshold-based DegradationDetector
# ============================================================================
from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

import numpy as np

from src.degradations.types import DegradationReport, DegradationType, Severity

logger = logging.getLogger(__name__)

MU_WATER = 0.192

# 伪影类型字符串 → DegradationType 映射
_ARTIFACT_NAME_MAP: dict[str, DegradationType] = {
    "metal": DegradationType.ARTIFACT_METAL,
    "artifact_metal": DegradationType.ARTIFACT_METAL,
    "ring": DegradationType.ARTIFACT_RING,
    "artifact_ring": DegradationType.ARTIFACT_RING,
    "motion": DegradationType.ARTIFACT_MOTION,
    "artifact_motion": DegradationType.ARTIFACT_MOTION,
    "beam_hardening": DegradationType.ARTIFACT_BEAM_HARDENING,
    "artifact_beam_hardening": DegradationType.ARTIFACT_BEAM_HARDENING,
    "scatter": DegradationType.ARTIFACT_SCATTER,
    "artifact_scatter": DegradationType.ARTIFACT_SCATTER,
    "truncation": DegradationType.ARTIFACT_TRUNCATION,
    "artifact_truncation": DegradationType.ARTIFACT_TRUNCATION,
    "sparse_view": DegradationType.ARTIFACT_SPARSE_VIEW,
    "artifact_sparse_view": DegradationType.ARTIFACT_SPARSE_VIEW,
    "limited_angle": DegradationType.ARTIFACT_LIMITED_ANGLE,
    "artifact_limited_angle": DegradationType.ARTIFACT_LIMITED_ANGLE,
    "focal_spot_blur": DegradationType.ARTIFACT_FOCAL_SPOT_BLUR,
    "artifact_focal_spot_blur": DegradationType.ARTIFACT_FOCAL_SPOT_BLUR,
    "low_dose": DegradationType.LOW_DOSE,
    "noise": DegradationType.NOISE,
    "blur": DegradationType.BLUR,
    "low_resolution": DegradationType.LOW_RESOLUTION,
    "none": DegradationType.UNKNOWN,
    "clean": DegradationType.UNKNOWN,
}

_SEVERITY_MAP: dict[str, Severity] = {
    "mild": Severity.MILD,
    "moderate": Severity.MODERATE,
    "severe": Severity.SEVERE,
}

_SYSTEM_PROMPT = """\
You are an expert CT image quality analyst. Your task is to identify artifact types \
and their severity in CT images.

## Artifact Types You Can Identify

1. **metal** — Metal artifact: bright/dark streaks radiating from metallic implants, photon starvation
2. **ring** — Ring artifact: concentric ring/band patterns centered on rotation center
3. **motion** — Motion artifact: blurred edges, ghosting, double contours from patient movement
4. **beam_hardening** — Beam hardening: cupping effect (center darker than edges), dark bands between dense structures
5. **scatter** — Scatter artifact: overall contrast loss, hazy/foggy appearance, reduced dynamic range
6. **truncation** — Truncation artifact: bright bands/streaks at FOV edges, peripheral intensity boost
7. **low_dose** — Low-dose noise: increased graininess/quantum noise, speckle pattern throughout
8. **sparse_view** — Sparse-view artifact: angular streak/aliasing patterns, view-dependent streaks
9. **limited_angle** — Limited-angle artifact: directional smearing, wedge-shaped missing information, structural distortion
10. **focal_spot_blur** — Focal spot blur: overall spatial resolution loss, blurred edges without motion characteristics
11. **none** — Clean image with no significant artifacts

## Severity Levels
- **mild**: Subtle, barely noticeable, does not significantly affect diagnosis
- **moderate**: Clearly visible, may partially obscure anatomical structures
- **severe**: Prominent, significantly degrades image quality, may prevent accurate diagnosis

## Response Format
Respond ONLY with a JSON object. No markdown, no extra text.
```
{
  "artifacts": [
    {"type": "<artifact_type>", "severity": "<mild|moderate|severe>", "confidence": <0.0-1.0>}
  ],
  "primary_artifact": "<most prominent artifact type>",
  "primary_severity": "<severity of primary artifact>",
  "description": "<brief description of observed artifacts in 1-2 sentences>",
  "recoverability": "<high|medium|low|very_low>"
}
```

If the image is clean, return:
```
{"artifacts": [], "primary_artifact": "none", "primary_severity": "mild", "description": "Clean CT image.", "recoverability": "high"}
```
"""

_USER_PROMPT = """\
Analyze this CT image (brain window, WL=40/WW=80) for artifacts and degradation.
Identify all artifact types present and their severity levels.
"""


def _mu_to_png_b64(
    image: np.ndarray,
    window: str = "brain",
) -> str:
    """将 μ 值图像转为窗位显示的 PNG base64 字符串。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = {"brain": (40, 80), "subdural": (75, 215)}
    wl, ww = windows.get(window, (40, 80))
    hu_lo = wl - ww / 2.0
    hu_hi = wl + ww / 2.0
    mu_lo = MU_WATER * (1.0 + hu_lo / 1000.0)
    mu_hi = MU_WATER * (1.0 + hu_hi / 1000.0)
    display = np.clip((image - mu_lo) / max(mu_hi - mu_lo, 1e-10), 0.0, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.imshow(display, cmap="gray")
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _parse_llm_response(text: str) -> dict[str, Any]:
    """从 LLM 响应文本中提取 JSON。"""
    # 尝试直接解析
    text = text.strip()
    # 去除 markdown code fence
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse LLM response as JSON: %s", text[:200])
    return {}


class LLMDegradationDetector:
    """基于 VLM 的 CT 伪影检测器。

    Usage:
        from llm.api_client import LLMConfig, create_client
        client = create_client(LLMConfig(model="qwen/qwen-2.5-vl-72b-instruct", ...))
        detector = LLMDegradationDetector(client)
        report = detector.detect(mu_image)
    """

    def __init__(
        self,
        llm_client: Any,
        window: str = "brain",
        confidence_threshold: float = 0.3,
    ) -> None:
        self.llm_client = llm_client
        self.window = window
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> DegradationReport:
        """分析 CT 图像伪影。

        Args:
            image: μ 值空间的 CT 图像 (float, 416×416)

        Returns:
            DegradationReport
        """
        image_b64 = _mu_to_png_b64(image, self.window)

        try:
            response = self.llm_client.chat_with_image(
                text=_USER_PROMPT,
                image_b64=image_b64,
                system_prompt=_SYSTEM_PROMPT,
            )
            parsed = _parse_llm_response(response.text)
        except Exception as e:
            logger.error("LLM artifact detection failed: %s", e)
            return DegradationReport(
                metadata={"error": str(e), "source": "llm_detector"}
            )

        return self._build_report(parsed)

    def detect_with_raw(self, image: np.ndarray) -> tuple[DegradationReport, dict[str, Any]]:
        """检测并返回原始 LLM 响应。"""
        image_b64 = _mu_to_png_b64(image, self.window)

        try:
            response = self.llm_client.chat_with_image(
                text=_USER_PROMPT,
                image_b64=image_b64,
                system_prompt=_SYSTEM_PROMPT,
            )
            parsed = _parse_llm_response(response.text)
        except Exception as e:
            logger.error("LLM artifact detection failed: %s", e)
            parsed = {"error": str(e)}
            return DegradationReport(metadata={"error": str(e)}), parsed

        report = self._build_report(parsed)
        return report, parsed

    def _build_report(self, parsed: dict[str, Any]) -> DegradationReport:
        """从 LLM 解析结果构建 DegradationReport。"""
        report = DegradationReport()
        report.metadata["source"] = "llm_detector"
        report.metadata["llm_raw"] = parsed

        artifacts = parsed.get("artifacts", [])
        for art in artifacts:
            art_type_str = art.get("type", "").lower().strip()
            severity_str = art.get("severity", "moderate").lower().strip()
            confidence = art.get("confidence", 0.5)

            if confidence < self.confidence_threshold:
                continue

            deg_type = _ARTIFACT_NAME_MAP.get(art_type_str)
            severity = _SEVERITY_MAP.get(severity_str)

            if deg_type is None or deg_type == DegradationType.UNKNOWN:
                continue
            if severity is None:
                severity = Severity.MODERATE

            report.degradations.append((deg_type, severity))

        # 补充描述信息
        description = parsed.get("description", "")
        recoverability = parsed.get("recoverability", "")
        if description:
            report.metadata["description"] = description
        if recoverability:
            report.metadata["recoverability"] = recoverability

        # 设置 IQA scores (从 LLM 视角)
        primary = parsed.get("primary_artifact", "none")
        report.iqa_scores["llm_primary_artifact"] = hash(primary) % 1000 / 1000.0
        report.iqa_scores["llm_num_artifacts"] = float(len(report.degradations))

        return report
