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

1. **metal** — Metal artifact: bright/dark streaks radiating from metallic implants, photon starvation areas (black voids near metal)
2. **ring** — Ring artifact: **concentric circular bands** centered on the rotation axis. Key: the pattern is CIRCULAR, not radial. Looks like rings on a target.
3. **motion** — Motion artifact: blurred edges, ghosting, **double contours** of anatomical structures. Structures are still recognizable but appear duplicated or smeared.
4. **beam_hardening** — Beam hardening: **dark bands/streaks between two dense structures** (e.g., between petrous bones, or between bones and metal). Also causes cupping (center darker than periphery). Key: localized dark bands, NOT uniform haze.
5. **scatter** — Scatter artifact: **uniform haze/fog** across the entire image, overall contrast reduction, reduced dynamic range. Key: diffuse and uniform, NO localized bands.
6. **truncation** — Truncation artifact: bright bands/streaks at **FOV edges** (periphery), peripheral intensity boost
7. **low_dose** — Low-dose noise: increased **graininess/quantum mottle** throughout, salt-and-pepper speckle pattern. Noise is visibly signal-dependent (noisier in denser regions).
8. **sparse_view** — Sparse-view artifact: **radial streak patterns emanating outward from center** with angular periodicity (evenly spaced streaks). Key: streaks are RADIAL (pointing outward), NOT circular like rings.
9. **limited_angle** — Limited-angle artifact: **directional elongation/smearing** of structures, loss of structural information in one direction, structures appear stretched or distorted beyond recognition. Key: broad directional distortion, NOT simple ghosting.
10. **focal_spot_blur** — Focal spot blur: **uniform spatial resolution loss** in ALL directions (isotropic). Edges are blurred equally everywhere. Key: NO directional preference, unlike motion blur.
11. **none** — Clean image with no significant artifacts

## Common Confusions — READ CAREFULLY
- **beam_hardening vs scatter**: Beam hardening has LOCALIZED dark bands between bones; scatter is UNIFORM contrast loss across the whole image. If you see dark streaks between dense structures → beam_hardening. If the whole image looks foggy → scatter.
- **sparse_view vs ring**: Sparse-view has RADIAL streaks (like spokes of a wheel); ring has CIRCULAR bands (like tree rings). They are perpendicular patterns.
- **sparse_view vs motion**: Sparse-view streaks are angularly PERIODIC and centered on image center; motion shows ghosting of SPECIFIC anatomical structures.
- **limited_angle vs motion**: Limited-angle causes STRUCTURAL DISTORTION and loss of information in a direction; motion causes GHOSTING where structures are still recognizable but doubled.
- **focal_spot_blur vs motion**: Focal spot blur is ISOTROPIC (same in all directions); motion blur is ANISOTROPIC (worse in one direction).
- **low_dose vs noise**: Low-dose noise is signal-dependent (Poisson), appears as quantum mottle; general noise is uniform Gaussian.

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

_USER_PROMPT_DUAL = """\
Two views of the same CT slice are provided:
- Image 1: Brain window (WL=40, WW=80) — optimized for soft tissue contrast
- Image 2: Subdural window (WL=75, WW=215) — broader dynamic range for bone and hemorrhage

Analyze BOTH views to identify all artifact types and their severity levels.
The brain window is better for detecting scatter (contrast loss) and beam hardening (dark bands).
The subdural window is better for detecting ring artifacts, truncation, and structural distortion.
"""


def _mu_to_png_b64(
    image: np.ndarray,
    window: str = "brain",
    max_size: int = 512,
) -> str:
    """将 μ 值图像转为窗位显示的 PNG base64 字符串。"""
    from PIL import Image as PILImage

    windows = {"brain": (40, 80), "subdural": (75, 215)}
    wl, ww = windows.get(window, (40, 80))
    hu_lo = wl - ww / 2.0
    hu_hi = wl + ww / 2.0
    mu_lo = MU_WATER * (1.0 + hu_lo / 1000.0)
    mu_hi = MU_WATER * (1.0 + hu_hi / 1000.0)
    display = np.clip((image - mu_lo) / max(mu_hi - mu_lo, 1e-10), 0.0, 1.0)

    img_uint8 = (display * 255).astype(np.uint8)
    pil_img = PILImage.fromarray(img_uint8, mode="L")

    # 限制尺寸以减少传输量
    h, w = pil_img.size
    if max(h, w) > max_size:
        pil_img.thumbnail((max_size, max_size), PILImage.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
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
        dual_window: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.window = window
        self.confidence_threshold = confidence_threshold
        self.dual_window = dual_window

    def _call_llm(self, image: np.ndarray, max_retries: int = 3) -> str:
        """调用 LLM，支持单窗位和双窗位模式，带重试。"""
        import time

        if self.dual_window:
            brain_b64 = _mu_to_png_b64(image, "brain")
            subdural_b64 = _mu_to_png_b64(image, "subdural")
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": _USER_PROMPT_DUAL},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{brain_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{subdural_b64}"}},
                ]},
            ]
            call_fn = lambda: self.llm_client.chat(messages)
        else:
            image_b64 = _mu_to_png_b64(image, self.window)
            call_fn = lambda: self.llm_client.chat_with_image(
                text=_USER_PROMPT,
                image_b64=image_b64,
                system_prompt=_SYSTEM_PROMPT,
            )

        last_err = None
        for attempt in range(max_retries):
            try:
                response = call_fn()
                return response.text
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning("LLM call attempt %d/%d failed: %s, retrying in %ds",
                               attempt + 1, max_retries, e, wait)
                time.sleep(wait)
        raise last_err

    def detect(self, image: np.ndarray) -> DegradationReport:
        """分析 CT 图像伪影。

        Args:
            image: μ 值空间的 CT 图像 (float, 416×416)

        Returns:
            DegradationReport
        """
        try:
            text = self._call_llm(image)
            parsed = _parse_llm_response(text)
        except Exception as e:
            logger.error("LLM artifact detection failed: %s", e)
            return DegradationReport(
                metadata={"error": str(e), "source": "llm_detector"}
            )

        return self._build_report(parsed)

    def detect_with_raw(self, image: np.ndarray) -> tuple[DegradationReport, dict[str, Any]]:
        """检测并返回原始 LLM 响应。"""
        try:
            text = self._call_llm(image)
            parsed = _parse_llm_response(text)
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
