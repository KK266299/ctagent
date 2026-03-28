# ============================================================================
# 模块职责: 运动伪影校正工具
#   motion_correction_tv:    TV 正则化抑制运动条纹
#   motion_correction_wiener: 方向性 Wiener 去卷积
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


def _safe_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


@ToolRegistry.register
class MotionCorrectionTV(BaseTool):
    """TV 正则化运动伪影校正。

    运动伪影表现为方向性条纹/鬼影，TV 去噪能有效抑制
    这类结构化噪声同时保持边缘。
    """

    @property
    def name(self) -> str:
        return "motion_correction_tv"

    @property
    def description(self) -> str:
        return "Suppress motion artifacts using anisotropic Total Variation regularization."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_motion"],
            expected_cost="medium",
            expected_safety="safe",
            params_schema={
                "weight": {"type": "float", "default": 0.15, "range": [0.01, 0.5]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import denoise_tv_chambolle

        weight = _safe_float(kwargs.get("weight", 0.15), 0.15)
        result = denoise_tv_chambolle(image.astype(np.float64), weight=weight)
        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"Motion correction (TV, weight={weight})",
            metadata={"weight": weight},
        )


@ToolRegistry.register
class MotionCorrectionWiener(BaseTool):
    """方向性 Wiener 去卷积运动校正。

    对检测到的主运动方向构建线性运动 PSF，
    用 Wiener 反卷积恢复。
    """

    @property
    def name(self) -> str:
        return "motion_correction_wiener"

    @property
    def description(self) -> str:
        return "Correct directional motion blur via Wiener deconvolution with estimated motion PSF."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_motion"],
            expected_cost="medium",
            expected_safety="moderate",
            params_schema={
                "motion_length": {"type": "int", "default": 5, "range": [1, 20]},
                "snr": {"type": "float", "default": 20.0, "range": [5.0, 100.0]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        motion_length = _safe_int(kwargs.get("motion_length", 5), 5)
        snr = _safe_float(kwargs.get("snr", 20.0), 20.0)

        arr = image.astype(np.float64)
        gy = np.diff(arr, axis=0)
        gx = np.diff(arr, axis=1)
        energy_y = np.sum(gy ** 2)
        energy_x = np.sum(gx ** 2)
        angle = np.degrees(np.arctan2(energy_y, energy_x))

        h, w = arr.shape
        psf = np.zeros((h, w))
        center_y, center_x = h // 2, w // 2
        rad = np.radians(angle)
        for i in range(motion_length):
            offset = i - motion_length // 2
            dy = int(round(offset * np.sin(rad)))
            dx = int(round(offset * np.cos(rad)))
            py, px = center_y + dy, center_x + dx
            if 0 <= py < h and 0 <= px < w:
                psf[py, px] = 1.0
        psf /= max(psf.sum(), 1e-10)

        F_img = np.fft.fft2(arr)
        F_psf = np.fft.fft2(psf)
        F_psf_conj = np.conj(F_psf)
        nsr = 1.0 / snr
        F_restored = F_img * F_psf_conj / (np.abs(F_psf) ** 2 + nsr)
        result = np.real(np.fft.ifft2(F_restored))
        result = np.clip(result, 0.0, arr.max() * 1.2)

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"Motion correction (Wiener, length={motion_length}, angle={angle:.1f}°)",
            metadata={"motion_length": motion_length, "angle": angle, "snr": snr},
        )
