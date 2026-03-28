# ============================================================================
# 模块职责: 射束硬化伪影校正工具
#   bhc_polynomial:  多项式校正 (模拟物理校正)
#   bhc_flatfield:   平场校正 (估计并减去低频杯状分量)
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
class BHCPolynomial(BaseTool):
    """多项式射束硬化校正。

    拟合 μ 值的二次多项式校正曲线，补偿低能光子
    优先吸收导致的非线性衰减。
    """

    @property
    def name(self) -> str:
        return "bhc_polynomial"

    @property
    def description(self) -> str:
        return "Beam hardening correction via polynomial fitting of attenuation values."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_beam_hardening"],
            expected_cost="cheap",
            expected_safety="safe",
            params_schema={
                "degree": {"type": "int", "default": 2, "range": [2, 4]},
                "strength": {"type": "float", "default": 1.0, "range": [0.1, 3.0]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        degree = _safe_int(kwargs.get("degree", 2), 2)
        strength = _safe_float(kwargs.get("strength", 1.0), 1.0)

        arr = image.astype(np.float64)
        body_mask = arr > 0.02
        if np.sum(body_mask) < 100:
            return ToolResult(image=arr.astype(np.float32), tool_name=self.name,
                              success=False, message="Not enough body tissue for BHC")

        vals = arr[body_mask]
        x_norm = vals / max(np.percentile(vals, 95), 1e-10)

        correction = np.zeros_like(x_norm)
        for d in range(2, degree + 1):
            correction += ((-1) ** d) * 0.01 * strength * (x_norm ** d)

        result = arr.copy()
        result[body_mask] = vals - correction

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"BHC polynomial (degree={degree}, strength={strength})",
            metadata={"degree": degree, "strength": strength},
        )


@ToolRegistry.register
class BHCFlatfield(BaseTool):
    """平场校正法射束硬化去除。

    估计杯状效应的低频分量（用大核高斯模糊），
    从原图中减去该分量来恢复均匀性。
    """

    @property
    def name(self) -> str:
        return "bhc_flatfield"

    @property
    def description(self) -> str:
        return "Beam hardening correction via flat-field estimation and subtraction."

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="artifact_removal",
            suitable_for=["artifact_beam_hardening"],
            expected_cost="cheap",
            expected_safety="safe",
            params_schema={
                "kernel_size": {"type": "float", "default": 50.0, "range": [20.0, 100.0]},
                "strength": {"type": "float", "default": 0.5, "range": [0.1, 1.0]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from scipy.ndimage import gaussian_filter

        kernel_size = _safe_float(kwargs.get("kernel_size", 50.0), 50.0)
        strength = _safe_float(kwargs.get("strength", 0.5), 0.5)

        arr = image.astype(np.float64)
        low_freq = gaussian_filter(arr, sigma=kernel_size)

        body_mask = arr > 0.02
        if np.sum(body_mask) < 100:
            return ToolResult(image=arr.astype(np.float32), tool_name=self.name,
                              success=False, message="Not enough body tissue for BHC")

        body_mean = np.mean(arr[body_mask])
        lf_mean = np.mean(low_freq[body_mask])
        if lf_mean < 1e-10:
            return ToolResult(image=arr.astype(np.float32), tool_name=self.name,
                              success=False, message="Low-freq component too small")

        correction = (low_freq - body_mean) * strength
        result = arr - correction
        result = np.clip(result, 0.0, arr.max() * 1.5)

        return ToolResult(
            image=result.astype(np.float32),
            tool_name=self.name,
            message=f"BHC flat-field (kernel={kernel_size}, strength={strength})",
            metadata={"kernel_size": kernel_size, "strength": strength},
        )
