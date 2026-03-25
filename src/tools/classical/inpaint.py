# ============================================================================
# 模块职责: 图像修补工具 — 对小区域缺损/伪影做 inpainting
#   inpaint_biharmonic: 基于双调和方程的 inpainting
# 参考: scikit-image restoration.inpaint_biharmonic
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry


@ToolRegistry.register
class BiharmonicInpaint(BaseTool):
    """双调和 Inpainting — 填补小区域缺损或伪影标记区域。

    适用: metal artifact 伪影带, 小区域缺损, streak 标记填补
    原理: 求解双调和方程 (biharmonic PDE) 用周围像素信息填充
    优势: 对小区域效果好, 平滑过渡
    局限: 大区域填补质量下降; 需要 mask 参数
    使用方式:
      1. 传入 mask=True 的区域为待修补区域
      2. 若不传 mask，自动检测极端值像素作为 mask
    """

    @property
    def name(self) -> str:
        return "inpaint_biharmonic"

    @property
    def description(self) -> str:
        return (
            "Biharmonic inpainting: fills small damaged regions (artifacts, "
            "missing pixels) using surrounding information."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="inpaint",
            suitable_for=["artifact"],
            expected_cost="expensive",
            expected_safety="moderate",
            params_schema={
                "mask": {"type": "ndarray", "default": "auto", "description": "boolean mask of damaged pixels"},
                "extreme_percentile": {"type": "float", "default": 99.5, "range": [95.0, 99.9]},
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        from skimage.restoration import inpaint_biharmonic

        mask = kwargs.get("mask", None)

        if mask is None:
            try:
                pct = float(kwargs.get("extreme_percentile", 99.5))
            except (TypeError, ValueError):
                pct = 99.5
            threshold = np.percentile(image, pct)
            mask = image > threshold
            if mask.sum() == 0:
                return ToolResult(
                    image=image.copy(),
                    tool_name=self.name,
                    success=True,
                    message="No extreme pixels detected; image returned as-is.",
                    metadata={"pixels_inpainted": 0},
                )

        mask = mask.astype(bool)
        pixels_to_fill = int(mask.sum())

        if pixels_to_fill > 0.1 * image.size:
            return ToolResult(
                image=image.copy(),
                tool_name=self.name,
                success=False,
                message=f"Mask too large ({pixels_to_fill} pixels, >{0.1*image.size:.0f}). "
                        "Biharmonic inpainting is only suitable for small regions.",
            )

        restored = inpaint_biharmonic(image.astype(np.float64), mask)
        return ToolResult(
            image=restored.astype(np.float32),
            tool_name=self.name,
            metadata={"pixels_inpainted": pixels_to_fill},
        )
