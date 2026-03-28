# ============================================================================
# 模块职责: 修复工具 (MCP 封装) — 调用已有工具链做图像修复，返回修复结果摘要
#   与 src/tools/ 下的 BaseTool 不同：此工具面向 LLM Agent 调用，
#   返回结构化 JSON (含修复前后 IQA 对比)，而非仅返回图像
# 参考: Earth-Agent (https://github.com/opendatalab/Earth-Agent/tree/main/agent/tools)
#       JarvisIR (https://github.com/LYL1015/JarvisIR) — tool orchestration
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.registry import ToolRegistry
from src.iqa.no_reference import NoReferenceIQA


class RestorationTool:
    """MCP-style 修复工具 — 调用底层修复工具并返回结构化报告。

    这是 src/tools/ 中具体修复工具的 MCP 封装层。
    """

    name = "ct_image_restoration"
    description = (
        "Apply a specified restoration tool to a CT image. "
        "Returns the restored image along with before/after quality comparison. "
        "Available tools: denoise_nlm, denoise_gaussian, sharpen_usm, "
        "histogram_clahe, ldct_denoiser, mar_rise, sr_ct."
    )

    parameters_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "Image identifier"},
            "tool_name": {"type": "string", "description": "Name of restoration tool to apply"},
            "params": {"type": "object", "description": "Tool-specific parameters"},
        },
        "required": ["image", "tool_name"],
    }

    def __init__(self) -> None:
        self.nr_iqa = NoReferenceIQA()

    def __call__(
        self,
        image: np.ndarray,
        tool_name: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """执行修复并返回结构化报告。"""
        params = params or {}

        # 修复前指标
        before_iqa = self.nr_iqa.evaluate(image)

        # 调用底层工具
        try:
            tool = ToolRegistry.create(tool_name)
            tool_result = tool.run(image, **params)
        except Exception as e:
            return {
                "tool": self.name,
                "success": False,
                "error": str(e),
                "applied_tool": tool_name,
            }

        # 修复后指标
        after_iqa = self.nr_iqa.evaluate(tool_result.image)

        return {
            "tool": self.name,
            "success": tool_result.success,
            "applied_tool": tool_name,
            "message": tool_result.message,
            "quality_before": before_iqa,
            "quality_after": after_iqa,
            "improvement": {
                k: after_iqa[k] - before_iqa[k] for k in before_iqa if k in after_iqa
            },
            # 修复后图像存在 metadata 中，不序列化到 JSON
            "_restored_image": tool_result.image,
        }

    def apply_chain(
        self,
        image: np.ndarray,
        steps: list[dict[str, Any]],
        reference: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """链式修复 — 按顺序执行多个工具, 跟踪每步质量变化。

        Args:
            image: 输入图像 (μ 值空间)
            steps: [{"tool_name": str, "params": dict}, ...]
            reference: GT 参考图 (可选, 有则计算 PSNR/SSIM)

        Returns:
            {
                "success": bool,
                "steps_executed": int,
                "quality_trace": [...],
                "overall_improvement": {...},
                "_restored_image": ndarray,
            }
        """
        from src.iqa.metrics import compute_psnr, compute_ssim

        current = image.copy()
        quality_trace: list[dict[str, Any]] = []
        any_success = False

        before_nr = self.nr_iqa.evaluate(current)
        before_fr: dict[str, float] = {}
        data_range = float(max(reference.max() - reference.min(), 1e-10)) if reference is not None else 1.0
        if reference is not None:
            before_fr = {
                "psnr": compute_psnr(current, reference, data_range),
                "ssim": compute_ssim(current, reference, data_range),
            }

        prev_ssim = before_fr.get("ssim", None)

        for i, step in enumerate(steps):
            tool_name = step.get("tool_name", "")
            params = step.get("params", {})

            step_record: dict[str, Any] = {
                "step": i,
                "tool_name": tool_name,
                "params": params,
                "success": False,
            }

            q_before = self.nr_iqa.evaluate(current)
            step_record["quality_before"] = q_before
            snapshot = current.copy()

            try:
                tool = ToolRegistry.create(tool_name)
                tool_result = tool.run(current, **params)
                if tool_result.success and tool_result.image is not None:
                    candidate = tool_result.image
                    if reference is not None and prev_ssim is not None:
                        step_ssim = compute_ssim(candidate, reference, data_range)
                        step_psnr = compute_psnr(candidate, reference, data_range)
                        prev_psnr = compute_psnr(current, reference, data_range)
                        ssim_drop = prev_ssim - step_ssim
                        psnr_drop = prev_psnr - step_psnr
                        if ssim_drop > 0.03 or psnr_drop > 2.0:
                            step_record["reverted"] = True
                            step_record["reason"] = (
                                f"Quality dropped SSIM {prev_ssim:.4f}->{step_ssim:.4f} "
                                f"PSNR {prev_psnr:.2f}->{step_psnr:.2f}, reverted"
                            )
                            quality_trace.append(step_record)
                            continue
                    current = candidate
                    step_record["success"] = True
                    any_success = True
                    step_record["message"] = tool_result.message
                else:
                    step_record["error"] = tool_result.message or "tool returned failure"
            except Exception as e:
                step_record["error"] = str(e)

            q_after = self.nr_iqa.evaluate(current)
            step_record["quality_after"] = q_after

            if reference is not None:
                step_record["psnr"] = compute_psnr(current, reference, data_range)
                step_record["ssim"] = compute_ssim(current, reference, data_range)
                prev_ssim = step_record["ssim"]

            quality_trace.append(step_record)

        after_nr = self.nr_iqa.evaluate(current)
        after_fr: dict[str, float] = {}
        if reference is not None:
            after_fr = {
                "psnr": compute_psnr(current, reference, data_range),
                "ssim": compute_ssim(current, reference, data_range),
            }

        overall = {
            k: after_nr[k] - before_nr[k] for k in before_nr if k in after_nr
        }
        if before_fr and after_fr:
            overall["psnr_delta"] = after_fr["psnr"] - before_fr["psnr"]
            overall["ssim_delta"] = after_fr["ssim"] - before_fr["ssim"]

        return {
            "success": any_success,
            "steps_executed": len(quality_trace),
            "quality_trace": quality_trace,
            "quality_before": {**before_nr, **before_fr},
            "quality_after": {**after_nr, **after_fr},
            "overall_improvement": overall,
            "_restored_image": current,
        }

    def list_available_tools(self) -> list[str]:
        """列出所有可用修复工具。"""
        return ToolRegistry.list_tools()
