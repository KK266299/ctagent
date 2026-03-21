# ============================================================================
# 模块职责: Diagnosis Caller — 通过 VLM 对 CT 图像进行诊断
#   封装: 图像编码 → prompt 构建 → VLM 调用 → DiagnosisResult 解析
#   此模块是 downstream/closed_api_adapter.py 的 LLM 交互后端
#   当前阶段: 定义接口骨架，实际调用仍由 src/downstream/closed_api_adapter.py 完成
#   后续阶段(Phase 2): closed_api_adapter.py 瘦身后将 API 调用委托给此模块
# 参考: MedAgent-Pro — VLM diagnosis calling
#       4KAgent — multimodal agent calling
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

from llm.api_client import BaseLLMClient, LLMResponse
from llm.prompt_builder import DIAGNOSIS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DiagnosisCaller:
    """VLM Diagnosis Caller — 将 CT 图像发送给 VLM 获取诊断结果。

    两种调用模式:
    1. direct: 纯图像 → VLM 诊断
    2. tool_augmented: 图像 + 工具分析结果 → VLM 诊断

    当前为接口骨架，后续 Phase 2 会从 closed_api_adapter.py 迁入实际逻辑。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        system_prompt: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt = system_prompt or DIAGNOSIS_SYSTEM_PROMPT

    def call_direct(self, image_b64: str, context: str = "") -> LLMResponse:
        """模式 1: 直接诊断 — 发送图像给 VLM。"""
        text = "Please analyze the following CT image and provide a diagnostic assessment."
        if context:
            text += f"\nImage context: {context}"
        return self.llm_client.chat_with_image(
            text=text,
            image_b64=image_b64,
            system_prompt=self.system_prompt,
        )

    def call_with_tools(
        self,
        image_b64: str,
        tool_results: dict[str, Any],
    ) -> LLMResponse:
        """模式 2: Tool-augmented 诊断 — 附加工具分析结果。"""
        import json
        tool_text = "\n".join(
            f"### {name}\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```"
            for name, result in tool_results.items()
        )
        text = (
            "Please analyze the following CT image. "
            "In addition to the image, the following automated analysis results are available:\n\n"
            f"{tool_text}\n\n"
            "Use both the image and the tool results to provide a comprehensive diagnostic assessment."
        )
        return self.llm_client.chat_with_image(
            text=text,
            image_b64=image_b64,
            system_prompt=self.system_prompt,
        )
