# ============================================================================
# 模块职责: 闭源多模态 API 诊断适配器 — 仅保留诊断语义接口
#   API 调用逻辑已委托给 llm/diagnosis_caller.py + llm/api_client.py
#   本模块职责:
#     1. 将 CT 图像编码为 base64
#     2. 调用 llm/DiagnosisCaller 获取 LLM 响应
#     3. 将 LLM 响应解析为 DiagnosisResult
#   向后兼容: 仍支持 APIConfig 构造，内部自动桥接到 llm/ 层
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — diagnosis adapter
#       4KAgent (https://github.com/taco-group/4KAgent) — multimodal agent
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.downstream.base import BaseDownstreamTask, DiagnosisResult
from src.downstream.prompt_builder import PromptBuilder
from src.downstream.response_parser import ResponseParser

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """闭源 API 配置 (保留向后兼容)。"""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 60


class ClosedAPIAdapter(BaseDownstreamTask):
    """闭源多模态 API 诊断适配器。

    两种构造方式:
    1. (旧) config-based:  ClosedAPIAdapter(config=APIConfig(...))
    2. (新) client-based:  ClosedAPIAdapter(llm_client=OpenAIClient(...))

    两种诊断模式:
    1. direct:         直接把 CT 图像发给 VLM 做诊断
    2. tool_augmented: 先用 MCP-style tools 提取信息，再让 VLM 综合诊断
    """

    def __init__(
        self,
        config: APIConfig | None = None,
        llm_client: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: ResponseParser | None = None,
    ) -> None:
        self.config = config or APIConfig()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or ResponseParser()
        self._llm_client = llm_client
        self._diagnosis_caller: Any | None = None

    @property
    def task_name(self) -> str:
        return f"closed_api_diagnosis_{self.config.provider}"

    # ------------------------------------------------------------------
    # 核心诊断接口
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray, **kwargs: Any) -> DiagnosisResult:
        """模式 1: 直接诊断 — 把图像编码后发给 VLM。"""
        caller = self._get_diagnosis_caller()
        image_b64 = self._encode_image(image)
        context = kwargs.get("context", "")
        response = caller.call_direct(image_b64, context=context)
        return self.response_parser.parse(response.text)

    def predict_with_tools(
        self,
        image: np.ndarray,
        tool_results: dict[str, Any],
        **kwargs: Any,
    ) -> DiagnosisResult:
        """模式 2: Tool-augmented 诊断 — 附加工具分析结果。"""
        caller = self._get_diagnosis_caller()
        image_b64 = self._encode_image(image)
        response = caller.call_with_tools(image_b64, tool_results)
        return self.response_parser.parse(response.text)

    def compare_diagnosis(
        self,
        original_image: np.ndarray,
        degraded_image: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, DiagnosisResult]:
        """对比实验: 分别对原始/退化图像做诊断。"""
        result_original = self.predict(original_image, context="original_ct", **kwargs)
        result_degraded = self.predict(degraded_image, context="degraded_ct", **kwargs)
        return {
            "original": result_original,
            "degraded": result_degraded,
        }

    # ------------------------------------------------------------------
    # llm/ 层桥接
    # ------------------------------------------------------------------

    def _get_diagnosis_caller(self) -> Any:
        """延迟初始化 DiagnosisCaller。"""
        if self._diagnosis_caller is not None:
            return self._diagnosis_caller

        from llm.diagnosis_caller import DiagnosisCaller

        client = self._llm_client or self._create_client_from_config()
        self._diagnosis_caller = DiagnosisCaller(
            llm_client=client,
            system_prompt=self.prompt_builder.system_prompt,
        )
        return self._diagnosis_caller

    def _create_client_from_config(self) -> Any:
        """从 APIConfig 创建 BaseLLMClient (向后兼容桥)。"""
        from llm.api_client import LLMConfig, create_client

        llm_config = LLMConfig(
            provider=self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )
        self._llm_client = create_client(llm_config)
        return self._llm_client

    # ------------------------------------------------------------------
    # 图像编码
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """将 numpy 数组编码为 base64 PNG 字符串。"""
        from llm.api_client import encode_image_b64
        return encode_image_b64(image)
