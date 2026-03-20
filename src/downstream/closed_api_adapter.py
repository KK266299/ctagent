# ============================================================================
# 模块职责: 闭源多模态 API 适配器 — 统一封装 GPT-4o / Claude 等 VLM API
#   支持：纯图像诊断 / tool-augmented 诊断 两种模式
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — API 调用
#       4KAgent (https://github.com/taco-group/4KAgent) — 多模态 agent
# ============================================================================
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.downstream.base import BaseDownstreamTask, DiagnosisResult
from src.downstream.prompt_builder import PromptBuilder
from src.downstream.response_parser import ResponseParser

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """闭源 API 配置。"""
    provider: str = "openai"            # "openai" | "anthropic"
    model: str = "gpt-4o"
    api_key: str | None = None          # 从环境变量或传入
    base_url: str | None = None         # 自定义 endpoint
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 60


class ClosedAPIAdapter(BaseDownstreamTask):
    """闭源多模态 API 适配器。

    两种工作模式：
    1. direct_diagnosis: 直接把 CT 图像发给 VLM 做诊断
    2. tool_augmented:   先用 MCP-style tools 提取信息，再让 VLM 综合诊断

    Usage:
        adapter = ClosedAPIAdapter(config=APIConfig(model="gpt-4o"))
        # 模式 1: 直接诊断
        result = adapter.predict(image)
        # 模式 2: tool-augmented 诊断
        result = adapter.predict_with_tools(image, tool_results={...})
    """

    def __init__(
        self,
        config: APIConfig | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: ResponseParser | None = None,
    ) -> None:
        self.config = config or APIConfig()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or ResponseParser()
        self._client: Any = None

    @property
    def task_name(self) -> str:
        return f"closed_api_diagnosis_{self.config.provider}"

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray, **kwargs: Any) -> DiagnosisResult:
        """模式 1: 直接诊断 — 把图像编码后发给 VLM。"""
        image_b64 = self._encode_image(image)
        prompt = self.prompt_builder.build_direct_diagnosis_prompt(**kwargs)
        messages = self._build_vision_messages(prompt, image_b64)
        raw_response = self._call_api(messages)
        return self.response_parser.parse(raw_response)

    def predict_with_tools(
        self,
        image: np.ndarray,
        tool_results: dict[str, Any],
        **kwargs: Any,
    ) -> DiagnosisResult:
        """模式 2: Tool-augmented 诊断 — 附加工具分析结果。"""
        image_b64 = self._encode_image(image)
        prompt = self.prompt_builder.build_tool_augmented_prompt(tool_results, **kwargs)
        messages = self._build_vision_messages(prompt, image_b64)
        raw_response = self._call_api(messages)
        return self.response_parser.parse(raw_response)

    def compare_diagnosis(
        self,
        original_image: np.ndarray,
        degraded_image: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, DiagnosisResult]:
        """对比实验: 分别对原始/退化图像做诊断，返回两份结果。"""
        result_original = self.predict(original_image, context="original_ct", **kwargs)
        result_degraded = self.predict(degraded_image, context="degraded_ct", **kwargs)
        return {
            "original": result_original,
            "degraded": result_degraded,
        }

    # ------------------------------------------------------------------
    # API 调用
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """延迟初始化 API client。"""
        if self._client is not None:
            return self._client

        if self.config.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("pip install openai") from e
            self._client = OpenAI(
                api_key=self.config.api_key or _get_env("OPENAI_API_KEY"),
                base_url=self.config.base_url,
            )
        elif self.config.provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError("pip install anthropic") from e
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key or _get_env("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        return self._client

    def _call_api(self, messages: list[dict]) -> str:
        """统一 API 调用，返回文本响应。"""
        client = self._get_client()

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""

        elif self.config.provider == "anthropic":
            # Anthropic 格式: system 单独传
            system_msg = ""
            user_content = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                elif m["role"] == "user":
                    user_content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
            response = client.messages.create(
                model=self.config.model,
                system=system_msg,
                messages=[{"role": "user", "content": user_content}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.content[0].text

        return ""

    # ------------------------------------------------------------------
    # 图像编码 / 消息构建
    # ------------------------------------------------------------------

    def _encode_image(self, image: np.ndarray) -> str:
        """将 numpy 数组编码为 base64 PNG 字符串。"""
        from PIL import Image
        import io

        # 归一化到 [0, 255]
        if image.max() <= 1.0:
            img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = image.clip(0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img_uint8)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_vision_messages(self, prompt: str, image_b64: str) -> list[dict]:
        """构建含图像的 messages 列表。"""
        system_prompt = self.prompt_builder.system_prompt

        if self.config.provider == "openai":
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ]
        elif self.config.provider == "anthropic":
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                    ],
                },
            ]
        return []


def _get_env(key: str) -> str:
    """从环境变量获取值。"""
    import os
    val = os.environ.get(key)
    if val is None:
        raise EnvironmentError(f"Environment variable {key} not set")
    return val
