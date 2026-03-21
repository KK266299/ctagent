# ============================================================================
# 模块职责: 统一 LLM API Client — 封装 OpenAI / Anthropic / 本地模型 的调用接口
#   所有 LLM 交互的底层入口，上层 caller 通过此接口调用 LLM
#   后续可扩展: LLaMA-Factory 部署的本地模型通过 OpenAI-compatible API 接入
# 参考: AgenticIR — llm/api_client.py
#       4KAgent — model caller abstraction
#       LLaMA-Factory — API server interface
# ============================================================================
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 调用的标准化响应。"""
    text: str
    raw: Any = None
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class LLMConfig:
    """LLM 调用配置。"""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60


class BaseLLMClient(ABC):
    """LLM Client 抽象基类。

    所有 LLM 交互都通过此接口进行，包括:
    - 纯文本 chat
    - 带图像的 vision chat
    - 结构化输出 (JSON mode)
    """

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        """发送 chat 请求。

        Args:
            messages: OpenAI 格式的 messages 列表
            **kwargs: 额外参数 (temperature, max_tokens 等覆盖)

        Returns:
            LLMResponse 标准化响应
        """
        ...

    def chat_with_image(
        self,
        text: str,
        image_b64: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """带图像的 vision chat (默认实现构建 messages 后调用 chat)。"""
        messages = self._build_vision_messages(text, image_b64, system_prompt)
        return self.chat(messages, **kwargs)

    def _build_vision_messages(
        self, text: str, image_b64: str, system_prompt: str
    ) -> list[dict[str, Any]]:
        """构建含图像的 messages (子类可覆盖以适配不同 provider)。"""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        })
        return messages


class OpenAIClient(BaseLLMClient):
    """OpenAI API Client (含 OpenAI-compatible 本地部署)。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client: Any = None

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        temperature = kwargs.pop("temperature", self.config.temperature)
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)

        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            raw=response,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason or "",
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        import os
        self._client = OpenAI(
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=self.config.base_url,
        )
        return self._client


class AnthropicClient(BaseLLMClient):
    """Anthropic API Client。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        self._client: Any = None

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        system_msg = ""
        user_content: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            elif m["role"] == "user":
                content = m["content"]
                if isinstance(content, list):
                    user_content = content
                else:
                    user_content = [{"type": "text", "text": content}]

        response = client.messages.create(
            model=self.config.model,
            system=system_msg,
            messages=[{"role": "user", "content": user_content}],
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            **kwargs,
        )
        return LLMResponse(
            text=response.content[0].text,
            raw=response,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
            },
            finish_reason=response.stop_reason or "",
        )

    def _build_vision_messages(
        self, text: str, image_b64: str, system_prompt: str
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
            ],
        })
        return messages

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e
        import os
        self._client = anthropic.Anthropic(
            api_key=self.config.api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        return self._client


def create_client(config: LLMConfig | None = None) -> BaseLLMClient:
    """工厂函数: 根据 config.provider 创建对应 client。"""
    config = config or LLMConfig()
    if config.provider in ("openai", "local"):
        return OpenAIClient(config)
    elif config.provider == "anthropic":
        return AnthropicClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def encode_image_b64(image: Any) -> str:
    """将 numpy 图像数组编码为 base64 PNG 字符串。

    处理: HU 范围归一化 → uint8 → PNG → base64。
    此函数将 CT 图像编码逻辑集中在 llm 层，
    供 DiagnosisCaller / ClosedAPIAdapter 等共用。
    """
    import base64
    import io
    import numpy as np
    from PIL import Image

    arr = np.asarray(image)
    if arr.max() <= 1.0:
        img_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = arr.clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
