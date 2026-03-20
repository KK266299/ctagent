# ============================================================================
# 模块职责: API 响应解析器 — 将 VLM 的文本响应解析为结构化 DiagnosisResult
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — response parsing
# ============================================================================
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.downstream.base import DiagnosisResult

logger = logging.getLogger(__name__)


class ResponseParser:
    """解析 VLM API 返回的诊断响应。"""

    def parse(self, raw_response: str) -> DiagnosisResult:
        """将 API 原始文本响应解析为 DiagnosisResult。

        尝试顺序:
        1. 提取 JSON 块并解析
        2. 正则提取关键字段
        3. 降级为纯文本结果
        """
        # 尝试 1: JSON 解析
        parsed = self._try_parse_json(raw_response)
        if parsed is not None:
            return self._json_to_result(parsed, raw_response)

        # 尝试 2: 正则提取
        extracted = self._try_regex_extract(raw_response)
        if extracted:
            return DiagnosisResult(
                prediction=extracted.get("diagnosis", raw_response),
                confidence=extracted.get("confidence", 0.0),
                metadata={"raw_response": raw_response, "parse_method": "regex", **extracted},
            )

        # 尝试 3: 降级
        logger.warning("Failed to parse structured response, using raw text.")
        return DiagnosisResult(
            prediction=raw_response.strip(),
            confidence=0.0,
            metadata={"raw_response": raw_response, "parse_method": "fallback"},
        )

    def _try_parse_json(self, text: str) -> dict | None:
        """尝试从文本中提取 JSON 对象。"""
        # 匹配 ```json ... ``` 或 { ... }
        patterns = [
            r"```json\s*\n?(.*?)\n?\s*```",
            r"```\s*\n?(.*?)\n?\s*```",
            r"(\{[^{}]*\})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        # 直接尝试整段解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _json_to_result(self, data: dict, raw: str) -> DiagnosisResult:
        """JSON dict → DiagnosisResult。"""
        return DiagnosisResult(
            prediction=data.get("diagnosis", data.get("prediction", "")),
            confidence=float(data.get("confidence", 0.0)),
            metadata={
                "findings": data.get("findings", []),
                "severity": data.get("severity", ""),
                "reasoning": data.get("reasoning", ""),
                "raw_response": raw,
                "parse_method": "json",
            },
        )

    def _try_regex_extract(self, text: str) -> dict[str, Any]:
        """正则提取关键字段。"""
        result: dict[str, Any] = {}
        # diagnosis
        m = re.search(r"diagnosis[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            result["diagnosis"] = m.group(1).strip().strip('"')
        # confidence
        m = re.search(r"confidence[:\s]+([\d.]+)", text, re.IGNORECASE)
        if m:
            result["confidence"] = float(m.group(1))
        # severity
        m = re.search(r"severity[:\s]+(\w+)", text, re.IGNORECASE)
        if m:
            result["severity"] = m.group(1).strip()
        return result
