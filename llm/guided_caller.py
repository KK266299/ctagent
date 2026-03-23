# ============================================================================
# 模块职责: Guided Caller — API-guided planner 的 LLM 调用封装
#   输入: GuidedPlanRequest (degradation + scores + history + image)
#   输出: GuidedDecision (decision + plan + reason)
#   纯 I/O 层, 不含决策逻辑 — 决策由 LLM 完成, 本层只负责:
#     request → prompt → API call → raw text → parsed decision
#   支持 vision (iter#0 传图) 和 text-only (iter#1+ 传 scores) 两种模式
# 参考: 4KAgent — LLM-based planning + reflection
#       AgenticIR — llm/planner_caller with structured output
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm.api_client import BaseLLMClient, LLMConfig, encode_image_b64
from llm.prompt_builder import (
    TOOL_CATALOG,
    build_guided_system_prompt,
    build_guided_user_prompt,
)
from llm.response_parser import GuidedDecision, parse_guided_decision

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """单次 GuidedCaller.call() 的完整记录，供实验分析用。"""
    iteration: int = 0
    has_image: bool = False
    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: str = ""
    parsed_decision: str = ""
    selected_tools: list[str] = field(default_factory=list)
    selected_params: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "has_image": self.has_image,
            "system_prompt": self.system_prompt[:200] + "..." if len(self.system_prompt) > 200 else self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response,
            "parsed_decision": self.parsed_decision,
            "selected_tools": self.selected_tools,
            "selected_params": self.selected_params,
            "reason": self.reason,
            "model": self.model,
            "usage": self.usage,
        }


@dataclass
class GuidedPlanRequest:
    """API-guided planner 的请求数据。"""
    iteration: int = 0
    degradation_summary: dict[str, Any] = field(default_factory=dict)
    current_scores: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    image: np.ndarray | None = None


class GuidedCaller:
    """Guided Planner 的 LLM 调用层。

    Usage:
        client = create_client(LLMConfig(model="gpt-4o"))
        caller = GuidedCaller(client)
        decision = caller.call(request)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tool_catalog: list[dict[str, Any]] | None = None,
        use_json_mode: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.tool_catalog = tool_catalog or TOOL_CATALOG
        self.use_json_mode = use_json_mode
        self._system_prompt = build_guided_system_prompt(self.tool_catalog)
        self.last_record: CallRecord | None = None

    def call(self, request: GuidedPlanRequest) -> GuidedDecision:
        """发送请求给 LLM, 返回结构化决策。

        iter#0 且有 image → vision call (传图)
        iter#1+ 或无 image → text-only call
        每次调用后 self.last_record 会被更新。
        """
        has_image = request.image is not None and request.iteration == 0

        user_prompt = build_guided_user_prompt(
            iteration=request.iteration,
            degradation_summary=request.degradation_summary,
            current_scores=request.current_scores,
            history=request.history,
            with_image=has_image,
        )

        kwargs: dict[str, Any] = {}
        if self.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if has_image:
            image_b64 = encode_image_b64(request.image)
            response = self.llm_client.chat_with_image(
                text=user_prompt,
                image_b64=image_b64,
                system_prompt=self._system_prompt,
                **kwargs,
            )
        else:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = self.llm_client.chat(messages, **kwargs)

        logger.info(
            "GuidedCaller iter#%d: %d chars, model=%s, usage=%s",
            request.iteration, len(response.text), response.model, response.usage,
        )
        logger.info("─── LLM raw response ───\n%s\n─── end raw response ───", response.text)

        decision = parse_guided_decision(response.text)

        tools = [s.tool_name for s in decision.plan.steps] if decision.plan and decision.plan.steps else []
        params = [s.params for s in decision.plan.steps] if decision.plan and decision.plan.steps else []
        logger.info(
            "─── Parsed decision ─── decision=%s | tools=%s | params=%s | reason=%s",
            decision.decision, tools, params, decision.reason,
        )

        self.last_record = CallRecord(
            iteration=request.iteration,
            has_image=has_image,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            raw_response=response.text,
            parsed_decision=decision.decision,
            selected_tools=tools,
            selected_params=params,
            reason=decision.reason,
            model=response.model,
            usage=dict(response.usage),
        )

        return decision
