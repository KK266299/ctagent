# ============================================================================
# 模块职责: 工具注册表 — 工具的注册、发现与实例化
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — tool registry
#       LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) — registry pattern
# ============================================================================
from __future__ import annotations

from typing import Any, Type

from src.tools.base import BaseTool, ToolMeta

# 安全性 / 代价排序用
_COST_ORDER = {"cheap": 0, "medium": 1, "expensive": 2}
_SAFETY_ORDER = {"safe": 0, "moderate": 1, "risky": 2}


class ToolRegistry:
    """全局工具注册表。支持 decorator 注册、按名称查找、按元信息筛选。"""

    _registry: dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """注册一个工具类（可用作 decorator）。"""
        instance = tool_class.__new__(tool_class)
        name = tool_class.name.fget(instance) if hasattr(tool_class.name, 'fget') else None
        if name is None:
            name = tool_class.__name__
        cls._registry[name] = tool_class
        return tool_class

    @classmethod
    def get(cls, name: str) -> Type[BaseTool] | None:
        return cls._registry.get(name)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseTool:
        tool_class = cls._registry.get(name)
        if tool_class is None:
            raise KeyError(f"Tool not found: {name!r}. Available: {list(cls._registry.keys())}")
        return tool_class(**kwargs)

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_descriptions(cls) -> dict[str, str]:
        result = {}
        for name, tool_cls in cls._registry.items():
            try:
                inst = tool_cls.__new__(tool_cls)
                desc = tool_cls.description.fget(inst) if hasattr(tool_cls.description, 'fget') else ""
                result[name] = desc
            except Exception:
                result[name] = ""
        return result

    @classmethod
    def _get_meta(cls, tool_cls: Type[BaseTool]) -> ToolMeta:
        try:
            inst = tool_cls.__new__(tool_cls)
            return inst.meta
        except Exception:
            return ToolMeta()

    @classmethod
    def list_tool_metas(cls) -> dict[str, dict[str, Any]]:
        """返回 {name: {category, suitable_for, cost, safety, description}} 字典。"""
        out: dict[str, dict[str, Any]] = {}
        for name, tool_cls in cls._registry.items():
            meta = cls._get_meta(tool_cls)
            try:
                inst = tool_cls.__new__(tool_cls)
                desc = tool_cls.description.fget(inst) if hasattr(tool_cls.description, 'fget') else ""
            except Exception:
                desc = ""
            out[name] = {
                "description": desc,
                "category": meta.category,
                "suitable_for": meta.suitable_for,
                "expected_cost": meta.expected_cost,
                "expected_safety": meta.expected_safety,
                "params_schema": meta.params_schema,
            }
        return out

    @classmethod
    def filter_by(
        cls,
        suitable_for: str | list[str] | None = None,
        category: str | None = None,
        max_cost: str | None = None,
        max_safety_risk: str | None = None,
        sort_by_cost: bool = True,
    ) -> list[str]:
        """按元信息筛选工具，返回排序后的工具名列表。"""
        if isinstance(suitable_for, str):
            suitable_for = [suitable_for]

        cost_limit = _COST_ORDER.get(max_cost, 999) if max_cost else 999
        safety_limit = _SAFETY_ORDER.get(max_safety_risk, 999) if max_safety_risk else 999

        candidates: list[tuple[int, int, str]] = []
        for name, tool_cls in cls._registry.items():
            meta = cls._get_meta(tool_cls)

            if suitable_for and not any(s in meta.suitable_for for s in suitable_for):
                continue
            if category and meta.category != category:
                continue
            c = _COST_ORDER.get(meta.expected_cost, 1)
            s = _SAFETY_ORDER.get(meta.expected_safety, 1)
            if c > cost_limit or s > safety_limit:
                continue
            candidates.append((c, s, name))

        if sort_by_cost:
            candidates.sort(key=lambda x: (x[0], x[1]))

        return [name for _, _, name in candidates]
