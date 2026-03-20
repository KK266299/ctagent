# ============================================================================
# 模块职责: 工具注册表 — 工具的注册、发现与实例化
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — tool registry
#       LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) — registry pattern
# ============================================================================
from __future__ import annotations

from typing import Any, Type

from src.tools.base import BaseTool


class ToolRegistry:
    """全局工具注册表。支持 decorator 注册和按名称查找。"""

    _registry: dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """注册一个工具类（可用作 decorator）。

        Usage:
            @ToolRegistry.register
            class MyTool(BaseTool):
                ...
        """
        instance = tool_class.__new__(tool_class)
        # 需要临时获取 name，使用无参构造
        name = tool_class.name.fget(instance) if hasattr(tool_class.name, 'fget') else None
        if name is None:
            name = tool_class.__name__
        cls._registry[name] = tool_class
        return tool_class

    @classmethod
    def get(cls, name: str) -> Type[BaseTool] | None:
        """按名称获取工具类。"""
        return cls._registry.get(name)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseTool:
        """创建工具实例。"""
        tool_class = cls._registry.get(name)
        if tool_class is None:
            raise KeyError(f"Tool not found: {name!r}. Available: {list(cls._registry.keys())}")
        return tool_class(**kwargs)

    @classmethod
    def list_tools(cls) -> list[str]:
        """列出所有已注册工具名。"""
        return list(cls._registry.keys())

    @classmethod
    def list_descriptions(cls) -> dict[str, str]:
        """列出所有工具名及描述。"""
        result = {}
        for name, tool_cls in cls._registry.items():
            try:
                inst = tool_cls.__new__(tool_cls)
                desc = tool_cls.description.fget(inst) if hasattr(tool_cls.description, 'fget') else ""
                result[name] = desc
            except Exception:
                result[name] = ""
        return result
