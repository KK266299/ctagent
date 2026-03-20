# ============================================================================
# 模块职责: 配置管理 — YAML 配置加载与合并
# 参考: LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) — config pattern
# ============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """深度合并多个配置字典（后者覆盖前者）。"""
    result: dict[str, Any] = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """递归合并字典。"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
