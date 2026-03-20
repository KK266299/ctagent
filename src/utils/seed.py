# ============================================================================
# 模块职责: 随机种子管理 — 确保可复现性
# ============================================================================
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """设定全局随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
