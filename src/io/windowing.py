# ============================================================================
# 模块职责: CT 窗宽窗位处理 — HU 值到显示值的映射
# 参考: ProCT (https://github.com/Masaaki-75/proct)
# ============================================================================
from __future__ import annotations

import numpy as np


# 常用 CT 窗口预设
WINDOW_PRESETS = {
    "soft_tissue": {"center": 40, "width": 400},
    "lung": {"center": -600, "width": 1500},
    "bone": {"center": 400, "width": 1800},
    "brain": {"center": 40, "width": 80},
    "liver": {"center": 60, "width": 150},
}


def apply_window(
    image: np.ndarray,
    center: float = 40,
    width: float = 400,
    normalize: bool = True,
) -> np.ndarray:
    """对 HU 值图像应用窗宽窗位变换。

    Args:
        image: HU 值数组
        center: 窗位
        width: 窗宽
        normalize: 是否归一化到 [0, 1]

    Returns:
        窗口化后的图像
    """
    lower = center - width / 2
    upper = center + width / 2
    result = np.clip(image, lower, upper)
    if normalize:
        result = (result - lower) / (upper - lower)
    return result
