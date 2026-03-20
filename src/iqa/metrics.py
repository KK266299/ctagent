# ============================================================================
# 模块职责: 全参考 IQA 指标 — PSNR, SSIM, LPIPS 等
# 参考: IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch)
#       CAPIQA (https://github.com/aaz-imran/capiqa)
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """计算 PSNR (Peak Signal-to-Noise Ratio)。"""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """计算 SSIM (Structural Similarity Index)。"""
    from skimage.metrics import structural_similarity

    return float(structural_similarity(pred, target, data_range=data_range))


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    metric_names: list[str] | None = None,
    data_range: float = 1.0,
) -> dict[str, float]:
    """批量计算指标。"""
    if metric_names is None:
        metric_names = ["psnr", "ssim"]

    results: dict[str, float] = {}
    metric_fns: dict[str, Any] = {
        "psnr": lambda: compute_psnr(pred, target, data_range),
        "ssim": lambda: compute_ssim(pred, target, data_range),
    }
    for name in metric_names:
        fn = metric_fns.get(name)
        if fn is not None:
            results[name] = fn()
        else:
            # TODO: 支持 LPIPS, FID 等需要 torch 的指标
            results[name] = float("nan")
    return results
