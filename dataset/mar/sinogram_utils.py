# ============================================================================
# 模块职责: 正弦图校正方法 — BHC 三阶多项式、线性插值 (LI)、MAR-BHC
#   用于生成各种校正结果：ma_CT, LI_CT, BHC_CT
# 参考: ADN — +helper/interpolate_projection.m, +helper/simulate_metal_artifact.m
# ============================================================================
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def apply_bhc(
    sinogram: np.ndarray,
    para_bhc: np.ndarray,
) -> np.ndarray:
    """水基射束硬化校正 (BHC) — 三阶多项式。

    BHC(p) = c1*p + c2*p² + c3*p³
    """
    p = sinogram.ravel().reshape(-1, 1)
    A = np.concatenate([p, p**2, p**3], axis=1)
    corrected = (A @ para_bhc).reshape(sinogram.shape)
    return corrected.astype(np.float32)


def interpolate_projection(
    sinogram: np.ndarray,
    metal_trace: np.ndarray,
) -> np.ndarray:
    """线性插值 (LI) 校正 — 用非金属位置的值插值替代金属位置。

    对正弦图每行：找到金属/非金属位置，用非金属值线性插值填充金属处。
    """
    result = sinogram.copy()
    views, bins = sinogram.shape

    for i in range(views):
        trace_row = metal_trace[i]
        metal_pos = np.where(trace_row > 0)[0]
        if len(metal_pos) == 0:
            continue

        non_metal_pos = np.where(trace_row == 0)[0]
        if len(non_metal_pos) < 2:
            continue

        result[i, metal_pos] = np.interp(
            metal_pos.astype(np.float64),
            non_metal_pos.astype(np.float64),
            sinogram[i, non_metal_pos],
        )

    return result


def compute_metal_trace(
    proj_metal: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """计算金属在正弦图中的投影轨迹 (二值掩模)。"""
    return (proj_metal > threshold).astype(np.uint8)


def apply_partial_volume_effect(
    proj_metal: np.ndarray,
    edge_fraction: float = 0.25,
) -> np.ndarray:
    """部分体积效应 (PVE) 校正 — 金属边缘衰减降低。

    金属边缘像素只占部分体素，腐蚀后取 XOR 得到边缘区域，
    边缘区域的投影值乘以 edge_fraction。
    """
    result = proj_metal.copy()
    metal_bw = proj_metal > 0

    struct = np.ones((1, 3))
    metal_eroded = binary_erosion(metal_bw, structure=struct)
    metal_edge = np.logical_xor(metal_bw, metal_eroded)

    result[metal_edge] *= edge_fraction
    return result


def mar_bhc(
    sinogram: np.ndarray,
    metal_mask_2d: np.ndarray,
    forward_fn,
    fbp_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """MAR 射束硬化校正。

    1. 对金属掩模做前向投影
    2. 对原始正弦图做 LI 校正
    3. 计算残差 = 原始 - LI
    4. 在金属投影区域做三阶多项式最小二乘拟合
    5. 用拟合结果校正正弦图
    6. FBP 重建

    Returns:
        (bhc_ct, bhc_sinogram)
    """
    proj_metal = forward_fn(metal_mask_2d)
    metal_trace = proj_metal > 0

    proj_interp = interpolate_projection(sinogram, metal_trace.astype(np.uint8))
    proj_diff = sinogram - proj_interp

    metal_indices = np.where(metal_trace.ravel())[0]
    if len(metal_indices) < 10:
        return fbp_fn(sinogram), sinogram

    proj_m_flat = proj_metal.ravel()[metal_indices]
    diff_flat = proj_diff.ravel()[metal_indices]

    A = np.column_stack([proj_m_flat, proj_m_flat**2, proj_m_flat**3])
    x0, _, _, _ = np.linalg.lstsq(A, diff_flat, rcond=None)

    proj_metal_flat = proj_metal.ravel()
    proj_delta_flat = (
        x0[0] * proj_metal_flat
        - (x0[0] * proj_metal_flat + x0[1] * proj_metal_flat**2 + x0[2] * proj_metal_flat**3)
    )
    proj_delta = proj_delta_flat.reshape(sinogram.shape)

    proj_bhc = sinogram + proj_delta
    bhc_ct = fbp_fn(proj_bhc)

    return bhc_ct, proj_bhc.astype(np.float32)
