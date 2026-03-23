# ============================================================================
# 模块职责: 多能量投影转换 (pkev2kvp) + 泊松噪声仿真
#   将参考单能量 (kev) 下的投影转换为 X 射线管多色 (kVp) 投影
#   模拟真实 X 射线探测器的光子计数统计噪声
# 参考: ADN — +helper/pkev2kvp.m
# ============================================================================
from __future__ import annotations

import numpy as np

from dataset.mar.physics_params import ATTEN_MODE_COL


def pkev2kvp(
    proj_kev_all: np.ndarray,
    spectrum: np.ndarray,
    energies: np.ndarray,
    kev: int,
    mu_all: list[np.ndarray],
) -> np.ndarray:
    """单能 → 多能投影转换 (Beer-Lambert 定律 + 能谱加权)。

    对每个能量 ien，将 kev 参考能量下的投影按衰减系数比值缩放，
    然后按能谱加权叠加所有能量的透射强度，最后取负对数。

    Args:
        proj_kev_all: (views, bins, num_materials) 各材料在 kev 下的投影
        spectrum: (energy_max,) X 射线能谱强度分布
        energies: [energy_min, ..., energy_max] 能量网格
        kev: 参考单能量 (如 70 keV)
        mu_all: 每种材料的衰减系数表列表, 每个 shape (120, num_modes)

    Returns:
        proj_kvp: (views, bins) 多色投影
    """
    num_materials = proj_kev_all.shape[2]
    views, bins = proj_kev_all.shape[:2]

    proj_energy = np.zeros((views, bins), dtype=np.float64)
    spectrum_sum = 0.0

    for ien in energies:
        proj_total = np.zeros((views, bins), dtype=np.float64)
        for imat in range(num_materials):
            ratio = (
                mu_all[imat][ien - 1, ATTEN_MODE_COL]
                / mu_all[imat][kev - 1, ATTEN_MODE_COL]
            )
            proj_total += ratio * proj_kev_all[:, :, imat]
        proj_energy += spectrum[ien - 1] * np.exp(-proj_total)
        spectrum_sum += spectrum[ien - 1]

    proj_kvp = -np.log(proj_energy / spectrum_sum + 1e-30)
    return proj_kvp


def add_poisson_noise(
    proj_kvp: np.ndarray,
    photon_num: float = 2e7,
    scatter_photon: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """泊松噪声仿真 — 模拟 X 射线探测器光子计数统计噪声。

    expected = round(exp(-proj) × photon_num) + scatter
    actual = Poisson(expected)
    noisy_proj = -log(actual / photon_num)
    """
    if rng is None:
        rng = np.random.default_rng()

    expected = np.round(np.exp(-proj_kvp) * photon_num).astype(np.float64)
    expected += scatter_photon
    expected = np.maximum(expected, 1.0)

    actual = rng.poisson(expected).astype(np.float64)
    actual = np.maximum(actual, 1.0)

    proj_noisy = -np.log(actual / photon_num)
    return proj_noisy
