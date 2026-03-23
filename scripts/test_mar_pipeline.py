#!/usr/bin/env python
"""MAR 流水线端到端验证 — 使用合成 phantom + 合成物理参数。

不需要真实 .mat 文件和 DeepLesion 数据，验证:
1. CT 几何 (前向投影 + FBP 重建)
2. 组织分解
3. 多能量转换 (pkev2kvp)
4. 泊松噪声
5. BHC
6. 金属伪影注入 + LI/BHC 校正
7. HDF5 保存 + Dataset 加载

用法:
    PYTHONPATH=. python scripts/test_mar_pipeline.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_mar")


def make_synthetic_phantom(size: int = 416) -> np.ndarray:
    """合成 Shepp-Logan 风格的 HU phantom。"""
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r = np.sqrt(x**2 + y**2)

    phantom = np.full((size, size), -1000.0, dtype=np.float64)  # air

    body_mask = r < size * 0.4
    phantom[body_mask] = 0.0  # soft tissue = 0 HU

    bone_mask = (r < size * 0.15) & (r > size * 0.1)
    phantom[bone_mask] = 800.0

    lesion_y, lesion_x = size // 2 + 30, size // 2 + 50
    lesion_mask = ((y - 30) ** 2 + (x - 50) ** 2) < (20**2)
    phantom[lesion_mask] = 50.0

    return phantom


def make_synthetic_metal_mask(size: int = 416) -> np.ndarray:
    """合成圆形金属植入物掩模。"""
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    mask = np.zeros((size, size), dtype=np.float32)
    r = np.sqrt((x + 60) ** 2 + (y - 40) ** 2)
    mask[r < 8] = 1.0
    return mask


def make_mock_mu_table(n_energies: int = 120) -> np.ndarray:
    """合成衰减系数表 (120, 7)，模拟真实 .mat 的结构。"""
    mu = np.zeros((n_energies, 7), dtype=np.float64)
    for col in range(7):
        base = 0.15 + col * 0.01
        mu[:, col] = base * np.exp(-0.01 * np.arange(n_energies))
    mu[69, 6] = 0.192  # water @ 70keV column 6
    return mu


def test_geometry():
    """测试 CT 几何的前向/反向投影。"""
    logger.info("=== Test 1: CT Geometry ===")
    from dataset.mar.ct_geometry import CTGeometry, CTGeometryConfig

    cfg = CTGeometryConfig(impl="astra_cuda")
    geo = CTGeometry(cfg)

    phantom_hu = make_synthetic_phantom(cfg.image_size)
    mu = phantom_hu / 1000.0 * 0.192 + 0.192

    sino = geo.forward(mu)
    recon = geo.fbp(sino)

    logger.info("  sinogram shape: %s, range: [%.4f, %.4f]", sino.shape, sino.min(), sino.max())
    logger.info("  recon shape: %s, range: [%.4f, %.4f]", recon.shape, recon.min(), recon.max())

    assert sino.shape == (640, 641), f"unexpected sino shape {sino.shape}"
    assert recon.shape == (416, 416), f"unexpected recon shape {recon.shape}"
    assert sino.max() > 0, "sinogram should have positive values"
    logger.info("  PASSED")
    return geo


def test_tissue_decompose():
    """测试组织分解。"""
    logger.info("=== Test 2: Tissue Decompose ===")
    from dataset.mar.tissue_decompose import decompose_tissue, hu_to_mu

    phantom = make_synthetic_phantom()
    mu = hu_to_mu(phantom)
    thresh_w = 100 / 1000 * 0.192 + 0.192
    thresh_b = 1500 / 1000 * 0.192 + 0.192

    water, bone = decompose_tissue(mu, thresh_w, thresh_b)

    logger.info("  water range: [%.4f, %.4f]", water.min(), water.max())
    logger.info("  bone range: [%.4f, %.4f]", bone.min(), bone.max())
    assert water.shape == mu.shape
    assert bone.shape == mu.shape
    logger.info("  PASSED")
    return water, bone


def test_energy_convert():
    """测试多能量转换 + 泊松噪声。"""
    logger.info("=== Test 3: Energy Convert ===")
    from dataset.mar.energy_convert import add_poisson_noise, pkev2kvp

    views, bins = 640, 641
    p_kev = np.random.rand(views, bins).astype(np.float64) * 0.5
    proj_all = np.stack([p_kev, p_kev * 0.3], axis=-1)

    mu_water = make_mock_mu_table()
    mu_bone = make_mock_mu_table() * 1.5

    energies = np.arange(20, 121)
    spectrum = np.ones(120) * 0.01
    spectrum[60:80] = 0.05

    proj_kvp = pkev2kvp(proj_all, spectrum, energies, 70, [mu_water, mu_bone])
    logger.info("  proj_kvp shape: %s, range: [%.4f, %.4f]", proj_kvp.shape, proj_kvp.min(), proj_kvp.max())

    proj_noisy = add_poisson_noise(proj_kvp, photon_num=1e5)
    logger.info("  noisy proj range: [%.4f, %.4f]", proj_noisy.min(), proj_noisy.max())

    assert proj_kvp.shape == (views, bins)
    assert not np.allclose(proj_kvp, proj_noisy)
    logger.info("  PASSED")


def test_sinogram_utils():
    """测试正弦图校正方法。"""
    logger.info("=== Test 4: Sinogram Utils ===")
    from dataset.mar.sinogram_utils import (
        apply_bhc,
        apply_partial_volume_effect,
        compute_metal_trace,
        interpolate_projection,
    )

    views, bins = 64, 65
    sino = np.random.rand(views, bins).astype(np.float64) * 2.0

    para_bhc = np.array([[1.0], [0.01], [-0.001]])
    bhc_result = apply_bhc(sino, para_bhc)
    logger.info("  BHC shape: %s", bhc_result.shape)

    metal_proj = np.zeros((views, bins), dtype=np.float64)
    metal_proj[:, 30:35] = 5.0
    trace = compute_metal_trace(metal_proj)
    logger.info("  metal trace sum: %d", trace.sum())

    pve = apply_partial_volume_effect(metal_proj)
    assert pve.max() <= metal_proj.max()
    logger.info("  PVE applied, max: %.4f -> %.4f", metal_proj.max(), pve.max())

    li = interpolate_projection(sino, trace)
    logger.info("  LI interpolation done, diff at metal: %.4f", np.abs(li[:, 32] - sino[:, 32]).mean())

    logger.info("  PASSED")


def test_full_pipeline(geo):
    """完整流水线测试 (合成数据)。"""
    logger.info("=== Test 5: Full Pipeline (Synthetic) ===")
    from dataset.mar.physics_params import ATTEN_MODE_COL, PhysicsParams, PhysicsConfig
    from dataset.mar.mar_simulator import MARSimulator
    from dataset.mar.tissue_decompose import hu_to_mu

    phy = PhysicsParams.__new__(PhysicsParams)
    phy.config = PhysicsConfig()
    phy.mu_water = make_mock_mu_table()
    phy.mu_bone = make_mock_mu_table() * 1.5

    mu_ti = make_mock_mu_table() * 10.0
    mu_fe = make_mock_mu_table() * 15.0
    mu_cu = make_mock_mu_table() * 12.0
    mu_au = make_mock_mu_table() * 20.0
    phy.mu_metals = np.stack([mu_ti, mu_fe, mu_cu, mu_au], axis=-1)

    spectrum = np.ones(120) * 0.01
    spectrum[60:80] = 0.05
    phy.spectrum = spectrum
    phy.energies = np.arange(20, 121)
    phy.thresh_water = 100 / 1000 * 0.192 + 0.192
    phy.thresh_bone = 1500 / 1000 * 0.192 + 0.192
    phy.metal_atten = 4.5 * mu_ti[69, ATTEN_MODE_COL]

    para_bhc = np.array([[1.0], [0.001], [-0.0001]])
    phy.para_bhc = para_bhc

    simulator = MARSimulator(geo, phy, seed=42)

    phantom_hu = make_synthetic_phantom(geo.config.image_size)
    metal_mask = make_synthetic_metal_mask(geo.config.image_size)

    logger.info("  Running simulation ...")
    result = simulator.simulate(phantom_hu, [metal_mask])

    logger.info("  gt_ct range: [%.4f, %.4f]", result.gt_ct.min(), result.gt_ct.max())
    logger.info("  poly_ct range: [%.4f, %.4f]", result.poly_ct.min(), result.poly_ct.max())
    assert len(result.mask_results) == 1

    mr = result.mask_results[0]
    logger.info("  ma_CT range: [%.4f, %.4f]", mr["ma_CT"].min(), mr["ma_CT"].max())
    logger.info("  LI_CT range: [%.4f, %.4f]", mr["LI_CT"].min(), mr["LI_CT"].max())
    logger.info("  BHC_CT range: [%.4f, %.4f]", mr["BHC_CT"].min(), mr["BHC_CT"].max())

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_case"
        simulator.save_h5(result, out_dir)

        import h5py
        with h5py.File(str(out_dir / "gt.h5"), "r") as f:
            assert "image" in f
            assert "poly_sinogram" in f
            assert "poly_CT" in f
            logger.info("  gt.h5 keys: %s", list(f.keys()))

        with h5py.File(str(out_dir / "0.h5"), "r") as f:
            assert "ma_CT" in f
            assert "LI_CT" in f
            assert "BHC_CT" in f
            assert "metal_trace" in f
            logger.info("  0.h5 keys: %s", list(f.keys()))

        from dataset.mar.mar_dataset import MARDataset
        ds = MARDataset(data_dir=out_dir.parent, split="train")
        assert len(ds) >= 1, f"Dataset should have samples, got {len(ds)}"
        gt_t, ma_t, li_t, meta = ds[0]
        logger.info("  Dataset sample: gt=%s ma=%s", gt_t.shape, ma_t.shape)
        assert gt_t.shape == (1, 416, 416)

    logger.info("  PASSED")


def main():
    logger.info("Starting MAR pipeline verification ...")

    geo = test_geometry()
    test_tissue_decompose()
    test_energy_convert()
    test_sinogram_utils()
    test_full_pipeline(geo)

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
