# ============================================================================
# 模块职责: MAR 仿真主流水线 — 7 步物理仿真
#   输入: 干净 CT (HU) + 金属掩模 + 物理参数
#   输出: gt.h5 + {mask_idx}.h5 (含 ma_CT, LI_CT, BHC_CT, sinograms, metal_trace)
# 参考: ADN — +helper/simulate_metal_artifact.m
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from dataset.mar.ct_geometry import CTGeometry
from dataset.mar.energy_convert import add_poisson_noise, pkev2kvp
from dataset.mar.physics_params import ATTEN_MODE_COL, PhysicsParams
from dataset.mar.sinogram_utils import (
    apply_bhc,
    apply_partial_volume_effect,
    compute_metal_trace,
    interpolate_projection,
    mar_bhc,
)
from dataset.mar.tissue_decompose import decompose_tissue, hu_to_mu

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """单张 CT 的仿真结果。"""
    gt_ct: np.ndarray
    poly_sinogram: np.ndarray
    poly_ct: np.ndarray
    mask_results: list[dict[str, Any]]


class MARSimulator:
    """MAR 仿真器 — 完整的 7 步物理仿真流水线。

    Usage:
        geo = CTGeometry(geo_config)
        params = PhysicsParams(physics_config)
        params.load()
        simulator = MARSimulator(geo, params)
        result = simulator.simulate(hu_image, metal_masks)
        simulator.save_h5(result, output_dir)
    """

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        self.geo = geometry
        self.phy = physics
        self.rng = np.random.default_rng(seed)
        self.minimal = minimal

    def simulate(
        self,
        hu_image: np.ndarray,
        metal_masks: list[np.ndarray],
    ) -> SimulationResult:
        """运行完整 7 步仿真流水线。

        Args:
            hu_image: (H, W) HU 值域 CT 图像
            metal_masks: 金属掩模列表，每个 (H', W') 二值图
        """
        cfg = self.phy.config
        target_size = self.geo.config.image_size

        # --- Step 1: HU → 线衰减系数 ---
        mu_image = hu_to_mu(hu_image, cfg.mu_water_70kev)
        gt_ct = mu_image.astype(np.float32)

        # --- Step 2: 组织分解 ---
        img_water, img_bone = decompose_tissue(
            mu_image, self.phy.thresh_water, self.phy.thresh_bone,
        )

        # --- Step 3: 前向投影 ---
        p_water_kev = self.geo.forward(img_water)
        p_bone_kev = self.geo.forward(img_bone)

        # --- Step 4: 单能 → 多能 (无金属) ---
        proj_kev_2mat = np.stack([p_water_kev, p_bone_kev], axis=-1)
        mu_list_2 = [self.phy.mu_water, self.phy.mu_bone]

        proj_kvp = pkev2kvp(
            proj_kev_2mat, self.phy.spectrum, self.phy.energies,
            cfg.kev, mu_list_2,
        )

        # --- Step 5: 泊松噪声 ---
        proj_kvp_noise = add_poisson_noise(
            proj_kvp, cfg.photon_num, cfg.scatter_photon, self.rng,
        )

        # --- Step 6: 水基 BHC ---
        poly_sinogram = apply_bhc(proj_kvp_noise, self.phy.para_bhc)
        poly_ct = self.geo.fbp(poly_sinogram)

        # --- Step 7: 逐掩模金属伪影注入 ---
        mask_results = []
        for mask_idx, mask_raw in enumerate(metal_masks):
            try:
                mr = self._simulate_one_mask(
                    mask_raw, mask_idx, target_size,
                    p_water_kev, p_bone_kev, mu_list_2,
                    poly_sinogram,
                )
                mask_results.append(mr)
            except Exception as e:
                logger.warning("Mask %d failed: %s", mask_idx, e)

        return SimulationResult(
            gt_ct=gt_ct,
            poly_sinogram=poly_sinogram,
            poly_ct=poly_ct.astype(np.float32),
            mask_results=mask_results,
        )

    def _simulate_one_mask(
        self,
        mask_raw: np.ndarray,
        mask_idx: int,
        target_size: int,
        p_water_kev: np.ndarray,
        p_bone_kev: np.ndarray,
        mu_list_2: list[np.ndarray],
        poly_sinogram: np.ndarray,
    ) -> dict[str, Any]:
        """对单个金属掩模执行伪影注入 + 校正。"""
        cfg = self.phy.config

        # 7a: 缩放掩模 + 前向投影
        if mask_raw.shape[0] != target_size or mask_raw.shape[1] != target_size:
            mask_pil = Image.fromarray(mask_raw.astype(np.float32))
            mask_pil = mask_pil.resize((target_size, target_size), Image.NEAREST)
            metal_mask = np.array(mask_pil).astype(np.float32)
        else:
            metal_mask = mask_raw.astype(np.float32)

        metal_mask_bw = (metal_mask > 0.5).astype(np.float32)
        p_metal_kev = self.geo.forward(metal_mask_bw)
        metal_trace = compute_metal_trace(p_metal_kev)

        # 7b: 部分体积效应
        p_metal_kev_atten = p_metal_kev * self.phy.metal_atten
        p_metal_kev_pve = apply_partial_volume_effect(p_metal_kev_atten)

        # 7c: 含金属多能投影
        proj_kev_3mat = np.stack([p_water_kev, p_bone_kev, p_metal_kev_pve], axis=-1)

        mid = cfg.material_id
        mu_metal_single = self.phy.mu_metals[:, :, mid]
        mu_list_3 = [self.phy.mu_water, self.phy.mu_bone, mu_metal_single]

        proj_kvp_metal = pkev2kvp(
            proj_kev_3mat, self.phy.spectrum, self.phy.energies,
            cfg.kev, mu_list_3,
        )

        proj_kvp_metal_noise = add_poisson_noise(
            proj_kvp_metal, cfg.photon_num, cfg.scatter_photon, self.rng,
        )

        # 7d-1: 直接重建 (ma_CT)
        ma_sinogram = apply_bhc(proj_kvp_metal_noise, self.phy.para_bhc)
        ma_ct = self.geo.fbp(ma_sinogram)

        result = {
            "mask_idx": mask_idx,
            "ma_CT": ma_ct.astype(np.float32),
        }

        if not self.minimal:
            li_sinogram = interpolate_projection(ma_sinogram, metal_trace)
            li_ct = self.geo.fbp(li_sinogram)
            bhc_ct, bhc_sinogram = mar_bhc(
                ma_sinogram, metal_mask_bw,
                self.geo.forward, self.geo.fbp,
            )
            result.update({
                "LI_CT": li_ct.astype(np.float32),
                "BHC_CT": bhc_ct.astype(np.float32),
                "ma_sinogram": ma_sinogram,
                "LI_sinogram": li_sinogram.astype(np.float32),
                "BHC_sinogram": bhc_sinogram,
                "metal_trace": metal_trace,
            })

        return result

    @staticmethod
    def save_h5(
        result: SimulationResult,
        output_dir: str | Path,
        compression: str = "gzip",
        minimal: bool = False,
    ) -> None:
        """保存仿真结果为 HDF5 文件。

        minimal=False (完整模式):
            gt.h5:  image, poly_sinogram, poly_CT
            {idx}.h5: ma_CT, LI_CT, BHC_CT, sinograms, metal_trace

        minimal=True (精简模式):
            gt.h5:  image
            {idx}.h5: ma_CT
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        gt_path = out / "gt.h5"
        with h5py.File(str(gt_path), "w") as f:
            f.create_dataset("image", data=result.gt_ct, compression=compression)
            if not minimal:
                f.create_dataset("poly_sinogram", data=result.poly_sinogram, compression=compression)
                f.create_dataset("poly_CT", data=result.poly_ct, compression=compression)

        for mr in result.mask_results:
            mask_path = out / f"{mr['mask_idx']}.h5"
            with h5py.File(str(mask_path), "w") as f:
                f.create_dataset("ma_CT", data=mr["ma_CT"], compression=compression)
                if not minimal:
                    for key in ["LI_CT", "BHC_CT", "ma_sinogram", "LI_sinogram", "BHC_sinogram", "metal_trace"]:
                        if key in mr:
                            f.create_dataset(key, data=mr[key], compression=compression)

        logger.info(
            "Saved gt.h5 + %d mask h5 to %s", len(result.mask_results), out,
        )
