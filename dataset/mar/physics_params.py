# ============================================================================
# 模块职责: 物理参数加载 — 材料衰减系数、X射线能谱、BHC多项式系数
#   从 .mat 文件加载水/骨/金属的质量衰减系数表和 GE 120kVp 能谱
#   预计算水基 BHC 三阶多项式拟合系数
# 参考: ADN — +helper/get_mar_params.m
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

logger = logging.getLogger(__name__)

MATERIAL_NAMES = ["Ti", "Fe", "Cu", "Au"]
MATERIAL_DENSITIES = [4.5, 7.8, 8.9, 2.0]  # g/cm³
ATTEN_MODE_COL = 6  # column index for total attenuation with coherent scattering


@dataclass
class PhysicsConfig:
    """物理仿真参数配置。"""
    mat_dir: str = "data/mar_physics"  # .mat 文件目录

    kVp: int = 120
    kev: int = 70
    energy_min: int = 20
    energy_max: int = 120
    photon_num: float = 2e7
    scatter_photon: int = 20
    mu_water_70kev: float = 0.192

    thresh_water_hu: float = 100.0
    thresh_bone_hu: float = 1500.0

    material_id: int = 0  # 0=Ti, 1=Fe, 2=Cu, 3=Au

    # BHC 水厚度网格
    bhc_thickness_max: float = 50.0
    bhc_thickness_step: float = 0.05

    # .mat 文件名
    mat_water: str = "MiuofH2O.mat"
    mat_bone: str = "MiuofBONE_Cortical_ICRU44.mat"
    mat_metals: list[str] = field(default_factory=lambda: [
        "MiuofTi.mat", "MiuofFe.mat", "MiuofCu.mat", "MiuofAu.mat",
    ])
    mat_spectrum: str = "GE14Spectrum120KVP.mat"
    mat_masks: str = "metal_masks.mat"

    def __post_init__(self) -> None:
        """YAML safe_load 可能把科学计数法解析为字符串，此处强制转换。"""
        self.kVp = int(self.kVp)
        self.kev = int(self.kev)
        self.energy_min = int(self.energy_min)
        self.energy_max = int(self.energy_max)
        self.photon_num = float(self.photon_num)
        self.scatter_photon = int(self.scatter_photon)
        self.mu_water_70kev = float(self.mu_water_70kev)
        self.thresh_water_hu = float(self.thresh_water_hu)
        self.thresh_bone_hu = float(self.thresh_bone_hu)
        self.material_id = int(self.material_id)


class PhysicsParams:
    """物理参数管理器。

    加载材料衰减系数、X射线能谱，并预计算 BHC 多项式系数。

    Usage:
        params = PhysicsParams(PhysicsConfig(mat_dir="/path/to/mats"))
        params.load()
    """

    def __init__(self, config: PhysicsConfig | None = None) -> None:
        self.config = config or PhysicsConfig()
        self._loaded = False

        self.mu_water: np.ndarray | None = None     # (120, num_modes)
        self.mu_bone: np.ndarray | None = None       # (120, num_modes)
        self.mu_metals: np.ndarray | None = None     # (120, num_modes, 4)
        self.spectrum: np.ndarray | None = None      # (120,)
        self.energies: np.ndarray | None = None      # [20..120]
        self.metal_atten: float = 0.0
        self.para_bhc: np.ndarray | None = None      # (3, 1) BHC polynomial
        self.thresh_water: float = 0.0
        self.thresh_bone: float = 0.0

    def load(self) -> None:
        """加载所有 .mat 文件并预计算 BHC 系数。"""
        cfg = self.config
        mat_dir = Path(cfg.mat_dir)

        self.energies = np.arange(cfg.energy_min, cfg.energy_max + 1)

        kVp = cfg.kVp
        self.mu_water = self._load_mat(mat_dir / cfg.mat_water, max_rows=kVp)
        self.mu_bone = self._load_mat(mat_dir / cfg.mat_bone, max_rows=kVp)

        metals = []
        for fname in cfg.mat_metals:
            metals.append(self._load_mat(mat_dir / fname, max_rows=kVp))
        self.mu_metals = np.stack(metals, axis=-1)  # (kVp, modes, 4)

        self.spectrum = self._load_spectrum(mat_dir / cfg.mat_spectrum)

        mu_w = cfg.mu_water_70kev
        self.thresh_water = cfg.thresh_water_hu / 1000.0 * mu_w + mu_w
        self.thresh_bone = cfg.thresh_bone_hu / 1000.0 * mu_w + mu_w

        mid = cfg.material_id
        self.metal_atten = (
            MATERIAL_DENSITIES[mid]
            * self.mu_metals[cfg.kev - 1, ATTEN_MODE_COL, mid]
        )
        logger.info(
            "Metal attenuation (material=%s): %.4f cm^-1",
            MATERIAL_NAMES[mid], self.metal_atten,
        )

        self.para_bhc = self._compute_bhc_coefficients()
        self._loaded = True
        logger.info("PhysicsParams loaded from %s", mat_dir)

    @staticmethod
    def _load_mat(path: Path, max_rows: int | None = None) -> np.ndarray:
        """加载 .mat 文件，优先按文件名stem匹配key，否则取第一个2D数组。

        Args:
            max_rows: 截断到前 N 行 (用于统一不同材料表的行数到 kVp)
        """
        data = sio.loadmat(str(path))
        stem = path.stem  # e.g. "MiuofH2O"

        arr = None
        if stem in data and isinstance(data[stem], np.ndarray):
            arr = data[stem].astype(np.float64)
        else:
            for key, val in data.items():
                if not key.startswith("__") and isinstance(val, np.ndarray) and val.ndim == 2:
                    arr = val.astype(np.float64)
                    break

        if arr is None:
            raise ValueError(f"No valid array found in {path}")

        if max_rows is not None and arr.shape[0] > max_rows:
            arr = arr[:max_rows, :]

        return arr

    def _load_spectrum(self, path: Path) -> np.ndarray:
        """加载 X 射线能谱，取前 kVp 行的第二列。"""
        data = sio.loadmat(str(path))
        stem = path.stem

        arr = None
        if stem in data and isinstance(data[stem], np.ndarray):
            arr = data[stem].astype(np.float64)
        else:
            for key, val in data.items():
                if not key.startswith("__") and isinstance(val, np.ndarray) and val.ndim == 2:
                    arr = val.astype(np.float64)
                    break

        if arr is None:
            raise ValueError(f"No valid spectrum in {path}")

        kVp = self.config.kVp
        if arr.shape[0] > kVp:
            arr = arr[:kVp, :]

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1]
        return arr.ravel()

    def _compute_bhc_coefficients(self) -> np.ndarray:
        """预计算水基 BHC 三阶多项式拟合系数。"""
        cfg = self.config
        thickness = np.arange(0, cfg.bhc_thickness_max + cfg.bhc_thickness_step,
                              cfg.bhc_thickness_step).reshape(-1, 1)

        mu_water_kev = self.mu_water[cfg.kev - 1, ATTEN_MODE_COL]
        p_water_kev = mu_water_kev * thickness  # mono-energy projection

        p_water_kvp = self._pkev2kvp_1d(
            p_water_kev,
            material_mu=self.mu_water,
        )

        A = np.concatenate([p_water_kvp, p_water_kvp**2, p_water_kvp**3], axis=1)
        para_bhc = np.linalg.pinv(A) @ p_water_kev  # (3, 1)

        logger.info("BHC polynomial coefficients: %s", para_bhc.ravel())
        return para_bhc

    def _pkev2kvp_1d(
        self,
        proj_kev: np.ndarray,
        material_mu: np.ndarray,
    ) -> np.ndarray:
        """1D 版 pkev2kvp，用于 BHC 系数预计算（单材料）。

        Args:
            proj_kev: (N, 1) 单能投影
            material_mu: (120, modes) 该材料的衰减系数表
        """
        cfg = self.config
        proj_kvp_sum = np.zeros_like(proj_kev)
        spectrum_sum = 0.0

        for ien in self.energies:
            ratio = (
                material_mu[ien - 1, ATTEN_MODE_COL]
                / material_mu[cfg.kev - 1, ATTEN_MODE_COL]
            )
            proj_ien = proj_kev * ratio
            proj_kvp_sum += self.spectrum[ien - 1] * np.exp(-proj_ien)
            spectrum_sum += self.spectrum[ien - 1]

        proj_kvp = -np.log(proj_kvp_sum / spectrum_sum + 1e-30)
        return proj_kvp

    def get_mu_all_materials(self) -> dict[str, np.ndarray]:
        """返回所有需要的材料衰减系数。"""
        return {
            "water": self.mu_water,
            "bone": self.mu_bone,
            "metals": self.mu_metals,
        }

    def load_metal_masks(self, path: str | None = None) -> list[np.ndarray]:
        """加载金属掩模 .mat 文件，返回掩模列表。

        优先使用 'CT_samples_bwMetal' key (ADN/MAR_SynCode 格式)，
        否则回退到第一个 3D uint8 数组。
        """
        if path is None:
            path = str(Path(self.config.mat_dir) / self.config.mat_masks)

        data = sio.loadmat(path)

        arr = None
        if "CT_samples_bwMetal" in data:
            arr = np.array(data["CT_samples_bwMetal"], dtype=np.float32)
        else:
            for key, val in data.items():
                if key.startswith("__"):
                    continue
                v = np.array(val)
                if v.ndim == 3 and v.shape[0] > 10 and v.shape[1] > 10:
                    arr = v.astype(np.float32)
                    break

        masks = []
        if arr is not None:
            if arr.ndim == 3:
                for i in range(arr.shape[2]):
                    masks.append(arr[:, :, i])
            elif arr.ndim == 2:
                masks.append(arr)

        logger.info("Loaded %d metal masks from %s", len(masks), path)
        return masks
