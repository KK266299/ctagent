# ============================================================================
# 模块职责: MAR PyTorch Dataset — 训练/验证数据加载
#   从 HDF5 文件加载配对的 (GT, 退化图像)，支持归一化和数据增强
# 参考: ADN — data/adn_dataset.py
# ============================================================================
from __future__ import annotations

import logging
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MU_MIN = 0.0
MU_MAX = 0.5


def normalize_mu(data: np.ndarray) -> np.ndarray:
    """线衰减系数 [0.0, 0.5] → [-1, 1]。"""
    data = np.clip(data, MU_MIN, MU_MAX)
    return (data - MU_MIN) / (MU_MAX - MU_MIN) * 2.0 - 1.0


def denormalize_mu(data: np.ndarray) -> np.ndarray:
    """[-1, 1] → 线衰减系数 [0.0, 0.5]。"""
    return (data + 1.0) / 2.0 * (MU_MAX - MU_MIN) + MU_MIN


class MARDataset(Dataset):
    """MAR 配对数据集。

    每个 sample 返回:
        gt: (1, H, W)     GT 图像 (归一化到 [-1, 1])
        ma: (1, H, W)     含伪影图像
        li: (1, H, W)     LI 校正图像
        metadata: dict     元信息

    Usage:
        dataset = MARDataset(
            data_dir="/path/to/output",
            split="train",
            augment=True,
        )
        gt, ma, li, meta = dataset[0]
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        input_key: str = "ma_CT",
        augment: bool = False,
        max_masks_per_slice: int | None = None,
    ) -> None:
        """
        Args:
            data_dir: 根目录（包含 patient/slice/ 子目录）
            split: "train" / "val" / "test" — 用于过滤
            input_key: 退化图像的 key ("ma_CT", "LI_CT", "BHC_CT")
            augment: 是否启用数据增强
            max_masks_per_slice: 每个 slice 最多使用多少个掩模
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.input_key = input_key
        self.augment = augment
        self.max_masks = max_masks_per_slice

        self.samples: list[tuple[Path, Path]] = []
        self._scan_data()
        logger.info(
            "MARDataset [%s]: %d samples from %s (input=%s)",
            split, len(self.samples), data_dir, input_key,
        )

    def _scan_data(self) -> None:
        """扫描目录结构，收集所有 (gt.h5, mask.h5) 对。"""
        for gt_path in sorted(self.data_dir.rglob("gt.h5")):
            parent = gt_path.parent
            mask_h5s = sorted(parent.glob("[0-9]*.h5"))

            if self.max_masks is not None:
                mask_h5s = mask_h5s[: self.max_masks]

            for mask_path in mask_h5s:
                if mask_path.name == "gt.h5":
                    continue
                self.samples.append((gt_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        gt_path, mask_path = self.samples[idx]

        with h5py.File(str(gt_path), "r") as f:
            gt = f["image"][:].astype(np.float32)

        with h5py.File(str(mask_path), "r") as f:
            ma = f[self.input_key][:].astype(np.float32)
            li = f["LI_CT"][:].astype(np.float32) if "LI_CT" in f else ma.copy()

        gt = normalize_mu(gt)
        ma = normalize_mu(ma)
        li = normalize_mu(li)

        if self.augment and random.random() > 0.5:
            gt = np.flip(gt, axis=1).copy()
            ma = np.flip(ma, axis=1).copy()
            li = np.flip(li, axis=1).copy()

        gt_t = torch.from_numpy(gt).unsqueeze(0)
        ma_t = torch.from_numpy(ma).unsqueeze(0)
        li_t = torch.from_numpy(li).unsqueeze(0)

        metadata = {
            "gt_path": str(gt_path),
            "mask_path": str(mask_path),
            "mask_idx": mask_path.stem,
        }
        return gt_t, ma_t, li_t, metadata
