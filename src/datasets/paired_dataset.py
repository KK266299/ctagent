# ============================================================================
# 模块职责: 配对 CT 数据集 — 退化/清晰图像对，用于评估
# 参考: IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch) — paired dataset
#       ProCT (https://github.com/Masaaki-75/proct)
# ============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

from torch.utils.data import Dataset

from src.io import read_ct


class PairedCTDataset(Dataset):
    """配对 CT 数据集：退化图像 + 参考图像。"""

    def __init__(
        self,
        degraded_dir: Union[str, Path],
        clean_dir: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        self.degraded_dir = Path(degraded_dir)
        self.clean_dir = Path(clean_dir)
        self.transform = transform

        self.degraded_files = sorted(self.degraded_dir.rglob("*.*"))
        self.clean_files = sorted(self.clean_dir.rglob("*.*"))
        assert len(self.degraded_files) == len(self.clean_files), (
            f"Mismatch: {len(self.degraded_files)} degraded vs {len(self.clean_files)} clean"
        )

    def __len__(self) -> int:
        return len(self.degraded_files)

    def __getitem__(self, idx: int) -> dict:
        degraded = read_ct(self.degraded_files[idx])
        clean = read_ct(self.clean_files[idx])
        sample = {
            "degraded": degraded,
            "clean": clean,
            "degraded_path": str(self.degraded_files[idx]),
            "clean_path": str(self.clean_files[idx]),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
