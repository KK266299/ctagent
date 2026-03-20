# ============================================================================
# 模块职责: 通用 CT 数据集 — 单图像加载，用于推理 / 退化检测
# 参考: LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) — dataset pattern
#       MedQ-Bench (https://github.com/liujiyaoFDU/MedQ-Bench)
# ============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from src.io import read_ct


class CTDataset(Dataset):
    """通用 CT 图像数据集。"""

    def __init__(
        self,
        data_dir: Union[str, Path],
        file_list: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        extensions: tuple[str, ...] = (".dcm", ".nii", ".nii.gz", ".png", ".npy"),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform

        if file_list is not None:
            self.files = [self.data_dir / f for f in file_list]
        else:
            self.files = sorted(
                f for f in self.data_dir.rglob("*") if f.suffix.lower() in extensions
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        image = read_ct(path)
        sample = {"image": image, "path": str(path)}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
