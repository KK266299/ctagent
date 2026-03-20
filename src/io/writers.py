# ============================================================================
# 模块职责: CT 图像写入器 — 保存处理结果到不同格式
# 参考: ProCT (https://github.com/Masaaki-75/proct)
# ============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def write_ct(image: np.ndarray, path: Union[str, Path]) -> None:
    """根据扩展名自动选择写入方式。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".png":
        write_png(image, path)
    elif suffix in (".nii", ".gz"):
        write_nifti(image, path)
    elif suffix == ".npy":
        np.save(path, image)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def write_png(image: np.ndarray, path: Union[str, Path]) -> None:
    """将 [0, 1] 浮点数组保存为 PNG。"""
    from PIL import Image

    image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(str(path))


def write_nifti(image: np.ndarray, path: Union[str, Path]) -> None:
    """保存为 NIfTI 格式。"""
    import nibabel as nib

    nii = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(nii, str(path))
