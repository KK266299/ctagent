# ============================================================================
# 模块职责: CT 图像读取器 — 支持 DICOM / NIfTI / PNG / NumPy 格式
# 参考: ProCT (https://github.com/Masaaki-75/proct)
#       MedQ-Bench (https://github.com/liujiyaoFDU/MedQ-Bench)
# ============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def read_ct(path: Union[str, Path]) -> np.ndarray:
    """根据文件扩展名自动选择读取方式，返回 HU 值 numpy 数组。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        return read_dicom(path)
    elif suffix in (".nii", ".gz"):
        return read_nifti(path)
    elif suffix == ".png":
        return read_png(path)
    elif suffix == ".npy":
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def read_dicom(path: Union[str, Path]) -> np.ndarray:
    """读取单个 DICOM 文件，返回 HU 值。"""
    import pydicom

    ds = pydicom.dcmread(str(path))
    image = ds.pixel_array.astype(np.float32)
    # 转换为 HU
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    image = image * slope + intercept
    return image


def read_nifti(path: Union[str, Path]) -> np.ndarray:
    """读取 NIfTI 文件。"""
    import nibabel as nib

    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)


def read_png(path: Union[str, Path]) -> np.ndarray:
    """读取 PNG 图像，返回 [0, 1] 浮点数组。"""
    from PIL import Image

    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0
