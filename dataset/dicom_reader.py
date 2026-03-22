# ============================================================================
# 模块职责: DICOM Reader — 读取单个 DICOM 文件, 转换为标准化 2D 图像
#   DICOM pixel → HU (Hounsfield Unit) → 窗宽窗位归一化 → [0,1] float32
#   支持自定义窗宽窗位或使用 DICOM 头中的默认值
# 参考: ADN — 数据准备中的 HU 转换和归一化逻辑
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SliceData:
    """读取后的单个 slice 数据。"""
    pixel_hu: np.ndarray          # HU 值 (float32)
    pixel_normalized: np.ndarray  # 窗位归一化后 [0,1] (float32)
    instance_number: int = 0
    slice_location: float = 0.0
    image_position: tuple[float, ...] = ()
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


def read_dicom_slice(
    dcm_path: str,
    window_center: float | None = None,
    window_width: float | None = None,
    hu_clip: tuple[float, float] = (-1024.0, 3071.0),
) -> SliceData:
    """读取单个 DICOM 文件并转换为标准化 slice。

    Args:
        dcm_path: DICOM 文件路径
        window_center: 窗位 (None=使用 DICOM 头中的值)
        window_width: 窗宽 (None=使用 DICOM 头中的值)
        hu_clip: HU 值裁剪范围
    """
    import pydicom

    ds = pydicom.dcmread(dcm_path)
    pixel = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = pixel * slope + intercept
    hu = np.clip(hu, hu_clip[0], hu_clip[1])

    wc = window_center
    ww = window_width
    if wc is None:
        wc_raw = getattr(ds, "WindowCenter", None)
        wc = _to_float(wc_raw) if wc_raw is not None else 40.0
    if ww is None:
        ww_raw = getattr(ds, "WindowWidth", None)
        ww = _to_float(ww_raw) if ww_raw is not None else 400.0

    normalized = apply_window(hu, wc, ww)

    pos = getattr(ds, "ImagePositionPatient", None)
    image_position = tuple(float(p) for p in pos) if pos else ()

    return SliceData(
        pixel_hu=hu,
        pixel_normalized=normalized,
        instance_number=int(getattr(ds, "InstanceNumber", 0)),
        slice_location=float(getattr(ds, "SliceLocation", 0.0)),
        image_position=image_position,
        metadata={
            "patient_id": str(getattr(ds, "PatientID", "")),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            "series_description": str(getattr(ds, "SeriesDescription", "")),
            "rows": int(ds.Rows),
            "columns": int(ds.Columns),
            "rescale_slope": slope,
            "rescale_intercept": intercept,
            "window_center": wc,
            "window_width": ww,
            "hu_min": float(hu.min()),
            "hu_max": float(hu.max()),
        },
    )


def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """窗宽窗位归一化: HU → [0, 1]。"""
    low = center - width / 2.0
    high = center + width / 2.0
    normalized = (hu - low) / max(high - low, 1e-6)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _to_float(val: Any) -> float:
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return float(val[0])
    return float(val)
