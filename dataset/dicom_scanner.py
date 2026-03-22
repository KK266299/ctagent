# ============================================================================
# 模块职责: DICOM Scanner — 递归扫描数据目录, 发现所有 series 并组织为结构化列表
#   输入: 根数据目录 (如 /home/liuxinyao/data)
#   输出: List[SeriesInfo] — 按 patient/study/series 层级组织
#   支持 CQ500 风格的 4 级目录: batch → patient → study → series
#   也支持扁平或任意嵌套的 DICOM 目录 (通过 SeriesInstanceUID 聚合)
# 参考: ADN — prepare_spineweb.py 中的目录扫描逻辑
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SeriesInfo:
    """一个 DICOM series 的汇总信息。"""
    series_uid: str
    patient_id: str = ""
    study_uid: str = ""
    series_description: str = ""
    series_dir: str = ""
    num_slices: int = 0
    dcm_paths: list[str] = field(default_factory=list)
    sample_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_uid": self.series_uid,
            "patient_id": self.patient_id,
            "study_uid": self.study_uid,
            "series_description": self.series_description,
            "series_dir": self.series_dir,
            "num_slices": self.num_slices,
        }


def scan_dicom_directory(
    root_dir: str | Path,
    max_patients: int | None = None,
    series_filter: dict[str, Any] | None = None,
) -> list[SeriesInfo]:
    """递归扫描目录, 返回按 SeriesInstanceUID 聚合的 series 列表。

    Args:
        root_dir: 数据根目录
        max_patients: 限制扫描的 patient 数量 (用于快速测试)
        series_filter: 过滤条件, 如 {"min_slices": 10, "description_contains": "THIN"}
    """
    import pydicom

    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    dcm_files = sorted(root.rglob("*.dcm"))
    logger.info("Found %d .dcm files under %s", len(dcm_files), root)

    if not dcm_files:
        return []

    series_map: dict[str, SeriesInfo] = {}
    seen_patients: set[str] = set()

    for dcm_path in dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        except Exception:
            continue

        series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        patient_id = str(getattr(ds, "PatientID", "unknown"))

        if max_patients and patient_id not in seen_patients:
            if len(seen_patients) >= max_patients:
                continue
            seen_patients.add(patient_id)
        elif max_patients and patient_id not in seen_patients:
            continue
        else:
            seen_patients.add(patient_id)

        if series_uid not in series_map:
            series_map[series_uid] = SeriesInfo(
                series_uid=series_uid,
                patient_id=patient_id,
                study_uid=str(getattr(ds, "StudyInstanceUID", "")),
                series_description=str(getattr(ds, "SeriesDescription", "")),
                series_dir=str(dcm_path.parent),
                sample_metadata={
                    "rows": getattr(ds, "Rows", 0),
                    "columns": getattr(ds, "Columns", 0),
                    "rescale_slope": float(getattr(ds, "RescaleSlope", 1)),
                    "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0)),
                    "window_center": _parse_ds_value(getattr(ds, "WindowCenter", None)),
                    "window_width": _parse_ds_value(getattr(ds, "WindowWidth", None)),
                },
            )
        series_map[series_uid].dcm_paths.append(str(dcm_path))

    for info in series_map.values():
        info.num_slices = len(info.dcm_paths)

    result = list(series_map.values())

    sf = series_filter or {}
    min_slices = sf.get("min_slices", 0)
    desc_contains = sf.get("description_contains", "")
    if min_slices > 0 or desc_contains:
        before = len(result)
        result = [
            s for s in result
            if s.num_slices >= min_slices
            and (not desc_contains or desc_contains.upper() in s.series_description.upper())
        ]
        logger.info("Filtered %d → %d series (min_slices=%d, desc='%s')",
                     before, len(result), min_slices, desc_contains)

    logger.info("Discovered %d series from %d patients",
                len(result), len({s.patient_id for s in result}))
    return result


def _parse_ds_value(val: Any) -> float | None:
    """解析 DICOM DS (Decimal String) 值, 可能是多值。"""
    if val is None:
        return None
    if hasattr(val, "real"):
        return float(val)
    try:
        if hasattr(val, "__iter__") and not isinstance(val, str):
            return float(val[0])
        return float(val)
    except (TypeError, ValueError, IndexError):
        return None
