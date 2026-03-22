# ============================================================================
# 模块职责: CT Slice Exporter — 批量将 DICOM series 导出为 clean 2D npy slices
#   遍历 SeriesInfo 列表 → 逐 slice 读取+转换 → 存储 npy → 生成 manifest
#   参考 ADN prepare_spineweb.py: volume→逐slice→resize→npy+thumbnail
#   我们简化为: DICOM→HU→窗位归一化→npy, 不做 resize (保持原始 512x512)
# 参考: ADN — prepare_spineweb.py 的逐 slice 导出 + npy 存储模式
# ============================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dataset.dicom_reader import read_dicom_slice
from dataset.dicom_scanner import SeriesInfo
from dataset.manifest import write_manifest

logger = logging.getLogger(__name__)


def export_clean_slices(
    series_list: list[SeriesInfo],
    output_dir: str | Path,
    manifest_path: str | Path,
    window_center: float | None = None,
    window_width: float | None = None,
    max_series: int | None = None,
) -> int:
    """批量导出 clean 2D slices。

    Args:
        series_list: scan 结果
        output_dir: 输出根目录 (如 output/clean)
        manifest_path: manifest 输出路径
        window_center/window_width: 全局窗宽窗位 (None=使用 DICOM 头)
        max_series: 限制处理的 series 数量

    Returns:
        导出的 slice 总数
    """
    out_root = Path(output_dir)
    records: list[dict[str, Any]] = []
    total = 0

    series_to_process = series_list[:max_series] if max_series else series_list

    for si, series in enumerate(series_to_process):
        uid_short = series.series_uid[-12:] if len(series.series_uid) > 12 else series.series_uid
        safe_patient = _safe_name(series.patient_id)
        series_dir = out_root / safe_patient / uid_short
        series_dir.mkdir(parents=True, exist_ok=True)

        sorted_paths = _sort_dcm_paths(series.dcm_paths)

        for idx, dcm_path in enumerate(sorted_paths):
            try:
                sd = read_dicom_slice(
                    dcm_path,
                    window_center=window_center,
                    window_width=window_width,
                )
            except Exception as e:
                logger.warning("Failed to read %s: %s", dcm_path, e)
                continue

            npy_name = f"slice_{idx:04d}.npy"
            npy_path = series_dir / npy_name
            np.save(str(npy_path), sd.pixel_normalized)

            records.append({
                "slice_id": f"{safe_patient}__{uid_short}__slice_{idx:04d}",
                "patient_id": series.patient_id,
                "series_uid": series.series_uid,
                "series_description": series.series_description,
                "instance_number": sd.instance_number,
                "slice_location": round(sd.slice_location, 2),
                "npy_path": str(npy_path),
                "rows": sd.metadata.get("rows", 0),
                "columns": sd.metadata.get("columns", 0),
                "window_center": sd.metadata.get("window_center"),
                "window_width": sd.metadata.get("window_width"),
                "hu_min": round(sd.metadata.get("hu_min", 0), 1),
                "hu_max": round(sd.metadata.get("hu_max", 0), 1),
            })
            total += 1

        if (si + 1) % 10 == 0 or si == len(series_to_process) - 1:
            logger.info("Exported %d/%d series, %d slices total",
                        si + 1, len(series_to_process), total)

    write_manifest(records, manifest_path)
    logger.info("Clean dataset: %d slices from %d series → %s", total, len(series_to_process), out_root)
    return total


def _sort_dcm_paths(paths: list[str]) -> list[str]:
    """按文件名中的数字排序 DICOM 路径。"""
    import re

    def _num_key(p: str) -> int:
        m = re.search(r"(\d+)\.dcm$", p, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    return sorted(paths, key=_num_key)


def _safe_name(name: str) -> str:
    """将 patient ID 转为安全的文件名。"""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
