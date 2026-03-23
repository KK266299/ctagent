# ============================================================================
# 模块职责: CQ500 DICOM 数据读取器
#   扫描 CQ500 目录结构，按 patient → series → slice 组织
#   提供批量 slice 迭代器，输出 (HU ndarray, case_id) 对
#   适配理由: 原 MAR pipeline 依赖 DeepLesion PNG，现改为 CQ500 DICOM 输入
#   CQ500 数据格式: 512×512 int16 DICOM, RescaleSlope=1, RescaleIntercept=-1024
# 参考: dataset/dicom_scanner.py, dataset/dicom_reader.py (本项目已有的 DICOM 模块)
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SeriesRecord:
    """一个 DICOM series 的元信息。"""
    patient_id: str
    series_desc: str
    series_dir: Path
    dcm_paths: list[Path] = field(default_factory=list)

    @property
    def num_slices(self) -> int:
        return len(self.dcm_paths)

    @property
    def case_id(self) -> str:
        """唯一标识: PatientID/SeriesDesc (去空格)。"""
        desc_safe = self.series_desc.replace(" ", "_")
        return f"{self.patient_id}/{desc_safe}"


def scan_cq500(
    root_dir: str | Path,
    series_keywords: list[str] | None = None,
    min_slices: int = 20,
    max_patients: int | None = None,
) -> list[SeriesRecord]:
    """扫描 CQ500 目录，收集符合条件的 series。

    CQ500 目录结构:
      root/qctXX/CQ500CTYYY .../Unknown Study/CT SeriesDesc/CTXXXXXX.dcm

    Args:
        root_dir: CQ500 根目录
        series_keywords: series 筛选关键词 (如 ["THIN", "Plain"])，None=全部
        min_slices: 每个 series 最少切片数
        max_patients: 最多扫描多少个 patient (None=全部)
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"CQ500 root not found: {root}")

    series_list: list[SeriesRecord] = []
    patient_count = 0

    for batch_dir in sorted(root.iterdir()):
        if not batch_dir.is_dir() or batch_dir.name.startswith("."):
            continue

        for patient_dir in sorted(batch_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            if max_patients is not None and patient_count >= max_patients:
                break

            patient_id = patient_dir.name.split(" ")[0]
            study_dirs = list(patient_dir.rglob("Unknown Study"))
            if not study_dirs:
                study_dirs = [patient_dir]

            found_series = False
            for study_dir in study_dirs:
                for series_dir in sorted(study_dir.iterdir()):
                    if not series_dir.is_dir():
                        continue

                    series_desc = series_dir.name
                    if series_desc.startswith("CT "):
                        series_desc = series_desc[3:]

                    if series_keywords:
                        match = any(
                            kw.lower() in series_desc.lower()
                            for kw in series_keywords
                        )
                        if not match:
                            continue

                    dcm_paths = sorted(series_dir.glob("*.dcm"))
                    if len(dcm_paths) < min_slices:
                        continue

                    series_list.append(SeriesRecord(
                        patient_id=patient_id,
                        series_desc=series_desc,
                        series_dir=series_dir,
                        dcm_paths=dcm_paths,
                    ))
                    found_series = True

            if found_series:
                patient_count += 1

        if max_patients is not None and patient_count >= max_patients:
            break

    logger.info(
        "Scanned CQ500: %d series from %d patients (%s)",
        len(series_list), patient_count, root,
    )
    return series_list


def read_dicom_hu(
    dcm_path: str | Path,
    target_size: int = 416,
    hu_clip_min: float = -1000.0,
) -> np.ndarray:
    """读取单个 DICOM 文件，转为 HU 值并 resize。

    CQ500 格式: int16, RescaleSlope=1, RescaleIntercept=-1024
    HU = pixel_array * slope + intercept

    Returns:
        (target_size, target_size) float64 HU 图像
    """
    dcm = pydicom.dcmread(str(dcm_path))
    pixel = dcm.pixel_array.astype(np.float64)

    slope = np.float64(getattr(dcm, "RescaleSlope", 1))
    intercept = np.float64(getattr(dcm, "RescaleIntercept", 0))
    hu = pixel * slope + intercept

    if hu.shape[0] != target_size or hu.shape[1] != target_size:
        pil_img = Image.fromarray(hu.astype(np.float32))
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        hu = np.array(pil_img).astype(np.float64)

    hu[hu < hu_clip_min] = hu_clip_min
    return hu


def iter_slices(
    series_list: list[SeriesRecord],
    target_size: int = 416,
    stride: int = 1,
    hu_clip_min: float = -1000.0,
) -> "Generator[tuple[np.ndarray, str, Path], None, None]":
    """迭代所有 series 中的 slice，yield (hu_image, case_id, dcm_path)。

    Args:
        stride: 每隔多少 slice 取一个 (1=全部, 2=隔一取一, ...)
    """
    for sr in series_list:
        selected = sr.dcm_paths[::stride]
        for dcm_path in selected:
            try:
                hu = read_dicom_hu(dcm_path, target_size, hu_clip_min)
                slice_name = dcm_path.stem
                case_id = f"{sr.case_id}/{slice_name}"
                yield hu, case_id, dcm_path
            except Exception as e:
                logger.warning("Failed to read %s: %s", dcm_path, e)
