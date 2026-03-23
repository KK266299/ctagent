# ============================================================================
# 模块职责: CQ500 评测 manifest 构建
#   扫描 cq500_processed/ 目录 → 构建 case-level 三路对齐清单
#   clean: gt.h5["image"]   degraded: {idx}.h5["ma_CT"]
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SliceEntry:
    """单个 slice 的 clean + degraded 路径。"""
    slice_dir: Path
    gt_h5: Path
    degraded_h5: Path


@dataclass
class EvalCase:
    """一个患者级别的评测条目。"""
    case_id: str
    patient_id: str
    series: str
    slices: list[SliceEntry] = field(default_factory=list)
    restored_dir: Path | None = None


def build_eval_manifest(
    processed_dir: str | Path,
    label_case_ids: set[str] | None = None,
    max_slices_per_case: int = 5,
    mask_idx: int = 0,
    restored_dir: str | Path | None = None,
) -> list[EvalCase]:
    """扫描 cq500_processed 构建评测清单。

    Args:
        processed_dir: cq500_processed 根目录
        label_case_ids: 有标签的 case_id 集合 (用于过滤)
        max_slices_per_case: 每个 case 最多选多少张 slice
        mask_idx: 使用哪个金属掩模的退化结果 (默认 0)
        restored_dir: 修复后图像目录 (可选, 后续阶段使用)
    """
    root = Path(processed_dir)
    if not root.exists():
        logger.error("Processed directory not found: %s", root)
        return []

    cases: list[EvalCase] = []

    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("."):
            continue
        patient_id = patient_dir.name

        if label_case_ids and patient_id not in label_case_ids:
            continue

        for series_dir in sorted(patient_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            series_name = series_dir.name

            slice_entries: list[SliceEntry] = []
            for slice_dir in sorted(series_dir.iterdir()):
                if not slice_dir.is_dir():
                    continue
                gt_path = slice_dir / "gt.h5"
                deg_path = slice_dir / f"{mask_idx}.h5"
                if gt_path.exists() and deg_path.exists():
                    slice_entries.append(SliceEntry(
                        slice_dir=slice_dir,
                        gt_h5=gt_path,
                        degraded_h5=deg_path,
                    ))

            if not slice_entries:
                continue

            selected = _select_representative_slices(slice_entries, max_slices_per_case)

            rst_dir = None
            if restored_dir:
                candidate = Path(restored_dir) / patient_id / series_name
                if candidate.exists():
                    rst_dir = candidate

            cases.append(EvalCase(
                case_id=patient_id,
                patient_id=patient_id,
                series=series_name,
                slices=selected,
                restored_dir=rst_dir,
            ))

    logger.info("Built eval manifest: %d cases from %s", len(cases), root)
    return cases


def _select_representative_slices(
    slices: list[SliceEntry],
    max_n: int,
) -> list[SliceEntry]:
    """从中间区域均匀采样代表性 slice。"""
    n = len(slices)
    if n <= max_n:
        return slices

    start = n // 4
    end = n * 3 // 4
    mid_slices = slices[start:end]

    if len(mid_slices) <= max_n:
        return mid_slices

    step = len(mid_slices) / max_n
    indices = [int(i * step) for i in range(max_n)]
    return [mid_slices[i] for i in indices]
