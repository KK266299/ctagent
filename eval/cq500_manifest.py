# ============================================================================
# 模块职责: CQ500 评测 manifest 构建
#   扫描 cq500_processed/ 目录 → 构建 case-level 三路对齐清单
#   clean: gt.h5["image"]   degraded: {idx}.h5["ma_CT"]
#   支持 BHX 标注感知的病灶层优先选择策略
# ============================================================================
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SliceEntry:
    """单个 slice 的 clean + degraded 路径。"""
    slice_dir: Path
    gt_h5: Path
    degraded_h5: Path
    has_lesion: bool = False
    lesion_labels: list[str] = field(default_factory=list)
    lesion_area: float = 0.0


@dataclass
class EvalCase:
    """一个患者级别的评测条目。"""
    case_id: str
    patient_id: str
    series: str
    slices: list[SliceEntry] = field(default_factory=list)
    restored_dir: Path | None = None
    n_lesion_slices: int = 0
    selection_method: str = "middle_uniform"


class SOPIndex:
    """SOPInstanceUID ↔ processed slice path 的双向映射。

    由 scripts/build_sop_index.py 预先生成的 JSON 加载。
    """

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._sop_to_slice: dict[str, dict[str, str]] = {}
        self._slice_to_sop: dict[str, str] = {}
        self._series_uid_to_folder: dict[str, dict[str, str]] = {}
        self._study_uid_to_patient: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        with open(self.index_path) as f:
            data = json.load(f)
        self._sop_to_slice = data.get("sop_to_slice", {})
        self._series_uid_to_folder = data.get("series_uid_to_folder", {})
        self._study_uid_to_patient = data.get("study_uid_to_patient", {})

        for sop_uid, info in self._sop_to_slice.items():
            self._slice_to_sop[info["processed_path"]] = sop_uid

        logger.info(
            "SOPIndex loaded: %d SOPs, %d series, %d studies",
            len(self._sop_to_slice),
            len(self._series_uid_to_folder),
            len(self._study_uid_to_patient),
        )

    def sop_to_processed_path(self, sop_uid: str) -> str | None:
        info = self._sop_to_slice.get(sop_uid)
        return info["processed_path"] if info else None

    def processed_path_to_sop(self, processed_path: str) -> str | None:
        return self._slice_to_sop.get(processed_path)

    def get_series_uid(self, patient_id: str, series_folder: str) -> str | None:
        """通过 patient_id 和 series_folder 反查 SeriesInstanceUID。"""
        for uid, info in self._series_uid_to_folder.items():
            if info["patient_id"] == patient_id and info["series_folder"] == series_folder:
                return uid
        return None

    def get_study_uid(self, patient_id: str) -> str | None:
        for uid, pid in self._study_uid_to_patient.items():
            if pid == patient_id:
                return uid
        return None


def build_eval_manifest(
    processed_dir: str | Path,
    label_case_ids: set[str] | None = None,
    max_slices_per_case: int = 5,
    mask_idx: int = 0,
    restored_dir: str | Path | None = None,
    sop_index: SOPIndex | None = None,
    bhx_annotations: Any | None = None,
) -> list[EvalCase]:
    """扫描 cq500_processed 构建评测清单。

    当提供 sop_index + bhx_annotations 时, 优先选择病灶所在层面;
    否则退化为从中间区域均匀采样。

    Args:
        processed_dir: cq500_processed 根目录
        label_case_ids: 有标签的 case_id 集合 (用于过滤)
        max_slices_per_case: 每个 case 最多选多少张 slice
        mask_idx: 使用哪个金属掩模的退化结果 (默认 0)
        restored_dir: 修复后图像目录 (可选)
        sop_index: SOPInstanceUID 映射索引
        bhx_annotations: BHXAnnotations 实例
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
                if not (gt_path.exists() and deg_path.exists()):
                    continue

                entry = SliceEntry(
                    slice_dir=slice_dir,
                    gt_h5=gt_path,
                    degraded_h5=deg_path,
                )

                if sop_index and bhx_annotations:
                    proc_path = f"{patient_id}/{series_name}/{slice_dir.name}"
                    sop_uid = sop_index.processed_path_to_sop(proc_path)
                    if sop_uid and bhx_annotations.has_annotation(sop_uid):
                        ann = bhx_annotations.by_sop[sop_uid]
                        entry.has_lesion = True
                        entry.lesion_labels = list(ann.label_names)
                        entry.lesion_area = ann.total_area

                slice_entries.append(entry)

            if not slice_entries:
                continue

            selected, method = _select_slices_with_lesion_priority(
                slice_entries, max_slices_per_case,
            )

            rst_dir = None
            if restored_dir:
                candidate = Path(restored_dir) / patient_id / series_name
                if candidate.exists():
                    rst_dir = candidate

            n_lesion = sum(1 for s in selected if s.has_lesion)

            cases.append(EvalCase(
                case_id=patient_id,
                patient_id=patient_id,
                series=series_name,
                slices=selected,
                restored_dir=rst_dir,
                n_lesion_slices=n_lesion,
                selection_method=method,
            ))

    n_with_lesion = sum(1 for c in cases if c.n_lesion_slices > 0)
    logger.info(
        "Built eval manifest: %d cases (%d with lesion slices) from %s",
        len(cases), n_with_lesion, root,
    )
    return cases


def _select_slices_with_lesion_priority(
    slices: list[SliceEntry],
    max_n: int,
) -> tuple[list[SliceEntry], str]:
    """病灶优先 + 上下文补充的切片选择策略。

    Returns:
        (selected_slices, selection_method_name)
    """
    n = len(slices)
    if n <= max_n:
        return slices, "all"

    lesion_slices = [s for s in slices if s.has_lesion]

    if not lesion_slices:
        return _select_middle_uniform(slices, max_n), "middle_uniform"

    lesion_slices.sort(key=lambda s: s.lesion_area, reverse=True)

    if len(lesion_slices) >= max_n:
        return lesion_slices[:max_n], "lesion_top_area"

    selected = list(lesion_slices)
    remaining = max_n - len(selected)

    selected_dirs = {s.slice_dir for s in selected}

    lesion_indices = []
    for i, s in enumerate(slices):
        if s.has_lesion:
            lesion_indices.append(i)

    context_candidates: list[tuple[int, SliceEntry]] = []
    for li in lesion_indices:
        for offset in [-2, -1, 1, 2]:
            ci = li + offset
            if 0 <= ci < n and slices[ci].slice_dir not in selected_dirs:
                context_candidates.append((abs(offset), slices[ci]))

    context_candidates.sort(key=lambda x: x[0])
    for _, entry in context_candidates:
        if entry.slice_dir in selected_dirs:
            continue
        selected.append(entry)
        selected_dirs.add(entry.slice_dir)
        if len(selected) >= max_n:
            break

    if len(selected) < max_n:
        for s in slices:
            if s.slice_dir not in selected_dirs:
                selected.append(s)
                selected_dirs.add(s.slice_dir)
                if len(selected) >= max_n:
                    break

    idx_map = {s.slice_dir: i for i, s in enumerate(slices)}
    selected.sort(key=lambda s: idx_map.get(s.slice_dir, 0))

    return selected, "lesion_priority"


def _select_middle_uniform(
    slices: list[SliceEntry],
    max_n: int,
) -> list[SliceEntry]:
    """从中间区域均匀采样代表性 slice (无标注信息时的回退策略)。"""
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
