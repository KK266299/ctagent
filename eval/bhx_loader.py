# ============================================================================
# 模块职责: 加载 BHX (Brain Hemorrhage Extended) bounding-box 标注
#   解析 PhysioNet BHX 1.1 CSV → 按 SOPInstanceUID / StudyInstanceUID 索引
#   用于病灶层面感知的切片选择 (lesion-aware slice selection)
# 数据来源: https://physionet.org/content/bhx-brain-bounding-box/1.1/
# ============================================================================
from __future__ import annotations

import ast
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HEMORRHAGE_TYPES = [
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
    "Epidural",
    "Chronic",
]


@dataclass
class BBox:
    """单个 bounding-box 标注。"""
    x: float
    y: float
    width: float
    height: float
    label_name: str
    label_type: str

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class SliceAnnotation:
    """单张切片 (SOPInstanceUID) 上的全部标注。"""
    sop_uid: str
    series_uid: str
    study_uid: str
    boxes: list[BBox] = field(default_factory=list)

    @property
    def label_names(self) -> set[str]:
        return {b.label_name for b in self.boxes}

    @property
    def total_area(self) -> float:
        return sum(b.area for b in self.boxes)


class BHXAnnotations:
    """BHX 标注索引。

    提供两种查询方式:
        by_sop[sop_uid]     → SliceAnnotation
        by_study[study_uid] → list[SliceAnnotation]
    """

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.by_sop: dict[str, SliceAnnotation] = {}
        self.by_study: dict[str, list[SliceAnnotation]] = defaultdict(list)
        self.by_series: dict[str, list[SliceAnnotation]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sop_uid = row["SOPInstanceUID"]
                series_uid = row["SeriesInstanceUID"]
                study_uid = row["StudyInstanceUID"]

                try:
                    data = ast.literal_eval(row["data"])
                except (ValueError, SyntaxError):
                    continue

                bbox = BBox(
                    x=float(data.get("x", 0)),
                    y=float(data.get("y", 0)),
                    width=float(data.get("width", 0)),
                    height=float(data.get("height", 0)),
                    label_name=row.get("labelName", ""),
                    label_type=row.get("labelType", ""),
                )

                if sop_uid not in self.by_sop:
                    ann = SliceAnnotation(
                        sop_uid=sop_uid,
                        series_uid=series_uid,
                        study_uid=study_uid,
                    )
                    self.by_sop[sop_uid] = ann
                    self.by_study[study_uid].append(ann)
                    self.by_series[series_uid].append(ann)

                self.by_sop[sop_uid].boxes.append(bbox)

        n_slices = len(self.by_sop)
        n_studies = len(self.by_study)
        n_boxes = sum(len(a.boxes) for a in self.by_sop.values())
        logger.info(
            "BHX loaded: %d annotated slices, %d studies, %d boxes from %s",
            n_slices, n_studies, n_boxes, self.csv_path,
        )

    def get_annotated_sops_for_series(self, series_uid: str) -> list[SliceAnnotation]:
        """获取某个 SeriesInstanceUID 下的全部标注切片。"""
        return self.by_series.get(series_uid, [])

    def get_annotated_sops_for_study(self, study_uid: str) -> list[SliceAnnotation]:
        """获取某个 StudyInstanceUID 下的全部标注切片。"""
        return self.by_study.get(study_uid, [])

    def has_annotation(self, sop_uid: str) -> bool:
        return sop_uid in self.by_sop

    @property
    def all_study_uids(self) -> set[str]:
        return set(self.by_study.keys())

    @property
    def all_series_uids(self) -> set[str]:
        return set(self.by_series.keys())

    @property
    def stats(self) -> dict[str, Any]:
        label_counts: dict[str, int] = defaultdict(int)
        for ann in self.by_sop.values():
            for b in ann.boxes:
                label_counts[b.label_name] += 1
        return {
            "n_annotated_slices": len(self.by_sop),
            "n_studies": len(self.by_study),
            "n_series": len(self.by_series),
            "label_counts": dict(label_counts),
        }
