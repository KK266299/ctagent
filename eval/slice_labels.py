# ============================================================================
# 模块职责: 逐 Slice 标签生成
#   结合 CQ500Labels (case-level GT), BHXAnnotations (slice-level 病灶),
#   SOPIndex (SOP↔processed path 映射) 为每张 slice 分配诊断标签。
#
#   两种策略:
#     inherit_all — 有病灶 → 继承全部 case-level GT; 无病灶 → 全 0
#     bhx_aware   — 有病灶 → 出血类按 BHX label 映射; 非出血类继承 case-level
# ============================================================================
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval.bhx_loader import BHXAnnotations
from eval.cq500_labels import CQ500Labels, DIAGNOSIS_LABELS, normalize_case_id
from eval.cq500_manifest import SOPIndex

logger = logging.getLogger(__name__)

BHX_TO_DIAGNOSIS: dict[str, str] = {
    "Intraparenchymal": "IPH",
    "Intraventricular": "IVH",
    "Subarachnoid": "SAH",
    "Subdural": "SDH",
    "Epidural": "EDH",
}

HEMORRHAGE_SUBTYPES = {"IPH", "IVH", "SAH", "SDH", "EDH"}
NON_HEMORRHAGE_LABELS = [l for l in DIAGNOSIS_LABELS if l not in HEMORRHAGE_SUBTYPES and l != "ICH"]


@dataclass
class SliceLabelEntry:
    """单个 slice 的标签记录。"""
    patient_id: str
    series: str
    slice_name: str
    sop_uid: str
    bhx_coverage: bool
    lesion_present: int
    labels: dict[str, int]
    bhx_label_names: list[str] = field(default_factory=list)
    strategy: str = "inherit_all"


class SliceLabelGenerator:
    """逐 slice 标签生成器。"""

    def __init__(
        self,
        labels: CQ500Labels,
        bhx: BHXAnnotations,
        sop_index: SOPIndex,
        processed_dir: str | Path,
        strategy: str = "inherit_all",
    ) -> None:
        self.labels = labels
        self.bhx = bhx
        self.sop_index = sop_index
        self.processed_dir = Path(processed_dir)
        self.strategy = strategy

        self._bhx_patient_ids: set[str] = set()
        for study_uid in bhx.all_study_uids:
            pid = sop_index._study_uid_to_patient.get(study_uid)
            if pid:
                self._bhx_patient_ids.add(pid)
        logger.info("BHX covers %d patients", len(self._bhx_patient_ids))

    def generate(self) -> list[SliceLabelEntry]:
        """扫描 processed 目录，为每张 slice 生成标签。"""
        entries: list[SliceLabelEntry] = []

        for patient_dir in sorted(self.processed_dir.iterdir()):
            if not patient_dir.is_dir() or patient_dir.name.startswith("."):
                continue
            patient_id = patient_dir.name
            case_gt = self.labels.get_gt(patient_id)
            if case_gt is None:
                continue

            has_bhx = patient_id in self._bhx_patient_ids

            for series_dir in sorted(patient_dir.iterdir()):
                if not series_dir.is_dir():
                    continue
                series_name = series_dir.name

                for slice_dir in sorted(series_dir.iterdir()):
                    if not slice_dir.is_dir():
                        continue
                    gt_path = slice_dir / "gt.h5"
                    if not gt_path.exists():
                        continue

                    slice_name = slice_dir.name
                    proc_path = f"{patient_id}/{series_name}/{slice_name}"
                    sop_uid = self.sop_index.processed_path_to_sop(proc_path) or ""

                    entry = self._make_entry(
                        patient_id, series_name, slice_name, sop_uid,
                        has_bhx, case_gt,
                    )
                    entries.append(entry)

        logger.info(
            "Generated %d slice labels (%d lesion-positive, %d bhx-covered patients)",
            len(entries),
            sum(1 for e in entries if e.lesion_present),
            len(self._bhx_patient_ids),
        )
        return entries

    def _make_entry(
        self,
        patient_id: str,
        series: str,
        slice_name: str,
        sop_uid: str,
        has_bhx: bool,
        case_gt: dict[str, int],
    ) -> SliceLabelEntry:
        if not has_bhx:
            return SliceLabelEntry(
                patient_id=patient_id,
                series=series,
                slice_name=slice_name,
                sop_uid=sop_uid,
                bhx_coverage=False,
                lesion_present=0,
                labels={lbl: 0 for lbl in DIAGNOSIS_LABELS},
                strategy="case_fallback",
            )

        has_lesion = bool(sop_uid and self.bhx.has_annotation(sop_uid))
        if not has_lesion:
            return SliceLabelEntry(
                patient_id=patient_id,
                series=series,
                slice_name=slice_name,
                sop_uid=sop_uid,
                bhx_coverage=True,
                lesion_present=0,
                labels={lbl: 0 for lbl in DIAGNOSIS_LABELS},
                strategy=self.strategy,
            )

        ann = self.bhx.by_sop[sop_uid]
        bhx_names = list(ann.label_names)

        if self.strategy == "bhx_aware":
            slice_labels = self._bhx_aware_labels(bhx_names, case_gt)
        else:
            slice_labels = dict(case_gt)

        return SliceLabelEntry(
            patient_id=patient_id,
            series=series,
            slice_name=slice_name,
            sop_uid=sop_uid,
            bhx_coverage=True,
            lesion_present=1,
            labels=slice_labels,
            bhx_label_names=bhx_names,
            strategy=self.strategy,
        )

    @staticmethod
    def _bhx_aware_labels(
        bhx_names: list[str],
        case_gt: dict[str, int],
    ) -> dict[str, int]:
        """根据 BHX 标注精确映射出血子类型，非出血类继承 case-level。"""
        labels: dict[str, int] = {}
        mapped_subtypes: set[str] = set()

        for bname in bhx_names:
            diag = BHX_TO_DIAGNOSIS.get(bname)
            if diag:
                mapped_subtypes.add(diag)

        for lbl in DIAGNOSIS_LABELS:
            if lbl == "ICH":
                labels[lbl] = 1 if mapped_subtypes else 0
            elif lbl in HEMORRHAGE_SUBTYPES:
                labels[lbl] = 1 if lbl in mapped_subtypes else 0
            else:
                labels[lbl] = case_gt.get(lbl, 0)

        return labels


# ---------------------------------------------------------------------------
# I/O: 导出 CSV + JSONL
# ---------------------------------------------------------------------------

def save_slice_labels_csv(
    entries: list[SliceLabelEntry],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "patient_id", "series", "slice_name", "sop_uid",
        "bhx_coverage", "lesion_present",
        *DIAGNOSIS_LABELS,
        "bhx_labels", "strategy",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            row: dict[str, Any] = {
                "patient_id": e.patient_id,
                "series": e.series,
                "slice_name": e.slice_name,
                "sop_uid": e.sop_uid,
                "bhx_coverage": int(e.bhx_coverage),
                "lesion_present": e.lesion_present,
                "bhx_labels": "|".join(e.bhx_label_names),
                "strategy": e.strategy,
            }
            for lbl in DIAGNOSIS_LABELS:
                row[lbl] = e.labels.get(lbl, 0)
            writer.writerow(row)

    logger.info("Saved %d slice labels to %s", len(entries), path)


def save_slice_labels_jsonl(
    entries: list[SliceLabelEntry],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    logger.info("Saved %d slice labels to %s", len(entries), path)


def save_slice_label_stats(
    entries: list[SliceLabelEntry],
    output_path: str | Path,
) -> None:
    """输出标签分布统计。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(entries)
    bhx_covered = sum(1 for e in entries if e.bhx_coverage)
    lesion_pos = sum(1 for e in entries if e.lesion_present)
    patients = set(e.patient_id for e in entries)
    bhx_patients = set(e.patient_id for e in entries if e.bhx_coverage)

    label_dist: dict[str, dict[str, int]] = {}
    for lbl in DIAGNOSIS_LABELS:
        pos = sum(1 for e in entries if e.labels.get(lbl, 0) == 1)
        label_dist[lbl] = {"positive": pos, "negative": total - pos}

    bhx_label_counts: dict[str, int] = defaultdict(int)
    for e in entries:
        for bn in e.bhx_label_names:
            bhx_label_counts[bn] += 1

    stats = {
        "total_slices": total,
        "total_patients": len(patients),
        "bhx_covered_slices": bhx_covered,
        "bhx_covered_patients": len(bhx_patients),
        "non_bhx_slices": total - bhx_covered,
        "lesion_positive_slices": lesion_pos,
        "lesion_negative_slices": total - lesion_pos,
        "lesion_positive_rate": round(lesion_pos / max(total, 1), 4),
        "label_distribution": label_dist,
        "bhx_label_counts": dict(bhx_label_counts),
    }

    with open(path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("Saved label stats to %s", path)


# ---------------------------------------------------------------------------
# I/O: 加载
# ---------------------------------------------------------------------------

class SliceLabelStore:
    """加载已生成的 slice labels，提供 per-slice GT 查询。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, SliceLabelEntry] = {}
        self._load()

    def _key(self, patient_id: str, series: str, slice_name: str) -> str:
        return f"{patient_id}/{series}/{slice_name}"

    def _load(self) -> None:
        if self.path.suffix == ".csv":
            self._load_csv()
        else:
            self._load_jsonl()
        logger.info("SliceLabelStore loaded %d entries from %s", len(self._entries), self.path)

    def _load_csv(self) -> None:
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                labels = {lbl: int(row[lbl]) for lbl in DIAGNOSIS_LABELS}
                bhx_labels = row.get("bhx_labels", "")
                entry = SliceLabelEntry(
                    patient_id=row["patient_id"],
                    series=row["series"],
                    slice_name=row["slice_name"],
                    sop_uid=row.get("sop_uid", ""),
                    bhx_coverage=bool(int(row.get("bhx_coverage", 0))),
                    lesion_present=int(row.get("lesion_present", 0)),
                    labels=labels,
                    bhx_label_names=bhx_labels.split("|") if bhx_labels else [],
                    strategy=row.get("strategy", ""),
                )
                key = self._key(entry.patient_id, entry.series, entry.slice_name)
                self._entries[key] = entry

    def _load_jsonl(self) -> None:
        with open(self.path) as f:
            for line in f:
                data = json.loads(line)
                entry = SliceLabelEntry(**data)
                key = self._key(entry.patient_id, entry.series, entry.slice_name)
                self._entries[key] = entry

    def get_gt(
        self,
        patient_id: str,
        series: str,
        slice_name: str,
    ) -> dict[str, int] | None:
        key = self._key(patient_id, series, slice_name)
        entry = self._entries.get(key)
        return entry.labels if entry else None

    def get_entry(
        self,
        patient_id: str,
        series: str,
        slice_name: str,
    ) -> SliceLabelEntry | None:
        key = self._key(patient_id, series, slice_name)
        return self._entries.get(key)

    @property
    def entries(self) -> list[SliceLabelEntry]:
        return list(self._entries.values())

    def bhx_covered_entries(self) -> list[SliceLabelEntry]:
        return [e for e in self._entries.values() if e.bhx_coverage]
