# ============================================================================
# 模块职责: CQ500 标签加载 — reads.csv 解析、ID 归一化、majority vote GT
# ============================================================================
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DIAGNOSIS_LABELS = [
    "ICH", "IPH", "IVH", "SDH", "EDH", "SAH",
    "Fracture", "CalvarialFracture",
    "MassEffect", "MidlineShift",
]

ALL_READER_LABELS = [
    "ICH", "IPH", "IVH", "SDH", "EDH", "SAH",
    "BleedLocation-Left", "BleedLocation-Right", "ChronicBleed",
    "Fracture", "CalvarialFracture", "OtherFracture",
    "MassEffect", "MidlineShift",
]


def normalize_case_id(raw_id: str) -> str:
    """CQ500-CT-427 → CQ500CT427"""
    return re.sub(r"[^A-Za-z0-9]", "", raw_id)


class CQ500Labels:
    """CQ500 标签管理器。

    从 reads.csv 加载三名读片者的标签，计算 majority vote GT。
    """

    def __init__(self, reads_csv: str | Path) -> None:
        self.reads_csv = Path(reads_csv)
        self.raw_records: dict[str, dict[str, Any]] = {}
        self.gt: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        with open(self.reads_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_id = row["name"]
                case_id = normalize_case_id(raw_id)

                reader_votes: dict[str, list[int]] = {lbl: [] for lbl in ALL_READER_LABELS}
                for prefix in ["R1", "R2", "R3"]:
                    for lbl in ALL_READER_LABELS:
                        key = f"{prefix}:{lbl}"
                        val = row.get(key, "0")
                        reader_votes[lbl].append(int(float(val)))

                gt_labels = {}
                for lbl in DIAGNOSIS_LABELS:
                    votes = reader_votes[lbl]
                    gt_labels[lbl] = 1 if sum(votes) >= 2 else 0

                self.raw_records[case_id] = {
                    "raw_id": raw_id,
                    "category": row.get("Category", ""),
                    "reader_votes": reader_votes,
                }
                self.gt[case_id] = gt_labels

        logger.info("Loaded %d cases from %s", len(self.gt), self.reads_csv)

    def get_gt(self, case_id: str) -> dict[str, int] | None:
        nid = normalize_case_id(case_id)
        return self.gt.get(nid)

    def case_ids(self) -> list[str]:
        return list(self.gt.keys())

    def positive_counts(self) -> dict[str, int]:
        counts = {lbl: 0 for lbl in DIAGNOSIS_LABELS}
        for gt in self.gt.values():
            for lbl in DIAGNOSIS_LABELS:
                counts[lbl] += gt[lbl]
        return counts
