#!/usr/bin/env python3
# ============================================================================
# 模块职责: 构建 SOPInstanceUID → 已处理切片目录的映射索引
#   扫描原始 CQ500 DICOM 头 (stop_before_pixels), 只读取 UID 标签
#   与 cq500_processed/ 目录中已处理的 slice 交叉对齐
#   输出缓存 JSON 供评测时快速查询
# ============================================================================
"""
用法:
    python scripts/build_sop_index.py \
        --cq500-root /home/liuxinyao/data/cq500 \
        --processed-root /home/liuxinyao/data/cq500_processed \
        --output /home/liuxinyao/data/cq500_sop_index.json

输出 JSON 结构:
{
  "sop_to_slice": {
    "<SOPInstanceUID>": {
      "patient_id": "CQ500CT0",
      "series_folder": "PLAIN_THIN",
      "dcm_stem": "CT000000",
      "processed_path": "CQ500CT0/PLAIN_THIN/CT000000"
    },
    ...
  },
  "series_uid_to_folder": {
    "<SeriesInstanceUID>": {
      "patient_id": "CQ500CT0",
      "series_folder": "PLAIN_THIN"
    },
    ...
  },
  "study_uid_to_patient": {
    "<StudyInstanceUID>": "CQ500CT0",
    ...
  },
  "stats": { ... }
}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_original_dcm(
    patient_id: str,
    series_folder: str,
    dcm_stem: str,
    cq500_root: Path,
) -> Path | None:
    """在原始 CQ500 目录中定位 DICOM 文件。

    CQ500 结构: root/qctXX/<PatientID> <PatientID>/Unknown Study/CT <desc>/<dcm_stem>.dcm
    已处理 series_folder (如 PLAIN_THIN) 对应原始 "CT PLAIN THIN" 或 "CT Plain" 等。
    """
    for batch_dir in cq500_root.iterdir():
        if not batch_dir.is_dir() or batch_dir.name.startswith("."):
            continue
        for patient_dir in batch_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            if patient_dir.name.split(" ")[0] != patient_id:
                continue

            for study_dir in patient_dir.rglob("Unknown Study"):
                for series_dir in study_dir.iterdir():
                    if not series_dir.is_dir():
                        continue
                    series_desc = series_dir.name
                    if series_desc.startswith("CT "):
                        series_desc = series_desc[3:]
                    normalized = series_desc.replace(" ", "_")
                    if normalized == series_folder:
                        dcm_path = series_dir / f"{dcm_stem}.dcm"
                        if dcm_path.exists():
                            return dcm_path
    return None


def build_patient_dcm_lookup(cq500_root: Path) -> dict[str, dict[str, Path]]:
    """预先扫描全部 CQ500 目录，建立 patient→series→dcm_paths 查找表。"""
    lookup: dict[str, dict[str, Path]] = {}

    for batch_dir in sorted(cq500_root.iterdir()):
        if not batch_dir.is_dir() or batch_dir.name.startswith("."):
            continue
        for patient_dir in sorted(batch_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = patient_dir.name.split(" ")[0]

            for study_dir in patient_dir.rglob("Unknown Study"):
                for series_dir in study_dir.iterdir():
                    if not series_dir.is_dir():
                        continue
                    series_desc = series_dir.name
                    if series_desc.startswith("CT "):
                        series_desc = series_desc[3:]
                    normalized = series_desc.replace(" ", "_")

                    key = f"{pid}/{normalized}"
                    lookup[key] = {}
                    for dcm_path in series_dir.glob("*.dcm"):
                        lookup[key][dcm_path.stem] = dcm_path

    return lookup


def main():
    parser = argparse.ArgumentParser(description="Build SOPInstanceUID → processed slice index")
    parser.add_argument("--cq500-root", required=True, help="Original CQ500 data root")
    parser.add_argument("--processed-root", required=True, help="cq500_processed root")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    import pydicom

    cq500_root = Path(args.cq500_root)
    processed_root = Path(args.processed_root)
    output_path = Path(args.output)

    logger.info("Step 1: Scanning CQ500 directory structure...")
    dcm_lookup = build_patient_dcm_lookup(cq500_root)
    logger.info("Built lookup for %d patient/series combinations", len(dcm_lookup))

    logger.info("Step 2: Scanning processed directory for slice list...")
    tasks: list[tuple[str, str, str]] = []
    for patient_dir in sorted(processed_root.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("."):
            continue
        pid = patient_dir.name
        for series_dir in sorted(patient_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            sfolder = series_dir.name
            for slice_dir in sorted(series_dir.iterdir()):
                if not slice_dir.is_dir():
                    continue
                tasks.append((pid, sfolder, slice_dir.name))

    logger.info("Found %d processed slices to index", len(tasks))

    logger.info("Step 3: Reading DICOM headers...")
    sop_to_slice: dict[str, dict] = {}
    series_uid_to_folder: dict[str, dict] = {}
    study_uid_to_patient: dict[str, str] = {}
    n_found = 0
    n_missing = 0
    t0 = time.time()

    for i, (pid, sfolder, dcm_stem) in enumerate(tasks):
        key = f"{pid}/{sfolder}"
        dcm_paths = dcm_lookup.get(key, {})
        dcm_path = dcm_paths.get(dcm_stem)

        if dcm_path is None:
            n_missing += 1
            continue

        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
            series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
            study_uid = str(getattr(ds, "StudyInstanceUID", ""))

            if sop_uid:
                sop_to_slice[sop_uid] = {
                    "patient_id": pid,
                    "series_folder": sfolder,
                    "dcm_stem": dcm_stem,
                    "processed_path": f"{pid}/{sfolder}/{dcm_stem}",
                }
                n_found += 1

            if series_uid and series_uid not in series_uid_to_folder:
                series_uid_to_folder[series_uid] = {
                    "patient_id": pid,
                    "series_folder": sfolder,
                }

            if study_uid and study_uid not in study_uid_to_patient:
                study_uid_to_patient[study_uid] = pid

        except Exception as e:
            logger.debug("Failed to read %s: %s", dcm_path, e)
            n_missing += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info("  Progress: %d/%d (%.0f/s)", i + 1, len(tasks), rate)

    elapsed = time.time() - t0
    logger.info(
        "Step 3 done: %d indexed, %d missing, %.1fs",
        n_found, n_missing, elapsed,
    )

    index = {
        "sop_to_slice": sop_to_slice,
        "series_uid_to_folder": series_uid_to_folder,
        "study_uid_to_patient": study_uid_to_patient,
        "stats": {
            "n_processed_slices": len(tasks),
            "n_indexed": n_found,
            "n_missing": n_missing,
            "n_series": len(series_uid_to_folder),
            "n_studies": len(study_uid_to_patient),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info("Saved index to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
