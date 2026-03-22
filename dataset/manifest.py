# ============================================================================
# 模块职责: Manifest 读写 — JSONL 格式的数据集索引管理
#   每行一个 JSON 对象, 记录一个 slice 的路径和 metadata
#   支持: 写入、追加、读取、过滤、统计
# 参考: ADN — 数据集的目录结构索引思路, 改为 JSONL 以支持更灵活的 metadata
# ============================================================================
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def write_manifest(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """将记录列表写入 JSONL 文件。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    logger.info("Wrote %d records to %s", len(records), out)


def append_manifest(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """追加记录到 JSONL 文件。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    logger.info("Appended %d records to %s", len(records), out)


def read_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL manifest 为记录列表。"""
    path = Path(manifest_path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_manifest(manifest_path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行迭代 manifest (节省内存)。"""
    path = Path(manifest_path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def manifest_stats(manifest_path: str | Path) -> dict[str, Any]:
    """统计 manifest 的基本信息。"""
    records = read_manifest(manifest_path)
    if not records:
        return {"total": 0}

    patients = set()
    series = set()
    for r in records:
        patients.add(r.get("patient_id", ""))
        series.add(r.get("series_uid", ""))

    return {
        "total": len(records),
        "patients": len(patients),
        "series": len(series),
    }
