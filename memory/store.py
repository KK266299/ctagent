# ============================================================================
# 模块职责: ExperienceStore — 经验的持久化存储与检索
#   存储格式: JSON 文件 (一个文件存储所有 records)
#   检索能力: 按退化类型查询、按成功/失败过滤、获取最优路径
#   设计原则: 轻量、无外部依赖、科研友好 (JSON 可直接人工查看)
# 参考: AgenticIR — memory/experience_store.py
#       4KAgent — reflection memory storage
# ============================================================================
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from memory.experience import ExperienceRecord

logger = logging.getLogger(__name__)


class ExperienceStore:
    """JSON-based 经验存储。

    Usage:
        store = ExperienceStore("output/memory")
        store.add(record)
        hits = store.query_by_degradation(["noise"])
        best = store.get_best_route(["noise", "blur"])
    """

    def __init__(self, store_dir: str | Path = "output/memory") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._store_file = self.store_dir / "experiences.json"
        self._records: list[ExperienceRecord] = []
        self._load()

    def add(self, record: ExperienceRecord) -> None:
        """添加一条经验记录并持久化。"""
        self._records.append(record)
        self._save()
        logger.debug("Added experience %s (success=%s)", record.record_id, record.success)

    def query_by_degradation(self, degradation_types: list[str]) -> list[ExperienceRecord]:
        """按退化类型查询相关经验。

        匹配规则: 查询的退化类型是记录退化类型的子集即匹配。
        """
        query_set = set(degradation_types)
        return [
            r for r in self._records
            if query_set.issubset(set(r.degradation_types))
        ]

    def get_successful_routes(self, degradation_types: list[str] | None = None) -> list[ExperienceRecord]:
        """获取成功的修复路径。"""
        records = self._records if degradation_types is None else self.query_by_degradation(degradation_types)
        return [r for r in records if r.success]

    def get_best_route(self, degradation_types: list[str]) -> ExperienceRecord | None:
        """获取某退化组合下最优的成功路径 (按 quality score 排序)。"""
        successful = self.get_successful_routes(degradation_types)
        if not successful:
            return None
        return max(successful, key=lambda r: r.quality_verdict.get("score", 0.0))

    def get_failed_routes(self, degradation_types: list[str] | None = None) -> list[ExperienceRecord]:
        """获取失败的修复路径 (供 planner 避免重复)。"""
        records = self._records if degradation_types is None else self.query_by_degradation(degradation_types)
        return [r for r in records if not r.success]

    def summary(self) -> dict[str, Any]:
        """统计摘要。"""
        total = len(self._records)
        success = sum(1 for r in self._records if r.success)
        return {
            "total_records": total,
            "success_count": success,
            "failure_count": total - success,
            "success_rate": round(success / total, 3) if total > 0 else 0.0,
        }

    def __len__(self) -> int:
        return len(self._records)

    # ---- 持久化 ----

    def _save(self) -> None:
        data = [r.to_dict() for r in self._records]
        with open(self._store_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        if not self._store_file.exists():
            self._records = []
            return
        try:
            with open(self._store_file) as f:
                data = json.load(f)
            self._records = [ExperienceRecord.from_dict(d) for d in data]
            logger.info("Loaded %d experience records from %s", len(self._records), self._store_file)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to load experience store: %s", e)
            self._records = []
