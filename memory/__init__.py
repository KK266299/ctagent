# ============================================================================
# 模块职责: 经验记录与检索 — agent 的"长期记忆"
#   记录: 每次 pipeline 执行的完整经验 (退化→工具→judge→downstream)
#   检索: 按退化类型查询历史成功路径，供 planner 做 few-shot 参考
#   存储: JSON 文件 (轻量，科研友好，方便人工检查)
# 参考: AgenticIR — memory module (experience replay for agent)
#       4KAgent — reflection memory
# ============================================================================

from memory.experience import ExperienceRecord
from memory.store import ExperienceStore

__all__ = [
    "ExperienceRecord",
    "ExperienceStore",
]
