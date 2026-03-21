# ============================================================================
# 模块职责: 数据集 — CT 数据加载、toy 数据生成、退化模拟
# 参考: src/datasets/ — CTDataset / PairedCTDataset
#       src/degradations/simulator.py — DegradationSimulator
# ============================================================================

from dataset.toy import generate_toy_phantom, generate_toy_case, ToyLabel

__all__ = [
    "generate_toy_phantom",
    "generate_toy_case",
    "ToyLabel",
]
