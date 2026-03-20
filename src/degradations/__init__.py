# ============================================================================
# 模块职责: 退化建模与感知 — 检测图像退化类型和严重程度
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — degradation detection
#       ProCT (https://github.com/Masaaki-75/proct)
#       PromptCT (https://github.com/shibaoshun/PromptCT)
# ============================================================================

from src.degradations.detector import DegradationDetector
from src.degradations.simulator import DegradationSimulator
from src.degradations.types import DegradationType, DegradationReport

__all__ = [
    "DegradationDetector",
    "DegradationSimulator",
    "DegradationType",
    "DegradationReport",
]
