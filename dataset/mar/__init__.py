# ============================================================================
# 模块职责: MAR (Metal Artifact Reduction) 退化数据集构建
#   基于物理仿真合成含金属伪影的 CT 图像，形成配对 (GT, 退化) 数据集
# 参考: ADN (https://github.com/liaohaofu/adn) MATLAB 实现
# ============================================================================

from dataset.mar.ct_geometry import CTGeometry
from dataset.mar.physics_params import PhysicsParams
from dataset.mar.mar_simulator import MARSimulator

__all__ = [
    "CTGeometry",
    "PhysicsParams",
    "MARSimulator",
]
