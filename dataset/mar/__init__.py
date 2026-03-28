# ============================================================================
# 模块职责: MAR (Metal Artifact Reduction) 退化数据集构建
#   基于物理仿真合成含金属伪影的 CT 图像，形成配对 (GT, 退化) 数据集
# 参考: ADN (https://github.com/liaohaofu/adn) MATLAB 实现
# ============================================================================

try:
    from dataset.mar.ct_geometry import CTGeometry
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lightweight envs
    CTGeometry = None

from dataset.mar.physics_params import PhysicsParams

try:
    from dataset.mar.mar_simulator import MARSimulator
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lightweight envs
    MARSimulator = None
from dataset.mar.ct_artifact_simulator import (
    ARTIFACT_SIMULATOR_REGISTRY,
    BaseCTArtifactSimulator,
    BeamHardeningArtifactSimulator,
    CompositeArtifactSimulator,
    MotionArtifactSimulator,
    RingArtifactSimulator,
    ScatterArtifactSimulator,
    TruncationArtifactSimulator,
    create_artifact_simulator,
)

__all__ = [
    "CTGeometry",
    "PhysicsParams",
    "MARSimulator",
    "BaseCTArtifactSimulator",
    "RingArtifactSimulator",
    "MotionArtifactSimulator",
    "BeamHardeningArtifactSimulator",
    "ScatterArtifactSimulator",
    "TruncationArtifactSimulator",
    "CompositeArtifactSimulator",
    "ARTIFACT_SIMULATOR_REGISTRY",
    "create_artifact_simulator",
]
