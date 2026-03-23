# ============================================================================
# 模块职责: CT 扇束几何构建 + 前向/反向投影
#   使用 ODL 库构建 Fan-Beam CT 几何，提供 ray_trafo 和 FBP 算子
# 参考: ADN — +helper/get_mar_params.m 中的几何参数
#       ODL documentation (https://odlgroup.github.io/odl/)
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import odl
import odl.applications.tomo as odl_tomo


@dataclass
class CTGeometryConfig:
    """CT 几何配置参数。"""
    image_size: int = 416
    num_angles: int = 640
    num_detectors: int = 641
    source_obj_dist_pixels: float = 1075.0
    orig_pixels: int = 512
    orig_pixel_size_cm: float = 0.03
    filter_type: str = "Ram-Lak"
    frequency_scaling: float = 1.0
    impl: str = "astra_cuda"


class CTGeometry:
    """ODL Fan-Beam CT 几何，封装前向投影与 FBP 重建。

    Usage:
        geo = CTGeometry(CTGeometryConfig())
        sinogram = geo.forward(image_2d)
        recon = geo.fbp(sinogram)
    """

    def __init__(self, config: CTGeometryConfig | None = None) -> None:
        cfg = config or CTGeometryConfig()
        self.config = cfg

        reso = cfg.orig_pixels / cfg.image_size * cfg.orig_pixel_size_cm
        self.reso = reso
        sx = cfg.image_size * reso
        sy = cfg.image_size * reso
        su = 2.0 * np.sqrt(sx**2 + sy**2)

        self.reco_space = odl.uniform_discr(
            min_pt=[-sx / 2, -sy / 2],
            max_pt=[sx / 2, sy / 2],
            shape=[cfg.image_size, cfg.image_size],
            dtype="float32",
        )

        angle_partition = odl.uniform_partition(0, 2 * np.pi, cfg.num_angles)
        detector_partition = odl.uniform_partition(-su / 2, su / 2, cfg.num_detectors)

        src_radius = cfg.source_obj_dist_pixels * reso
        det_radius = cfg.source_obj_dist_pixels * reso

        geometry = odl_tomo.FanBeamGeometry(
            angle_partition, detector_partition,
            src_radius=src_radius, det_radius=det_radius,
        )

        self.ray_trafo = odl_tomo.RayTransform(
            self.reco_space, geometry, impl=cfg.impl,
        )
        self.fbp_op = odl_tomo.fbp_op(
            self.ray_trafo,
            filter_type=cfg.filter_type,
            frequency_scaling=cfg.frequency_scaling,
        )

    def forward(self, image: np.ndarray) -> np.ndarray:
        """前向投影: 图像 → 正弦图。"""
        result = self.ray_trafo(image.astype(np.float32))
        return np.asarray(result.data).copy()

    def fbp(self, sinogram: np.ndarray) -> np.ndarray:
        """FBP 重建: 正弦图 → 图像。"""
        result = self.fbp_op(sinogram.astype(np.float32))
        return np.asarray(result.data).copy()
