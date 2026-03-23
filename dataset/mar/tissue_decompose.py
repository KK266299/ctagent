# ============================================================================
# 模块职责: HU → 线衰减系数转换 + 水/骨组织成分分解
#   将 CT 图像分解为水和骨两个材料分量，用于独立建模不同材料的 X 射线衰减
#   支持多种输入源: DeepLesion PNG / CQ500 DICOM / 通用 HU ndarray
# 参考: ADN — +helper/simulate_metal_artifact.m Step 1-2
# ============================================================================
from __future__ import annotations

import numpy as np
from PIL import Image


def load_deeplesion_png(path: str, target_size: int = 416) -> np.ndarray:
    """加载 DeepLesion 16-bit PNG 并转为 HU 值。

    DeepLesion PNG 存储格式: uint16, 实际 HU = pixel * 65536 - 32768
    之后 resize 到 target_size × target_size 并截断下界到 -1000 HU。
    """
    img = np.array(Image.open(path)).astype(np.float64)
    hu = img / 65535.0 * (32767 - (-32768)) + (-32768)

    if img.shape[0] != target_size or img.shape[1] != target_size:
        pil_img = Image.fromarray(hu.astype(np.float32))
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        hu = np.array(pil_img).astype(np.float64)

    hu[hu < -1000] = -1000
    return hu


def load_ct_slice(
    path: str,
    target_size: int = 416,
    hu_clip_min: float = -1000.0,
    source_type: str = "auto",
) -> np.ndarray:
    """通用 CT 切片加载: 自动检测格式，统一输出 (target_size, target_size) HU 图像。

    Args:
        path: 文件路径 (.dcm / .png / .npy)
        target_size: 输出尺寸
        hu_clip_min: HU 下界截断
        source_type: "auto" / "dicom" / "deeplesion_png" / "npy"
    """
    if source_type == "auto":
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext == "dcm":
            source_type = "dicom"
        elif ext == "png":
            source_type = "deeplesion_png"
        elif ext == "npy":
            source_type = "npy"
        else:
            source_type = "dicom"

    if source_type == "dicom":
        from dataset.mar.cq500_reader import read_dicom_hu
        return read_dicom_hu(path, target_size, hu_clip_min)
    elif source_type == "deeplesion_png":
        return load_deeplesion_png(path, target_size)
    elif source_type == "npy":
        hu = np.load(path).astype(np.float64)
        if hu.shape[0] != target_size or hu.shape[1] != target_size:
            pil_img = Image.fromarray(hu.astype(np.float32))
            pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
            hu = np.array(pil_img).astype(np.float64)
        hu[hu < hu_clip_min] = hu_clip_min
        return hu
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


def hu_to_mu(hu_image: np.ndarray, mu_water: float = 0.192) -> np.ndarray:
    """HU → 线衰减系数 (linear attenuation coefficient)。

    μ = HU / 1000 × μ_water + μ_water
    """
    return hu_image / 1000.0 * mu_water + mu_water


def decompose_tissue(
    mu_image: np.ndarray,
    thresh_water: float,
    thresh_bone: float,
) -> tuple[np.ndarray, np.ndarray]:
    """将线衰减系数图分解为水成分和骨成分。

    纯水区 (μ <= thresh_water): 全部归水
    纯骨区 (μ >= thresh_bone): 全部归骨
    混合区: 按线性插值分配

    Returns:
        (img_water, img_bone) 两个分量图
    """
    img_water = np.zeros_like(mu_image)
    img_bone = np.zeros_like(mu_image)

    bw_water = mu_image <= thresh_water
    bw_bone = mu_image >= thresh_bone
    bw_both = ~bw_water & ~bw_bone

    img_water[bw_water] = mu_image[bw_water]
    img_bone[bw_bone] = mu_image[bw_bone]

    bone_frac = (mu_image[bw_both] - thresh_water) / (thresh_bone - thresh_water)
    img_bone[bw_both] = bone_frac * mu_image[bw_both]
    img_water[bw_both] = mu_image[bw_both] - img_bone[bw_both]

    return img_water, img_bone
