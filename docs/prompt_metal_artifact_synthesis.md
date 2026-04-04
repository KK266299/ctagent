# Cursor 提示词：为数据集构建流水线添加 CT 金属伪影退化类型

## 任务目标

在现有的退化数据集构建流水线中，新增 `metal_artifact`（金属伪影）退化类型。当前流水线已支持 `noise`、`blur`、`downsample`、`artifact`（简化条纹）四种退化。新增的金属伪影退化需要基于 **物理模型**（多能谱 X 射线投影、射束硬化、泊松噪声），参考 [MAR_SynCode](https://github.com/KK266299/MAR_SynCode) 的实现。

---

## 现有代码结构（必读）

### 退化构建器
- **`dataset/degradation_builder.py`** — 核心退化引擎
  - `DEGRADE_FN` 字典注册所有退化函数：`{"noise": degrade_noise, "blur": degrade_blur, ...}`
  - `DEFAULT_PARAMS` 字典注册每种退化的 severity 级别参数
  - `build_degraded_dataset()` 批量处理：读 clean manifest → 对每张切片施加退化 → 保存 .npy → 写入 degraded manifest
  - 每个退化函数签名：`def degrade_xxx(image: np.ndarray, ..., rng=None) -> np.ndarray`
  - 输入输出都是 **归一化到 [0,1] 的 float32 图像**（已经过窗宽窗位归一化）

### 配置文件
- **`configs/data/degradation.yaml`** — 退化配置
  ```yaml
  degradations:
    - type: "noise"
      severities: [1, 2, 3, 4, 5]
    - type: "metal_artifact"          # ← 新增
      severities: [1, 2, 3]
  ```

### 入口脚本
- **`scripts/build_degraded_dataset.py`** — 读取 YAML → 调用 `build_degraded_dataset()`

### 输出结构
```
output/degraded/metal_artifact/severity_1/{slice_id}.npy
output/degraded/metal_artifact/severity_2/{slice_id}.npy
output/degraded/metal_artifact/severity_3/{slice_id}.npy
```

---

## MAR_SynCode 物理模型（参考实现）

以下是金属伪影合成的完整物理流水线，你需要将其简化并集成到本项目中：

### 物理流程概览

```
干净 CT 图像 (归一化 [0,1])
    │
    ▼
1. 反归一化 → HU → 线性衰减系数 (μ)
    │
    ▼
2. 组织分解：将图像分为「水」和「骨」两种基材料
   μ_water = 0.192 cm⁻¹
   阈值: threshWater ≈ 0.2112, threshBone ≈ 0.480
   • μ ≤ threshWater → 纯水
   • μ ≥ threshBone  → 纯骨
   • 中间区域      → 按线性比例分配
    │
    ▼
3. 前向投影（Radon 变换）→ 生成水/骨的单能正弦图
   使用 ODL 库 (odl.tomo.RayTransform) 或 scikit-image 的 radon 变换
   扇束几何：640 views, 641 detector bins, SOD=1075
    │
    ▼
4. 金属掩码叠加
   • 生成或加载椭圆/矩形金属掩码（模拟植入物）
   • 前向投影金属掩码，乘以金属衰减系数 (钛: 4.5 × μ_Ti@70keV)
   • 部分体积效应：边缘像素衰减降低至 1/4
    │
    ▼
5. 多能谱投影合成 (pkev2kvp)
   对每个能量 E ∈ [20, 120] keV：
     P_total(E) = Σ_m [μ_m(E)/μ_m(70keV)] × P_m(70keV)
     I(E) = S(E) × exp(-P_total(E))
   合成投影: P_kvp = -ln(Σ_E I(E) / Σ_E S(E))
    │
    ▼
6. 泊松噪声 + 散射
   photons = Poisson(exp(-P_kvp) × N₀ + scatter)
   P_noisy = -ln(photons / N₀)
   N₀ = 2×10⁷, scatter = 20 photons
    │
    ▼
7. 射束硬化校正 (BHC) — 三阶多项式水校正
   P_corrected = a₁·P + a₂·P² + a₃·P³
    │
    ▼
8. FBP 重建 → 含金属伪影的 CT 图像
    │
    ▼
9. 归一化回 [0,1] → 输出退化图像
```

### 材料数据

| 材料 | 密度 (g/cm³) | 70keV 线性衰减系数 (cm⁻¹) |
|------|-------------|-------------------------|
| 水 (H₂O) | 1.0 | 0.192 |
| 皮质骨 | 1.92 | ~0.573 |
| 钛 (Ti) | 4.5 | ~1.328 |
| 铁 (Fe) | 7.8 | ~2.112 |

### Severity 级别建议

| 严重等级 | 金属类型 | 掩码大小 | 掩码数量 | 光子数 N₀ | 效果 |
|---------|---------|---------|---------|----------|------|
| 1 (轻) | 钛 | 小椭圆 (5-10px) | 1 | 2×10⁷ | 轻微条纹 |
| 2 (中) | 钛 | 中椭圆 (10-20px) | 1-2 | 1×10⁷ | 明显条纹+阴影 |
| 3 (重) | 铁 | 大椭圆 (15-30px) | 2-3 | 5×10⁶ | 严重条纹+光子饥饿 |

---

## 实现要求

### 1. 新增退化函数

在 `dataset/degradation_builder.py` 中添加：

```python
def degrade_metal_artifact(
    image: np.ndarray,
    metal_type: str = "Ti",
    mask_size: tuple[int, int] = (10, 15),
    num_masks: int = 1,
    photon_num: float = 2e7,
    scatter_photons: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """基于物理模型的 CT 金属伪影合成。"""
    ...
```

**关键要求：**
- 输入/输出均为 `[0, 1]` 归一化的 `float32` 图像
- 内部需要反归一化到 HU / 线性衰减系数域进行物理运算
- 使用 `scikit-image` 的 `radon`/`iradon` 代替 ODL（更轻量，无需 ASTRA CUDA）
- 金属掩码在图像中心附近随机放置（避免边缘），形状为随机椭圆
- 必须包含：组织分解 → 前向投影 → 金属叠加 → 多能谱合成 → 泊松噪声 → BHC → FBP 重建 完整流程
- 如果多能谱合成过于复杂，可以用 **简化版射束硬化模型**：对金属投影区域施加非线性衰减偏移

### 2. 简化版实现（如果完整物理模型太复杂）

可以采用以下简化策略，仍然能产生逼真的金属伪影：

```python
def degrade_metal_artifact_simplified(image, ...):
    """简化版金属伪影合成（正弦图域操作）。"""
    # 1. 生成随机椭圆金属掩码
    # 2. Radon 变换 → 正弦图
    # 3. 在正弦图中金属轨迹区域：
    #    a) 添加强偏移（模拟射束硬化）
    #    b) 添加泊松噪声（模拟光子饥饿）
    # 4. FBP 反投影重建
    # 5. 在金属区域恢复原始值
    # 6. 归一化回 [0,1]
```

### 3. 参数预设

在 `DEFAULT_PARAMS` 字典中注册：

```python
METAL_ARTIFACT_PARAMS: dict[int, dict[str, Any]] = {
    1: {"metal_type": "Ti", "mask_size": (5, 10), "num_masks": 1, "photon_num": 2e7},
    2: {"metal_type": "Ti", "mask_size": (10, 20), "num_masks": 2, "photon_num": 1e7},
    3: {"metal_type": "Fe", "mask_size": (15, 30), "num_masks": 2, "photon_num": 5e6},
}
```

### 4. 注册到 DEGRADE_FN

```python
DEGRADE_FN["metal_artifact"] = degrade_metal_artifact
```

### 5. 更新配置文件

在 `configs/data/degradation.yaml` 中新增：

```yaml
  - type: "metal_artifact"
    severities: [1, 2, 3]
```

### 6. 额外输出（可选但推荐）

每个金属伪影退化还应额外保存以下辅助数据（供 MAR 模型训练使用）：
- `metal_mask`: 金属二值掩码 (uint8)
- `metal_trace`: 正弦图域金属轨迹 (uint8)
- `li_image`: 线性插值校正后的图像 (float32)

这些可以保存为额外的 .npy 文件（同目录，命名为 `{slice_id}_mask.npy` 等），并在 manifest 中增加对应字段。

---

## 依赖

- `scikit-image`（`skimage.transform.radon`, `skimage.transform.iradon`）— 正/反 Radon 变换
- `scipy.ndimage`（`binary_erosion`, `gaussian_filter`）— 形态学操作
- `numpy`（已有）

**不要引入 ODL 或 ASTRA 依赖**，使用 scikit-image 的平行束 Radon 变换即可。

---

## 需要修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `dataset/degradation_builder.py` | 新增 `degrade_metal_artifact()` 函数、`METAL_ARTIFACT_PARAMS` 参数预设、注册到 `DEGRADE_FN` 和 `DEFAULT_PARAMS` |
| `configs/data/degradation.yaml` | 新增 `metal_artifact` 退化类型配置 |
| `docs/dataset_construction.md` | 新增金属伪影退化类型的文档说明 |

---

## 验证标准

1. `PYTHONPATH=. python scripts/build_degraded_dataset.py` 能正常运行并生成 `output/degraded/metal_artifact/` 目录
2. 生成的退化图像中可见典型的金属伪影特征：从金属位置辐射出的亮/暗条纹
3. severity 1→3 伪影强度递增
4. 输出图像仍为 `[0, 1]` 范围的 `float32`
5. manifest 正确记录 `degradation_type: "metal_artifact"` 及对应参数
