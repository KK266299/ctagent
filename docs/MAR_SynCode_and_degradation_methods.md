# MAR_SynCode 代码库说明与退化方法详解

## 一、项目概述

**MAR_SynCode** 是一个 **CT 金属伪影合成** 代码库，用于从干净的 CT 图像合成带有金属伪影的退化数据。项目基于 **DeepLesion** 数据集，通过物理建模（多能谱 X 射线、泊松噪声、射束硬化效应）生成真实感的金属伪影 CT 图像，为金属伪影去除（MAR）算法的训练和评估提供配对数据。

### 技术栈

| 组件 | 技术 |
|------|------|
| 正/反投影 | ODL (Operator Discretization Library) + ASTRA CUDA |
| 几何模型 | 扇束（Fan-Beam）CT |
| 数据格式 | HDF5 (`.h5`) |
| 物理参数 | MATLAB `.mat` 文件（材料衰减系数、能谱） |

---

## 二、代码结构

```
MAR_SynCode/
├── prepare_deep_lesion.py        # 主入口：批量生成退化数据
├── simulate_data.py              # 核心：单张 CT 的金属伪影合成
├── util_func.py                  # 物理模型工具函数
├── util_func_diff_spectrum.py    # 差异能谱版工具函数
├── gene_h5list.py                # 生成 HDF5 文件列表（训练用）
├── adn/
│   ├── build_gemotry.py          # CT 扇束几何与投影算子构建
│   └── utils.py                  # 配置读取等工具
├── config/
│   └── dataset_py_640geo.yaml    # 数据集与 CT 参数配置
├── data/deep_lesion/
│   ├── metal_masks/              # 金属掩码 + 材料衰减系数 (.mat)
│   └── image_list.txt            # DeepLesion 图像列表
└── docker/                       # 容器化部署
```

---

## 三、CT 扇束几何建模

**文件：** `adn/build_gemotry.py`

使用 ODL 库构建扇束 CT 的前向投影（Radon 变换）和滤波反投影（FBP）算子。

### 几何参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像尺寸 | 416 × 416 | 重建图像像素数 |
| 像素分辨率 | 512/416 × 0.03 cm | ≈ 0.0369 cm/pixel |
| 投影角度数 | 640 | 均匀分布于 [0, 2π] |
| 探测器单元数 | 641 | 扇束探测器 |
| 源到中心距 (SOD) | 1075 × reso | 射线源到旋转中心 |
| 中心到探测器距 | 1075 × reso | 旋转中心到探测器 |
| FBP 滤波器 | Ram-Lak | 标准滤波反投影 |

```python
# 核心构建
ray_trafo  = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')  # 前向投影
FBPOper    = odl.tomo.fbp_op(ray_trafo, filter_type='Ram-Lak')               # FBP 重建
```

---

## 四、退化方法详解

### 总体流程

```
                          干净 CT 图像 (HU)
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
          组织分割         前向投影          金属掩码
        (水/骨分离)      (Radon变换)        (叠加)
                │               │               │
                ▼               ▼               ▼
          ┌─────────────────────────────────────────┐
          │         多能谱投影合成 (pkev2kvp)         │
          │   单能 keV → 多能 kVp（射束硬化建模）      │
          └─────────────────────┬───────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   泊松噪声 + 散射光子   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  射束硬化校正 (BHC)     │
                    │  三阶多项式拟合          │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  FBP 重建 → 退化 CT    │
                    └───────────────────────┘
```

---

### 4.1 组织分割（水/骨分离）

**目的：** 将 CT 图像分解为水（软组织）和骨两种基材料，以便分别进行多能谱投影。

```python
# HU → 线性衰减系数
img = imgCT / 1000 * MiuWater + MiuWater    # MiuWater = 0.192 cm⁻¹

# 阈值分割
threshWater = 100/1000 * 0.192 + 0.192      # ≈ 0.2112
threshBone  = 1500/1000 * 0.192 + 0.192     # ≈ 0.480

# 三区域处理
# 区域 1: img ≤ threshWater    → 纯水
# 区域 2: img ≥ threshBone     → 纯骨
# 区域 3: 中间区域              → 按线性比例分配水和骨
imgBone[中间]  = (img - threshWater) / (threshBone - threshWater) * img
imgWater[中间] = img - imgBone
```

---

### 4.2 多能谱投影合成（射束硬化建模）

**文件：** `util_func.py` → `pkev2kvp()`

**核心思想：** 真实 X 射线是多能谱的（如 120 kVp），不同能量下材料的衰减系数不同。单能模型无法捕捉这种差异，导致射束硬化伪影。

#### 物理模型

对于每种材料 $m$ 在能量 $E$ 下：

$$P_m(E) = \frac{\mu_m(E)}{\mu_m(E_{ref})} \cdot P_m(E_{ref})$$

其中 $E_{ref} = 70$ keV 为参考单能。

多能投影的合成：

$$P_{kvp} = -\ln\left(\frac{\sum_{E} S(E) \cdot \exp\left(-\sum_m P_m(E)\right)}{\sum_{E} S(E)}\right)$$

其中 $S(E)$ 为 X 射线能谱（GE 120 kVp 管）。

#### 材料数据库

| 材料 | 用途 | 密度 (g/cm³) |
|------|------|-------------|
| H₂O (水) | 软组织基材料 | 1.0 |
| 骨 (BONE_Cortical_ICRU44) | 骨基材料 | — |
| 钛 (Ti) | 金属植入物 | 4.5 |
| 铁 (Fe) | 金属植入物 | 7.8 |
| 铜 (Cu) | 金属植入物 | 8.9 |
| 金 (Au) | 金属植入物 | 2.0* |

*配置中默认使用钛 (`materialID = 0`)。

#### 关键参数

```python
kVp = 120               # 管电压
energies = [20, 121)     # 能量范围 20-120 keV
kev = 70                 # 参考单能
photonNum = 2 × 10⁷      # 入射光子数
```

---

### 4.3 泊松噪声与散射模拟

模拟真实探测器的量子噪声：

```python
scatterPhoton = 20                                     # 散射光子数

# 1. 计算到达探测器的光子数（无噪声）
temp = round(exp(-projkvp) * photonNum)

# 2. 加入散射光子
temp = temp + scatterPhoton

# 3. 泊松采样（量子噪声）
ProjPhoton = Poisson(temp)
ProjPhoton[ProjPhoton == 0] = 1                        # 避免 log(0)

# 4. 对数变换回投影域
projkvpNoise = -ln(ProjPhoton / photonNum)
```

**效果：** 在低光子计数区域（如金属遮挡处）噪声显著增大，这正是金属伪影的物理来源之一。

---

### 4.4 射束硬化校正（BHC）

**文件：** `util_func.py` → `marBHC()`

#### 水校正（预处理）

对非金属区域的投影进行三阶多项式水硬化校正：

```python
# 标定：生成不同厚度水的单能/多能投影对
thickness = [0, 0.05, 0.10, ..., 50.0] cm
p_kev = μ_water(70keV) × thickness                     # 单能投影
p_kvp = pkev2kvp(p_kev, spectrum, ...)                  # 多能投影

# 拟合三阶多项式：p_kev ≈ a₁·p_kvp + a₂·p_kvp² + a₃·p_kvp³
paraBHC = pinv([p_kvp, p_kvp², p_kvp³]) · p_kev
```

#### 金属 BHC（后处理）

```python
# 1. 线性插值（LI）：用非金属区域插值替换金属遮挡区域
LI_sinogram = interpolate_projection(ma_sinogram, metal_trace)

# 2. 计算差值
projDiff = ma_sinogram - LI_sinogram

# 3. 一阶硬化校正：用金属厚度的多项式拟合差值
#    projDiff ≈ c₁·P_metal + c₂·P_metal² + c₃·P_metal³
coeffs = lstsq(A, projDiff)

# 4. 校正投影
projBHC = ma_sinogram + correction_term
```

---

### 4.5 金属伪影合成

**文件：** `simulate_data.py` → `simulate_metal_artifact()`

对于每个金属掩码：

```python
# 1. 金属掩码 → 前向投影
Pmetal_kev = ray_trafo(imgMetal)
Pmetal_kev = metalAtten * Pmetal_kev               # 乘以金属衰减系数

# 2. 部分体积效应模拟
#    边缘像素衰减降低为 1/4（模拟金属边界的部分体积效应）
Pmetal_edge = XOR(Pmetal > 0, erode(Pmetal > 0))
Pmetal_kev[edge] = Pmetal_kev[edge] / 4

# 3. 三材料多能投影（水 + 骨 + 金属）
projkevAll[:,:,0] = Pwater     # 水投影
projkevAll[:,:,1] = Pbone      # 骨投影
projkevAll[:,:,2] = Pmetal     # 金属投影
projkvpMetal = pkev2kvp(projkevAll, ...)

# 4. 泊松噪声
# 5. BHC 校正
# 6. FBP 重建 → ma_CT（含金属伪影的 CT）
```

---

### 4.6 线性插值校正（LI）

**文件：** `util_func.py` → `interpolate_projection()`

最基本的金属伪影校正方法 — 用非金属区域的投影值线性插值替换金属遮挡区域：

```python
for each projection row:
    metal_positions     = where(metal_trace == 1)
    non_metal_positions = where(metal_trace == 0)
    proj[metal_positions] = interp1d(non_metal_positions, proj[non_metal_positions])(metal_positions)
```

---

## 五、数据生成流程

**入口：** `prepare_deep_lesion.py`

```
DeepLesion PNG 图像
        │
        ▼
┌──────────────────────────────────────┐
│ 1. 读取 PNG → HU 转换                │
│    image = raw * 2¹⁶ - 32768         │
│    resize → 416 × 416                │
│    clip: HU < -1000 → -1000          │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 2. 加载 100 个金属掩码               │
│    训练集: 90 个掩码                  │
│    测试集: 10 个掩码                  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 3. simulate_metal_artifact()         │
│    对每张 CT × 每个金属掩码生成：      │
│    - gt.h5     (干净参考)             │
│    - {idx}.h5  (退化数据)             │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 4. gene_h5list.py                    │
│    生成训练文件列表                    │
└──────────────────────────────────────┘
```

### 数据集划分

| 划分 | CT 图像数 | 金属掩码数 | 采样方式 |
|------|----------|-----------|---------|
| 训练集 | 1000 | 90 | `np.arange(0,1000)*40` |
| 测试集 | 200 | 10 | `np.arange(0,200)*10 + 44999` |

### 输出 HDF5 结构

**`gt.h5`（干净参考）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | float32 | 干净 CT（线性衰减系数） |
| `poly_sinogram` | float32 | 多能谱正弦图（无金属） |
| `poly_CT` | float32 | 多能谱重建 CT（含噪声和射束硬化） |

**`{idx}.h5`（退化数据，每个金属掩码一个文件）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `ma_CT` | float32 | 含金属伪影的 CT |
| `ma_sinogram` | float32 | 含金属的正弦图 |
| `LI_CT` | float32 | 线性插值校正后的 CT |
| `LI_sinogram` | float32 | 线性插值校正后的正弦图 |
| `BHC_CT` | float32 | 射束硬化校正后的 CT |
| `BHC_sinogram` | float32 | 射束硬化校正后的正弦图 |
| `metal_trace` | uint8 | 正弦图域金属轨迹掩码 |

---

## 六、退化方法总结

本项目实现的退化是 **基于物理模型的 CT 金属伪影合成**，包含以下退化因素：

| 退化因素 | 物理来源 | 实现方式 |
|---------|---------|---------|
| **射束硬化** | 多能 X 射线穿过不同材料后的非线性衰减 | `pkev2kvp()` 多能谱积分 |
| **量子噪声** | 探测器光子统计涨落 | `np.random.poisson()` 泊松采样 |
| **散射** | 光子与物质的散射效应 | 固定 20 个散射光子叠加 |
| **部分体积效应** | 金属边界像素的混合衰减 | 边缘像素衰减降低至 1/4 |
| **金属遮挡** | 高衰减金属导致的光子饥饿 | 金属投影叠加到正弦图 |

这些退化因素的叠加产生了典型的 CT 金属伪影：亮/暗条纹、杯状伪影、以及金属周围的阴影区域。

---

## 七、与 ctagent 项目的对比

| 特性 | MAR_SynCode | ctagent |
|------|-------------|---------|
| **退化类型** | 金属伪影（物理建模） | 噪声/模糊/下采样/条纹伪影 |
| **物理精度** | 高（多能谱、泊松噪声、BHC） | 简化（高斯噪声、高斯模糊） |
| **投影域** | 正弦图域操作 | 纯图像域操作 |
| **数据源** | DeepLesion PNG | DICOM |
| **输出格式** | HDF5 | NPY + JSONL Manifest |
| **应用场景** | 金属伪影去除（MAR） | 通用 CT 图像质量增强 |
