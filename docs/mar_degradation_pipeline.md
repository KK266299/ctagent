# MAR 退化数据生成流水线

> 金属伪影 (Metal Artifact Reduction) 物理仿真退化生成  
> 参考实现: [MAR_SynCode](https://github.com/KK266299/MAR_SynCode)  
> 本项目实现位置: `dataset/mar/`

---

## 1. 数据总览

### 1.1 数据规模


| 项目                          | 数量                                            |
| --------------------------- | --------------------------------------------- |
| CQ500 原始患者数                 | ~490                                          |
| 已处理患者数 (`cq500_processed/`) | 388                                           |
| 已处理切片总数                     | 22,509                                        |
| 每患者平均切片数                    | ~58 (stride=5 采样自 ~290 张)                     |
| 每切片生成文件                     | `gt.h5` (原图) + `0.h5`~`9.h5` (退化图, 最多 10 个掩模) |
| 总占用磁盘                       | 143 GB                                        |
| reads.csv 标签患者数             | 491                                           |
| 标签-处理交集 (可评测)               | 388 cases                                     |
| 任意阳性发现的患者数                  | 218 / 491 (44.4%)                             |


### 1.2 CQ500 标签分布


| 标签                       | 阳性例数 | 占比    |
| ------------------------ | ---- | ----- |
| ICH (颅内出血, 任意类型)         | 205  | 41.7% |
| IPH (脑实质出血)              | 134  | 27.3% |
| MassEffect (占位效应)        | 127  | 25.9% |
| MidlineShift (中线偏移)      | 65   | 13.2% |
| SAH (蛛网膜下腔出血)            | 60   | 12.2% |
| SDH (硬膜下出血)              | 53   | 10.8% |
| Fracture (骨折)            | 39   | 7.9%  |
| CalvarialFracture (颅骨骨折) | 34   | 6.9%  |
| IVH (脑室内出血)              | 28   | 5.7%  |
| EDH (硬膜外出血)              | 13   | 2.6%  |


> 标签来自 3 名放射科医生多数投票 (majority vote)。

### 1.3 BHX 病灶层标注


| 项目              | 数量     |
| --------------- | ------ |
| BHX 标注总切片数      | 15,979 |
| 标注 Study 数      | 226    |
| 标注 Series 数     | 389    |
| Bounding Box 总数 | 27,203 |


BHX 标注类型分布:


| 出血类型                   | 标注框数  |
| ---------------------- | ----- |
| Subdural (硬膜下)         | 7,904 |
| Subarachnoid (蛛网膜下)    | 7,596 |
| Intraparenchymal (脑实质) | 5,264 |
| Chronic (慢性)           | 3,672 |
| Intraventricular (脑室内) | 2,219 |
| Epidural (硬膜外)         | 548   |


---

## 2. 病灶感知切片选择策略

### 2.1 问题: 中间采样导致漏诊

之前的策略是从每个 series 的中间 50% 区间均匀抽取 5 张切片送入 VLM 诊断。这导致:

- 病灶可能位于头顶或颅底, 被完全遗漏
- **clean 图像的 recall 也极低** (如 SAH=0%, EDH=0%, Fracture=0%)
- 即使模型能力正常, 也无法从无病灶的切片中做出正确诊断

### 2.2 解决: BHX 标注驱动的优先选择

通过 BHX bounding-box 数据集建立 `SOPInstanceUID → 已处理切片` 映射, 实现三级选择策略:

```
策略 1: lesion_top_area
  标注切片数 ≥ max_n → 按病灶面积降序取 top-N

策略 2: lesion_priority
  标注切片数 < max_n → 全选标注切片 + 上下文相邻层补足

策略 3: middle_uniform (回退)
  无标注信息 → 中间区域均匀采样
```

### 2.3 评测逻辑正确性

- **有病灶才能诊断**: 正样本 case 通过 BHX 标注定位病灶层, 确保送给 VLM 的切片包含出血区域
- **负样本不受影响**: 无标注的 case (阴性) 使用 middle_uniform 采样, 合理
- **映射链**: `BHX CSV (SOPInstanceUID)` → `SOP Index (DICOM header)` → `processed folder (CT000xxx)`

---

## 3. 评价指标体系

### 3.1 多标签分类指标

每个标签独立计算二分类指标:


| 指标                   | 公式            | 说明                       |
| -------------------- | ------------- | ------------------------ |
| Accuracy             | (TP+TN) / N   | 整体正确率 (类别不平衡时有误导性)       |
| Precision            | TP / (TP+FP)  | 预测阳性中的真阳性比例              |
| Recall (Sensitivity) | TP / (TP+FN)  | 真阳性被检出的比例                |
| Specificity          | TN / (TN+FP)  | 真阴性被正确排除的比例              |
| F1 Score             | 2×P×R / (P+R) | Precision 和 Recall 的调和平均 |


### 3.2 汇总指标


| 指标            | 定义                | 说明                   |
| ------------- | ----------------- | -------------------- |
| Macro F1      | 各标签 F1 的算术平均      | 等权重, 低频标签影响大         |
| Micro F1      | 全局 TP/FP/FN 汇总后计算 | 高频标签主导               |
| Mean Accuracy | 各标签 Accuracy 的平均  | 整体概览                 |
| AUROC         | ROC 曲线下面积         | 需要置信度输出 (confidence) |


### 3.3 退化影响指标


| 指标                | 公式                                                                | 说明          |
| ----------------- | ----------------------------------------------------------------- | ----------- |
| Degraded Drop     | (clean_acc - degraded_acc) / clean_acc × 100%                     | 退化导致的性能下降幅度 |
| Restored Recovery | (restored_acc - degraded_acc) / (clean_acc - degraded_acc) × 100% | 修复恢复了多少性能   |


### 3.4 之前评测结果 (middle_uniform, 无 BHX)


| 指标            | Clean  | Degraded | 变化       |
| ------------- | ------ | -------- | -------- |
| Mean Accuracy | 89.29% | 85.04%   | -4.25pp  |
| Macro F1      | 0.2559 | 0.0562   | **-78%** |
| Micro F1      | 0.3836 | 0.0866   | **-77%** |


> **注意**: Accuracy 虚高是因为类别不平衡 (大量阴性标签), Macro F1 才是真实诊断能力的指标。  
> 退化后 Macro F1 从 0.26 暴跌到 0.06, 说明金属伪影严重干扰了 VLM 的诊断能力。  
> 但 clean 的 Macro F1 = 0.26 也偏低, 部分原因是 middle_uniform 策略遗漏了病灶层。

---

## 4. MAR 退化仿真方法

### 4.1 方法概述

本项目的退化生成采用**物理仿真方法** (physics-based simulation), 而非简单的图像域噪声叠加。核心思想是模拟真实 CT 成像过程中金属对 X 射线的物理影响:

1. 金属高密度材料导致 X 射线**严重衰减** (光子饥饿, photon starvation)
2. X 射线管发射的是**多色能谱**, 穿过金属后低能光子被优先吸收 → **射束硬化** (beam hardening)
3. 这些效应在投影域 (sinogram) 产生错误, 经 FBP 重建后呈现为**暗条纹伪影** (streak artifacts)

这种方法比简单的图像域退化更物理真实, 生成的 (GT, 退化图) 配对可直接用于训练 MAR 网络。

### 4.2 参考来源


| 来源                                                      | 说明                                         |
| ------------------------------------------------------- | ------------------------------------------ |
| [MAR_SynCode](https://github.com/KK266299/MAR_SynCode)  | 主要参考, MATLAB+Python 双版本 MAR 仿真             |
| ADN ([liaohaofu/adn](https://github.com/liaohaofu/adn)) | 底层方法论, `+helper/simulate_metal_artifact.m` |
| ODL ([odlgroup/odl](https://github.com/odlgroup/odl))   | 扇束几何、前向投影、FBP 重建                           |
| ASTRA Toolbox                                           | GPU 加速的 CT 投影算子后端                          |


### 4.3 七步物理仿真流水线

```
原始 CT (HU)
  │
  ├─ Step 1: HU → μ (线衰减系数)
  │
  ├─ Step 2: 组织分解 (水 + 骨)
  │
  ├─ Step 3: 前向投影 (Fan-beam, ASTRA-CUDA)
  │
  ├─ Step 4: 单能 → 多能转换 (pkev2kvp)
  │
  ├─ Step 5: 泊松噪声仿真
  │
  ├─ Step 6: 水基 BHC 三阶多项式校正
  │
  └─ Step 7: 金属伪影注入
       │
       ├─ 7a: 金属掩模前向投影
       ├─ 7b: 部分体积效应 (PVE)
       ├─ 7c: 三材料多色投影 (水+骨+金属)
       ├─ 7d: 泊松噪声 + BHC + FBP
       │
       └─ 输出: ma_CT (含金属伪影 CT)
```

---

## 5. 每一步详解与代码对应

### Step 1: HU → 线衰减系数 μ

**物理原理**:  
CT 的 Hounsfield Unit (HU) 与线衰减系数 μ 的关系为:

$$
HU = \frac{\mu - \mu_{water}}{\mu_{water}} \times 1000
$$

反解得:

$$
\mu = \frac{HU}{1000} \times \mu_{water} + \mu_{water}
$$

其中 $\mu_{water} = 0.192\ \text{cm}^{-1}$ (水在 70 keV 下的线衰减系数)。

**代码位置**: `dataset/mar/tissue_decompose.py` → `hu_to_mu()`

```python
def hu_to_mu(hu_image, mu_water=0.192):
    return hu_image / 1000.0 * mu_water + mu_water
```

**与 MAR_SynCode 对比**: 完全一致。MAR_SynCode 中为 `img = imgCT / 1000 * MiuWater + MiuWater`。

---

### Step 2: 组织分解 (水 + 骨)

**物理原理**:  
将 μ 图像分解为水和骨两种基材料, 这是简化的双材料分解模型:

- **纯水区** ($\mu \leq \mu_{thresh,water}$): 全部归水
- **纯骨区** ($\mu \geq \mu_{thresh,bone}$): 全部归骨
- **混合区**: 按线性插值分配

阈值计算:

- $\mu_{thresh,water} = (100/1000) \times 0.192 + 0.192 = 0.2112\ \text{cm}^{-1}$
- $\mu_{thresh,bone} = (1500/1000) \times 0.192 + 0.192 = 0.4800\ \text{cm}^{-1}$

**代码位置**: `dataset/mar/tissue_decompose.py` → `decompose_tissue()`

```python
def decompose_tissue(mu_image, thresh_water, thresh_bone):
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
```

**与 MAR_SynCode 对比**: 完全一致, 阈值分解逻辑和线性插值公式相同。

---

### Step 3: 前向投影 (Fan-beam)

**物理原理**:  
模拟 X 射线穿过人体的过程 — 沿每条射线积分衰减系数, 得到正弦图 (sinogram)。使用扇束 (fan-beam) 几何:


| 参数     | 值                                 |
| ------ | --------------------------------- |
| 图像尺寸   | 416 × 416                         |
| 投影视角数  | 640                               |
| 探测器单元数 | 641                               |
| 源到中心距  | 1075 pixels × 0.0369 cm ≈ 39.7 cm |
| 像素尺寸   | 512/416 × 0.03 ≈ 0.0369 cm        |


**代码位置**: `dataset/mar/ct_geometry.py` → `CTGeometry`

```python
class CTGeometry:
    def __init__(self, config):
        # ODL Fan-beam 几何构建
        reso = cfg.orig_pixels / cfg.image_size * cfg.orig_pixel_size_cm
        geometry = odl_tomo.FanBeamGeometry(angle_partition, detector_partition,
                                            src_radius=src_radius, det_radius=det_radius)
        self.ray_trafo = odl_tomo.RayTransform(self.reco_space, geometry, impl="astra_cuda")
        self.fbp_op = odl_tomo.fbp_op(self.ray_trafo, filter_type="Ram-Lak")

    def forward(self, image):
        return np.asarray(self.ray_trafo(image))

    def fbp(self, sinogram):
        return np.asarray(self.fbp_op(sinogram))
```

**与 MAR_SynCode 对比**: 几何参数完全一致。后端同样使用 ODL + ASTRA-CUDA。

---

### Step 4: 单能 → 多能转换 (pkev2kvp)

**物理原理**:  
这是整个仿真中最关键的步骤。真实 X 射线管发射多色能谱, 不同能量下材料的衰减系数不同。多色投影的 Beer-Lambert 定律:

$$
p_{kVp} = -\ln \frac{\sum_{E} S(E) \cdot \exp\left(-\sum_{m} \frac{\mu_m(E)}{\mu_m(E_0)} \cdot p_m(E_0)\right)}{\sum_{E} S(E)}
$$

其中:

- $S(E)$: GE14 型 120 kVp 管球 X 射线能谱 (20~120 keV)
- $\mu_m(E)$: 第 $m$ 种材料在能量 $E$ 下的线衰减系数 (来自 NIST 数据表)
- $E_0 = 70$ keV: 参考单能
- $p_m(E_0)$: 第 $m$ 种材料在参考能量下的线投影

**代码位置**: `dataset/mar/energy_convert.py` → `pkev2kvp()`

```python
def pkev2kvp(proj_kev_all, spectrum, energies, kev, mu_all):
    for ien in energies:          # 遍历 20~120 keV
        proj_total = 0
        for imat in range(num_materials):
            ratio = mu_all[imat][ien-1, 6] / mu_all[imat][kev-1, 6]
            proj_total += ratio * proj_kev_all[:, :, imat]
        proj_energy += spectrum[ien-1] * np.exp(-proj_total)
        spectrum_sum += spectrum[ien-1]
    proj_kvp = -np.log(proj_energy / spectrum_sum + 1e-30)
    return proj_kvp
```

**与 MAR_SynCode 对比**: 完全一致。衰减系数列索引 `ATTEN_MODE_COL = 6` 对应 NIST 表中"total attenuation with coherent scattering"列。

---

### Step 5: 泊松噪声仿真

**物理原理**:  
X 射线探测器计数光子, 遵循泊松分布。加入散射光子 (常数近似):

$$
I_{expected} = \text{round}(e^{-p_{kVp}} \times N_{photon}) + N_{scatter}
$$
$$
I_{actual} \sim \text{Poisson}(I_{expected})
$$
$$
p_{noisy} = -\ln(I_{actual} / N_{photon})
$$

参数: $N_{photon} = 2 \times 10^7$, $N_{scatter} = 20$。

**代码位置**: `dataset/mar/energy_convert.py` → `add_poisson_noise()`

```python
def add_poisson_noise(proj_kvp, photon_num=2e7, scatter_photon=20, rng=None):
    expected = np.round(np.exp(-proj_kvp) * photon_num) + scatter_photon
    expected = np.maximum(expected, 1.0)
    actual = rng.poisson(expected)
    actual = np.maximum(actual, 1.0)
    proj_noisy = -np.log(actual / photon_num)
    return proj_noisy
```

**与 MAR_SynCode 对比**: 完全一致。噪声加在投影域 (log-transform 之前) 是物理正确的。

---

### Step 6: 水基 BHC (Beam Hardening Correction)

**物理原理**:  
射束硬化导致多色投影与单能投影之间的非线性偏差。水基 BHC 通过三阶多项式建立从多色投影到水等效单能投影的映射:

$$
p_{corrected} = c_1 \cdot p + c_2 \cdot p^2 + c_3 \cdot p^3
$$

系数 $(c_1, c_2, c_3)$ 预先通过对 0~50 cm 水厚度范围内的精确正/反向映射进行最小二乘拟合得到。

**代码位置**: 

- 系数计算: `dataset/mar/physics_params.py` → `PhysicsParams._compute_bhc_coefficients()`
- 应用校正: `dataset/mar/sinogram_utils.py` → `apply_bhc()`

```python
def apply_bhc(sinogram, para_bhc):
    p = sinogram.ravel().reshape(-1, 1)
    A = np.concatenate([p, p**2, p**3], axis=1)
    corrected = (A @ para_bhc).reshape(sinogram.shape)
    return corrected
```

**与 MAR_SynCode 对比**: 完全一致, 包括多项式阶数、拟合范围和应用方式。

---

### Step 7: 金属伪影注入

这是最复杂的一步, 分为多个子步骤:

#### 7a: 金属掩模前向投影

将二值金属掩模 (来自 `SampleMasks.mat`, 共 10 个掩模) resize 到 416×416, 前向投影得到金属在 sinogram 中的投影轨迹 (metal trace)。

#### 7b: 部分体积效应 (PVE)

金属边缘像素只部分占据体素, 通过形态学腐蚀检测边缘, 对边缘像素的投影值乘以 0.25:

```python
def apply_partial_volume_effect(proj_metal, edge_fraction=0.25):
    metal_eroded = binary_erosion(metal_bw, structure=np.ones((1, 3)))
    metal_edge = np.logical_xor(metal_bw, metal_eroded)
    result[metal_edge] *= edge_fraction
    return result
```

#### 7c: 三材料多色投影

将水、骨、金属三种材料的投影送入 `pkev2kvp` 计算含金属的多色投影:

```python
proj_kev_3mat = np.stack([p_water_kev, p_bone_kev, p_metal_kev_pve], axis=-1)
mu_list_3 = [mu_water, mu_bone, mu_metal_single]
proj_kvp_metal = pkev2kvp(proj_kev_3mat, spectrum, energies, kev, mu_list_3)
```

金属衰减系数: $\mu_{metal} = \rho_{metal} \times \mu_{mass}(E_0)$


| 金属     | 密度 (g/cm³) | material_id |
| ------ | ---------- | ----------- |
| Ti (钛) | 4.5        | 0 (默认)      |
| Fe (铁) | 7.8        | 1           |
| Cu (铜) | 8.9        | 2           |
| Au (金) | 2.0        | 3           |


> 当前配置使用 Ti (钛), 常见于牙科和骨科植入物。

#### 7d: 含金属重建

泊松噪声 → BHC → FBP 重建, 得到最终的 `ma_CT` (含金属伪影 CT)。

**与 MAR_SynCode 对比**: 完全一致, 包括金属衰减计算、PVE 处理和三材料投影逻辑。

---

## 6. 退化效果验证

### 6.1 4-Panel 对比图分析

参考 `try_output/CQ500CT0_PLAIN_THIN_CT000010_4panel.png`:


| 面板                     | 内容       | 观察                     |
| ---------------------- | -------- | ---------------------- |
| GT (clean)             | 原始无金属 CT | 脑组织、颅骨清晰, 无伪影          |
| ma_CT (artifact)       | 含金属伪影    | 可见两个高亮金属植入物 + 放射状暗条纹伪影 |
| LI_CT (LI corrected)   | 线性插值校正   | 条纹减弱但金属周围模糊            |
| BHC_CT (BHC corrected) | BHC 校正   | 部分条纹抑制, 残余伪影           |


### 6.2 伪影特征一致性

观察到的退化特征与物理预期完全一致:

1. **金属高亮**: 金属区域呈现极高亮度 (高衰减系数) ✓
2. **暗条纹伪影**: 从金属位置沿射线方向辐射的暗条带 ✓ (光子饥饿效应)
3. **亮条纹伪影**: 金属之间和周围的亮带 ✓ (射束硬化效应)
4. **伪影范围**: 伪影在金属连线方向最显著 ✓
5. **相对温和**: 因为使用钛 (Ti, 密度 4.5), 比铁/铜更低衰减, 加上高光子计数 (2×10⁷) ✓

### 6.3 代码正确性确认


| 检查项           | 状态  | 说明                                  |
| ------------- | --- | ----------------------------------- |
| HU→μ 公式       | ✓   | 与 MAR_SynCode 和 NIST 标准一致           |
| 组织分解阈值        | ✓   | water=100HU, bone=1500HU, 与参考一致     |
| 前向投影几何        | ✓   | 640 views × 641 detectors, fan-beam |
| pkev2kvp 能量积分 | ✓   | 遍历 20~120 keV, Beer-Lambert 公式正确    |
| 泊松噪声位置        | ✓   | 在投影域 (log 之前) 加噪, 物理正确              |
| BHC 多项式       | ✓   | 三阶拟合, 水厚度 0~50 cm                   |
| 金属衰减计算        | ✓   | 密度 × μ_mass, 查 NIST 表               |
| PVE 处理        | ✓   | 边缘腐蚀 + 0.25 系数                      |
| 输出格式          | ✓   | HDF5 gzip 压缩, gt.h5 + {idx}.h5      |


---

## 7. 代码结构说明

### 7.1 核心模块

```
dataset/mar/
├── mar_simulator.py         ← 主流水线: 7 步仿真 + HDF5 保存
├── ct_geometry.py           ← ODL 扇束几何、前向投影、FBP
├── physics_params.py        ← 物理参数加载 (.mat 文件)、BHC 系数
├── tissue_decompose.py      ← HU→μ 转换、水/骨分解
├── energy_convert.py        ← pkev2kvp 多能转换、泊松噪声
├── sinogram_utils.py        ← BHC 校正、线性插值、MAR-BHC
├── cq500_reader.py          ← CQ500 DICOM 扫描与读取
└── __init__.py
```

### 7.2 配置文件

```
configs/data/
└── mar_simulation_cq500_full.yaml  ← CQ500 全量处理配置
```

关键配置参数:

```yaml
input:
  source_type: "cq500"
  ct_image_dir: "/home/liuxinyao/data/cq500"

cq500:
  series_keywords: ["THIN", "Plain"]   # 只处理薄层和平扫序列
  min_slices: 20                        # 每序列最少切片数
  slice_stride: 5                       # 每 5 层取 1 层

output:
  data_dir: "/home/liuxinyao/data/cq500_processed"
  minimal: true                         # 只保存 GT + ma_CT

geometry:
  image_size: 416
  num_angles: 640
  num_detectors: 641
  impl: "astra_cuda"                    # GPU 加速

physics:
  kVp: 120
  kev: 70
  photon_num: 20000000.0
  material_id: 0                        # 钛 (Ti)
```

### 7.3 处理脚本

```
scripts/
├── build_mar_dataset.py     ← CLI 入口, 支持并行 GPU 处理 (--workers)
├── build_sop_index.py       ← 构建 SOPInstanceUID 映射索引
└── run_cq500_api_eval.py    ← API 评测入口
```

### 7.4 评测模块

```
eval/
├── bhx_loader.py            ← 加载 BHX bounding-box 标注
├── cq500_manifest.py        ← 病灶感知 manifest 构建 + SOPIndex
├── cq500_labels.py          ← reads.csv GT 标签加载
├── cq500_api_eval.py        ← VLM API 批量评测
└── metrics.py               ← 多标签分类指标计算
```

---

## 8. 处理命令参考

```bash
# 1. MAR 退化数据生成 (已完成, 8 workers GPU 并行)
PYTHONPATH=. python scripts/build_mar_dataset.py \
    --config configs/data/mar_simulation_cq500_full.yaml \
    --workers 8

# 2. 构建 SOP 索引 (一次性, ~3 分钟)
PYTHONPATH=. python scripts/build_sop_index.py \
    --cq500-root /home/liuxinyao/data/cq500 \
    --processed-root /home/liuxinyao/data/cq500_processed \
    --output /home/liuxinyao/data/cq500_sop_index.json

# 3. API 评测 (使用 BHX 病灶感知选择)
source ~/network.sh
PYTHONPATH=. python scripts/run_cq500_api_eval.py \
    --config configs/experiment/cq500_api_eval.yaml \
    --model openai/gpt-4o \
    --max-cases 5
```

---

## 9. 已知局限与改进方向

### 仿真局限


| 局限     | 说明             | 可能改进             |
| ------ | -------------- | ---------------- |
| 两材料分解  | 只区分水和骨, 不含脂肪/肺 | 引入更多基材料          |
| 散射模型简化 | 常数 20 光子近似     | Monte Carlo 散射仿真 |
| 2D 仿真  | 扇束, 不支持 3D 锥束  | 升级到 3D cone-beam |
| 单一金属类型 | 每次只用一种金属       | 混合金属支持           |
| 掩模来源固定 | 10 个预设掩模循环使用   | 随机生成或临床提取        |


### 评测改进


| 改进方向                                | 状态           |
| ----------------------------------- | ------------ |
| BHX 病灶感知切片选择                        | ✓ 已实现        |
| SOPInstanceUID 映射索引                 | ✓ 已实现        |
| 多窗位评测 (骨窗/脑窗)                       | 待实现          |
| Restored 图像评测                       | 待 MAR 网络训练完成 |
| 多模型对比 (GPT-4o vs Qwen VL vs Gemini) | 待实现          |


