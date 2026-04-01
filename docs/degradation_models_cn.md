# CT 退化模型完整文档

> 本文档记录 `ctagent` 项目中已实现的 **10 类 CT 退化仿真模型**，包含物理原理、操作方法、参数配置、合理性分析及可视化结果。

---

## 总览

| 编号 | 类型 | 注入域 | 可修复性 |
|------|------|--------|----------|
| 1 | 金属伪影 (Metal Artifact) | 三材料多能流水线 | 中等 |
| 2 | 环状伪影 (Ring Artifact) | BHC 后 sinogram | **高** |
| 3 | 运动伪影 (Motion Artifact) | BHC 后 sinogram | 中等 |
| 4 | 束硬化伪影 (Beam Hardening) | BHC 前 (proj_kvp_noise) | **高** |
| 5 | 散射伪影 (Scatter Artifact) | BHC 前透射域 | 中高 |
| 6 | 截断伪影 (Truncation) | BHC 后 sinogram | 中等 |
| 7 | 低剂量噪声 (Low-Dose Noise) | 干净投影 (proj_kvp) | **高** |
| 8 | 稀疏角采样 (Sparse-View) | BHC 后 sinogram | **低** |
| 9 | 有限角采样 (Limited-Angle) | BHC 后 sinogram | **低** |
| 10 | 焦点模糊 (Focal Spot Blur) | BHC 后 sinogram | 中等 |

---

## 1. 通用基线流水线 (Base Pipeline)

所有退化模型共享统一的物理仿真基线 `_prepare_base_state`，与临床 CT 扫描物理过程对齐：

```
输入: HU 图像
  Step 1: HU -> mu (线衰减系数)       mu = HU/1000 * mu_water + mu_water  (mu_water=0.192)
  Step 2: 组织分解                     img_water + img_bone (阈值线性插值)
  Step 3: 前向投影 (Radon 变换)        ASTRA-CUDA fan-beam, 416x416, 640角, 641探测器
  Step 4: 单能->多能转换 (pkev2kvp)    Beer-Lambert + X射线能谱加权
          => proj_kvp (干净多色投影)                          [保留]
  Step 5: 泊松噪声 + 高斯电子噪声
          => proj_kvp_noise (含噪 BHC 前投影)                [保留]
  Step 6: 水基 BHC                     BHC(p) = c1*p + c2*p^2 + c3*p^3
          => poly_sinogram -> poly_ct = FBP(poly_sinogram)   [GT 参考]
```

**关键中间态及各退化的使用关系：**

| 变量 | 含义 | 使用者 |
|------|------|--------|
| `proj_kvp` | 干净多色投影（无噪声） | 低剂量噪声 |
| `proj_kvp_noise` | 含噪多色投影（BHC 前） | 束硬化、散射 |
| `poly_sinogram` | BHC 后正弦图 | 环状、运动、截断、稀疏角、有限角、焦点模糊 |
| `poly_ct` | 干净 FBP 重建 | 所有退化（作为 GT） |

**合理性：** 多能谱建模（30-120 keV, ~90 能量点）、泊松光子计数（I0=2e7）、高斯电子噪声（sigma_e=10）、三阶多项式 BHC（物理拟合系数）。

---

## 2. 金属伪影 (Metal Artifact)

**源码：** `dataset/mar/mar_simulator.py` — `MARSimulator`（7 步流水线）

### 物理原理
金属植入物（假牙、骨钉、人工关节）对 X 射线有极高衰减系数：
- **光子饥饿**：穿过金属路径的探测器计数接近零
- **BHC 残留**：高原子序数材料导致严重多色效应，水基 BHC 无法充分校正
- **条纹伪影**：重建后产生典型放射状明暗条纹

### 操作方法
```
Steps 1-6: 与基线相同（无金属）
Step 7: 逐掩模金属伪影注入
  7a: 缩放金属二值掩模 -> 前向投影 -> metal_trace
  7b: 部分体积效应（PVE）：边缘像素衰减 *= 0.25
  7c: 三材料多能投影：[水, 骨, 金属] -> pkev2kvp
      金属可选：Ti（钛）, 不锈钢, CoCr（钴铬）, Au（金）
  7d: 泊松噪声 -> BHC -> FBP 重建 -> ma_CT
  7e: 校正结果：LI_CT（线性插值）, BHC_CT（MAR-BHC）
```

### 参数
- 金属掩模：`SampleMasks.mat`（100 个掩模，各种形状/大小）
- 金属材料：Ti / 不锈钢 / CoCr / Au（由 `material_id` 选择）
- I0 = 2e7，PVE edge_fraction = 0.25

### 合理性评估：**强** ✅
最物理真实的退化模型。完整三材料多能仿真 + 部分体积效应 + 光子饥饿。参考：ADN（MICCAI 2019）。

---

## 3. 环状伪影 (Ring Artifact)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `RingArtifactSimulator`

### 物理原理
探测器单元的**增益漂移**或**恒定偏移**导致 sinogram 中某些列系统性偏差。FBP 重建后表现为以旋转中心为圆心的**同心圆环**。

### 操作方法（BHC 后 sinogram）
```
1. 随机选择 n_bad 个探测器列（占比 0.5%-3.0%）
2. 对每个坏列施加：
   a. 乘性增益扰动：  col *= gain       （gain 范围 [0.85, 1.15]）
   b. 加性偏移：      col += bias       （正比于 sinogram 标准差）
   c. 时变漂移：      col += A*sin(2*pi*t)*cos(phase)
3. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| bad_detector_fraction | 0.5%-1.0% | 1.0%-2.0% | 2.0%-3.0% |
| gain_range | [0.95, 1.05] | [0.90, 1.10] | [0.85, 1.15] |
| additive_bias_scale | 0.003 | 0.006 | 0.010 |
| drift_strength | 0.002 | 0.004 | 0.008 |

### 合理性评估：**强** ✅
结合乘性 + 加性 + 时变三种探测器异常模式，覆盖真实探测器主要失效模式。参考：Sarepy, Muench et al. 2009。

---

## 4. 运动伪影 (Motion Artifact)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `MotionArtifactSimulator`

### 物理原理
患者在扫描期间发生**刚体平移**，导致部分角度的投影数据对应不同空间位置：
- **鬼影**（重复边缘）、**模糊**（细节丢失）、**条纹**（高对比度界面附近）

### 操作方法（BHC 后 sinogram）
```
1. 选取连续角度范围 [start, end]（占总角度 8%-30%）
2. 在该范围内逐行做沿探测器方向的渐进亚像素平移：
   shift = signed * shift_max * linear_ramp(t)
3. 使用 np.interp 进行亚像素精确插值
4. 鬼影混合：(1-blend)*shifted + blend*original
5. 沿 view 方向高斯平滑（sigma=0.8）消除跳变
6. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| motion_fraction | 8%-12% | 12%-20% | 20%-30% |
| translation_mm | 1-5 mm | 5-10 mm | 10-15 mm |
| ghost_blend | 0.15 | 0.25 | 0.35 |

### 合理性评估：**良好** ✅
渐进式平移 + 鬼影混合模拟了真实的渐进运动效应。参考：TorchIO RandomMotion。

---

## 5. 束硬化伪影 (Beam Hardening)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `BeamHardeningArtifactSimulator`

### 物理原理
多色 X 射线穿过物体时**低能光子优先被吸收**，导致射束"变硬"。标准水基 BHC 若**校正不完美**，会残留：
- **Cupping 效应**：图像中心偏暗（穿过更多物质 = 硬化更严重）
- **条带伪影**：高密度骨结构之间出现暗带

### 操作方法（BHC 前：proj_kvp_noise）
```
1. 取 proj_kvp_noise（含噪多色投影，未经 BHC 校正）
2. 扰动 BHC 多项式系数：
   perturbed_bhc = para_bhc * bhc_scale    （bhc_scale < 1 = 欠校正）
3. 用扰动系数做 BHC：
   result = (c1*s)*p + (c2*s)*p^2 + (c3*s)*p^3
4. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| bhc_scale | 0.85-0.95 | 0.65-0.85 | 0.40-0.65 |

bhc_scale < 1 表示**欠校正**（undercorrection），这是临床最常见的束硬化残留类型。

### 合理性评估：**强** ✅
在 BHC 前域（proj_kvp_noise）上操作——**物理正确**。通过扰动 BHC 系数是模拟不完美校正的最精确方法。之前版本在 BHC 后叠加非线性项导致"反 cupping"（物理方向错误），已修正。参考：Hsieh 2004, MAR_SynCode guide。

---

## 6. 散射伪影 (Scatter Artifact)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `ScatterArtifactSimulator`

### 物理原理
康普顿散射使散射光子到达探测器的非对应位置，产生**低频平滑背景信号**：
- **对比度下降**、**图像雾化**、**类 cupping 效应**

### 操作方法（BHC 前：proj_kvp_noise，透射域）
```
1. 取 proj_kvp_noise（含噪多色投影，BHC 前）
2. 转透射域：primary = exp(-proj_kvp_noise)
3. 生成散射信号：scatter = scatter_ratio * GaussianBlur(primary, sigma)
4. 合并测量信号：measured = primary + scatter  （强度域加性！）
5. 转回投影域：contaminated = -log(measured)
6. 做 BHC：result = BHC(contaminated, para_bhc)
7. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| scatter_ratio | 1%-3% | 3%-6% | 6%-10% |
| blur_sigma_mm | 30-45 mm | 45-60 mm | 60-80 mm |

### 合理性评估：**强** ✅
在 BHC 前透射域操作——**物理正确**（散射在强度域是加性的）。低频高斯核是散射的标准一阶近似。参考：Siewerdsen & Jaffray 2001, DeepDRR。

---

## 7. 截断伪影 (Truncation Artifact)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `TruncationArtifactSimulator`

### 物理原理
扫描对象**超出扫描仪视野（FOV）**时，sinogram 两侧投影数据缺失或衰减：
- **边缘亮环/亮带**、**外围 HU 值偏移**

### 操作方法（BHC 后 sinogram）
```
1. 计算截断宽度：width = bins * truncate_ratio
2. 在 sinogram 两侧施加余弦渐变窗：
   ramp = min_frac + (1 - min_frac) * 0.5 * (1 - cos(pi*t))
   边缘衰减到 min_fraction（不归零！）
3. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| truncate_ratio | 5%-10% | 10%-20% | 20%-30% |
| min_fraction | 0.40 | 0.20 | 0.05 |
| fill_mode | cosine | cosine | cosine |

### 合理性评估：**良好** ✅
余弦渐变 + min_fraction 避免直接置零导致的 Gibbs 振铃。参考：Hsieh 2004。

---

## 8. 低剂量噪声 (Low-Dose Noise)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `LowDoseNoiseSimulator`

### 物理原理
降低管电流（mA）或曝光时间 -> 光子数减少 -> 泊松噪声方差增大 -> **颗粒状噪声**。这是 LDCT（低剂量 CT）去噪研究的核心场景。

### 操作方法（干净投影：proj_kvp，无噪声）
```
1. 取 proj_kvp（干净多色投影，零噪声）
2. 计算低剂量光子数：I0_low = I0 * dose_fraction
3. 重新采样泊松噪声：
   noisy = Poisson(exp(-proj_kvp) * I0_low + scatter) / I0_low
4. 加高斯电子噪声：noisy += N(0, sigma_e / I0_low)
5. 做 BHC：result = BHC(noisy, para_bhc)
6. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| dose_fraction | 20%-50% | 5%-20% | 1%-5% |
| electronic_sigma | 5.0 | 10.0 | 20.0 |

### 合理性评估：**强** ✅
使用 proj_kvp（干净、无噪声）避免双重噪声，从头重新采样泊松过程——**物理正确**。参考：Mayo LDCT simulation, XCIST。

---

## 9. 稀疏角采样 (Sparse-View)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `SparseViewArtifactSimulator`

### 物理原理
减少投影角度数（如 640 降到 20-120），违反 Nyquist-Shannon 采样定理：
- **视角混叠条纹**、**细节丢失**

### 操作方法（BHC 后 sinogram）
```
1. 从 640 个角度中均匀选取 num_views 个
2. 保留选中行，缺失行用相邻角度线性插值填充
3. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| num_views | 90-120 | 45-90 | 20-45 |
| 采样比 | ~15%-19% | ~7%-14% | ~3%-7% |
| 插值方式 | 线性 | 线性 | 线性 |

### 合理性评估：**良好** ✅
标准均匀角度欠采样场景。**局限：信息缺失型退化**，FBP 无法恢复缺失角度信息，深度学习可改善观感但无法恢复真值。参考：TAMP。

---

## 10. 有限角采样 (Limited-Angle)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `LimitedAngleArtifactSimulator`

### 物理原理
CT 旋转只覆盖部分角度范围（如 60-160 度而非 360 度）：
- **Radon 变换不完整**（数学上病态），**方向性伪影**（沿缺失角度方向模糊/条纹）

### 操作方法（BHC 后 sinogram）
```
1. 随机选择起始角度，保留 [start, end] 范围内的行，其余置零
2. 边界施加余弦过渡窗（10% 宽度）减少 Gibbs 效应
3. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| angle_range_deg | 140-160 度 | 100-140 度 | 60-100 度 |
| transition_fraction | 10% | 10% | 10% |

### 合理性评估：**良好** ✅
**局限：最病态的逆问题之一**，信息根本性缺失。参考：EPNet（MICCAI 2021）。

---

## 11. 焦点模糊 (Focal Spot Blur)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `FocalSpotBlurSimulator`

### 物理原理
X 射线焦点非理想点源（通常 0.3-1.2 mm），加上探测器光学串扰：
- **空间分辨率下降**、**边缘模糊**、**微小结构不可分辨**

### 操作方法（BHC 后 sinogram）
```
1. 沿探测器方向（axis=1）高斯模糊：sigma_bins（焦点大小 + 探测器宽度）
2. 可选：沿角度方向（axis=0）轻微模糊：sigma_views（旋转采样间隔效应）
3. FBP 重建
```

### 参数配置

| 参数 | mild | moderate | severe |
|------|------|----------|--------|
| blur_sigma_bins | 0.5-1.0 | 1.0-2.0 | 2.0-4.0 |
| axial_sigma_views | 0.0-0.3 | 0.3-0.8 | 0.8-1.5 |

### 合理性评估：**良好** ✅
Sinogram 域高斯 PSF 是标准焦点模型，双方向控制更灵活。参考：XCIST。

---

## 12. 混合退化 (Composite)

**源码：** `dataset/mar/ct_artifact_simulator.py` — `CompositeArtifactSimulator`

支持将多种退化按特定顺序叠加：

```python
# 指定组合
composite.simulate_composed(hu_image, recipe=[
    (ring_sim, "mild"),
    (scatter_sim, "moderate"),
])

# 随机组合
composite.simulate_random(hu_image, num_artifacts=2, severity="moderate")
```

### 推荐物理顺序

```
proj_kvp（干净投影）
  -> [低剂量噪声]         在 proj_kvp 上重新采样光子噪声
  -> proj_kvp_noise
  -> [散射]               在 proj_kvp_noise 上透射域叠加
  -> [束硬化]             在 proj_kvp_noise 上扰动 BHC 系数
  -> BHC -> poly_sinogram
  -> [环状伪影]           探测器增益异常（BHC 后）
  -> [运动伪影]           几何偏移（BHC 后）
  -> [截断伪影]           FOV 边缘损失（BHC 后）
  -> [稀疏角/有限角]      采样限制（BHC 后）
  -> [焦点模糊]           系统 PSF（BHC 后）
  -> FBP 重建
```

---

## 13. 可视化结果

### 脑窗 (WL=40, WW=80)

![10 类退化全景 - 脑窗](../try_output/degradation_review/CQ500CT0_PLAIN_THIN_CT000120_all_10_degradations_brain.png)

### 硬膜下窗 (WL=75, WW=215)

![10 类退化全景 - 硬膜下窗](../try_output/degradation_review/CQ500CT0_PLAIN_THIN_CT000120_all_10_degradations_subdural.png)

### 差异热图 |退化 - GT|

![差异热图](../try_output/degradation_review/CQ500CT0_PLAIN_THIN_CT000120_diff_maps_10types.png)

### 图片行说明（从上到下）

| 行 | 退化类型 | 第 0 列 | 第 1 列 | 第 2 列 | 第 3 列 |
|----|----------|---------|---------|---------|---------|
| 1 | 金属伪影 | GT | mask #0 | mask #1 | mask #2 |
| 2 | 环状伪影 | GT | mild | moderate | severe |
| 3 | 运动伪影 | GT | mild | moderate | severe |
| 4 | 束硬化伪影 | GT | mild | moderate | severe |
| 5 | 散射伪影 | GT | mild | moderate | severe |
| 6 | 截断伪影 | GT | mild | moderate | severe |
| 7 | 低剂量噪声 | GT | mild | moderate | severe |
| 8 | 稀疏角采样 | GT | mild | moderate | severe |
| 9 | 有限角采样 | GT | mild | moderate | severe |
| 10 | 焦点模糊 | GT | mild | moderate | severe |

每张退化图标注 PSNR (dB) 和 SSIM 指标，值越低表示退化越严重。

### 图片路径

```
try_output/degradation_review/
  CQ500CT0_..._all_10_degradations_brain.png       # 脑窗 10 类退化
  CQ500CT0_..._all_10_degradations_subdural.png     # 硬膜下窗 10 类退化
  CQ500CT0_..._diff_maps_10types.png                # 差异热图
```

---

## 14. 可修复性评估

| 退化类型 | 可修复性 | 说明 |
|----------|----------|------|
| 金属伪影 | 中等 | LI/MAR-BHC 可减轻条纹；DL-MAR 可大幅改善；密集金属仍有残留 |
| 环状伪影 | **高** | 极坐标中值滤波 / 小波分解可有效去除 |
| 运动伪影 | 中等 | TV 正则化 / Wiener 反卷积可改善；严重运动难以完全恢复 |
| 束硬化伪影 | **高** | 多项式 BHC / 迭代校正 / 双能 CT 可基本消除 |
| 散射伪影 | 中高 | 散射核估计 + 去趋势 / CLAHE 效果显著 |
| 截断伪影 | 中等 | 边界外推 + TV 平滑可改善；严重截断有信息丢失 |
| 低剂量噪声 | **高** | BM3D / DnCNN / 迭代重建效果非常好 |
| 稀疏角采样 | **低** | 信息缺失型：深度学习可改善观感但无法恢复真值 |
| 有限角采样 | **低** | 最病态的逆问题之一，信息根本性缺失 |
| 焦点模糊 | 中等 | Richardson-Lucy / Wiener 反卷积可部分恢复 |

---

## 15. 运行命令

### 生成 10 类退化全景可视化

```bash
PYTHONPATH=. bash -c 'eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/visualize_all_degradations.py \
    --output-dir try_output/degradation_review --gpu 1 \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/viz_degrad_$(date +%Y%m%d_%H%M%S).log'
```

### Rule-Based 修复评估

```bash
PYTHONPATH=. bash -c 'eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_ct_artifact_restoration.py \
    --planner rule --num-slices -1 \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/restoration_rule_$(date +%Y%m%d_%H%M%S).log'
```

### LLM 引导修复评估

```bash
PYTHONPATH=. bash -c 'eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_ct_artifact_restoration.py \
    --planner llm --llm-model qwen/qwen-2.5-vl-72b-instruct \
    --llm-base-url https://openrouter.ai/api/v1 --num-slices 5 \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/restoration_llm_$(date +%Y%m%d_%H%M%S).log'
```

---

## 附录：代码文件索引

| 文件 | 职责 |
|------|------|
| `dataset/mar/ct_artifact_simulator.py` | 9 类退化仿真器（环状/运动/束硬化/散射/截断/稀疏角/有限角/低剂量/焦点模糊） |
| `dataset/mar/mar_simulator.py` | 金属伪影仿真器（7 步完整物理流水线） |
| `dataset/mar/energy_convert.py` | 多能谱转换 `pkev2kvp` + 泊松噪声 `add_poisson_noise` |
| `dataset/mar/sinogram_utils.py` | BHC 多项式、线性插值、金属轨迹、PVE、MAR-BHC |
| `dataset/mar/tissue_decompose.py` | HU -> mu 转换、水/骨组织分解 |
| `dataset/mar/ct_geometry.py` | CT 几何配置 + ASTRA-CUDA 前向投影/FBP |
| `dataset/mar/physics_params.py` | 物理参数加载（能谱、衰减系数、BHC 系数） |
| `scripts/visualize_all_degradations.py` | 10 类退化全景可视化脚本 |
