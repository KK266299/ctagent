# CT 合成退化方法审查报告

> 参考基准: `/home/liuxinyao/project/MAR_SynCode/docs/degradation_synthesis_guide.md`
> 审查对象: `/home/liuxinyao/project/ctagent/dataset/mar/ct_artifact_simulator.py` 及其依赖模块
> 日期: 2026-03-30

---

## 一、两套系统对照

### 1.1 退化类型覆盖

| 退化类型 | MAR_SynCode | ctagent | 备注 |
|----------|:-----------:|:-------:|------|
| 金属伪影 (MAR) | ✅ 主线 | ✅ `MARSimulator` | 架构一致，共享物理模块 |
| 低剂量噪声 (LDCT) | ✅ `simulate_data_lowdose.py` | ✅ `LowDoseNoiseSimulator` | **注入点不同** (见下) |
| 稀疏角 (Sparse-View) | ✅ `simulate_data_angle.py` | ✅ `SparseViewArtifactSimulator` | **实现策略不同** |
| 有限角 (Limited-Angle) | ✅ `simulate_data_angle.py` | ✅ `LimitedAngleArtifactSimulator` | **实现策略不同** |
| 环状伪影 (Ring) | ❌ | ✅ `RingArtifactSimulator` | ctagent 新增 |
| 运动伪影 (Motion) | ❌ | ✅ `MotionArtifactSimulator` | ctagent 新增 |
| 束硬化伪影 (Beam Hardening) | ⚠️ 内含于 pkev2kvp | ✅ `BeamHardeningArtifactSimulator` | **建模方式根本不同** |
| 散射伪影 (Scatter) | ⚠️ 仅常数 20 photon | ✅ `ScatterArtifactSimulator` | ctagent 模型更精细 |
| 截断伪影 (Truncation) | ❌ | ✅ `TruncationArtifactSimulator` | ctagent 新增 |
| 焦点模糊 (Focal Spot) | ❌ | ✅ `FocalSpotBlurSimulator` | ctagent 新增 |
| 混合退化组合 | ✅ 文档规范 | ✅ `CompositeArtifactSimulator` | **顺序约束需加强** |

### 1.2 公共物理链路对比

**MAR_SynCode 物理链路 (基准)**:
```
HU → μ → 组织分解 → 正投影 → pkev2kvp(多色) → Poisson噪声(I₀) → 高斯电子噪声 → BHC → FBP
                                     ↑                    ↑               ↑
                               金属注入点           低剂量注入点     欠采样注入点(角度)
```

**ctagent `_prepare_base_state` 物理链路**:
```
HU → μ → 组织分解 → 正投影 → pkev2kvp(多色) → Poisson噪声(I₀=2e7) → BHC → [poly_sinogram]
                                                                           ↑
                                                                     ⚠️ 缺少高斯电子噪声
                                                                           ↓
                                                              5类伪影在此处注入 → FBP
```

---

## 二、逐类退化物理正确性审查

### 2.1 环状伪影 (Ring) ✅ 物理正确

**当前实现**: 对 sinogram 特定探测器列施加增益偏差 + 加性偏置 + 视角相关漂移

**物理评判**: **正确**。环状伪影本质就是探测器响应不一致，在 sinogram 域表现为固定列的增益/偏置异常。该模型的三分量 (gain + bias + drift) 覆盖了：
- 死探测器 / 敏感度偏移 → gain
- 暗电流偏差 → bias
- 温度漂移 / 时间相关效应 → drift

**注入点**: BHC 后的 sinogram → **可接受**。真实中探测器异常发生在采集时（BHC 前），但由于 BHC 是逐像素多项式变换，在 BHC 前后注入的差异仅体现在非线性缩放上，对于小幅探测器异常影响不大。

**改进建议**: 无必须修改项。可选：增加"死探测器"模式 (output=0) 和"卡死探测器"模式 (output=常数)。

---

### 2.2 运动伪影 (Motion) ⚠️ 物理近似合理，模型简化

**当前实现**: 在连续视角范围内对 sinogram 行做探测器方向 1D 平移（ramp 渐变），混合 ghost 成分

**物理评判**: **简化但可接受**。真实运动伪影由患者刚体运动（平移+旋转）引起，在 sinogram 中表现为：
- 平移运动 → 探测器读数偏移（当前已建模）
- 旋转运动 → 视角与探测器联合偏移（**未建模**）
- 呼吸/心跳 → 周期性体积变化（**未建模**）

**注入点**: BHC 后 → **可接受**。运动在采集时发生，但 BHC 的多项式校正对平移后的线积分变化影响可忽略。

**问题**:
1. `sigma=0.8` 角向平滑是硬编码的，应随 severity 变化
2. 仅建模了 1D 平移，缺少旋转分量
3. `ghost_blend` 混合的是"移动前"的读数，物理上应该是"移动中"的积分混叠

**改进建议**:
- 增加旋转运动分量：在视角方向也做小幅偏移 (`angle_shift`)
- 角向平滑 sigma 应与 `motion_fraction` 和 `translation_mm` 关联
- 考虑增加周期性运动模式（呼吸模型）

---

### 2.3 束硬化伪影 (Beam Hardening) ❌ 物理建模有误

**当前实现**:
```python
bone_weight = normalize(p_bone_kev) ** 1.5
result = sinogram + bone_weight * (alpha * sinogram² + beta * sinogram³)
```

**物理评判**: **建模逻辑反向**。

**问题 1: BHC 残差方向错误**

真实物理流程:
```
多色投影 P_kvp (含束硬化) → BHC校正 → P_bhc (理想无硬化)
```
如果 BHC 不完美，残差 = `P_bhc - P_ideal`，表现为**高衰减区域的低估**（cupping 效应）。

当前实现在 **已经 BHC 校正后** 的 `poly_sinogram` 上叠加 `+ bone_weight * (α·P² + β·P³)`。这意味着：
- α, β > 0 时：增大了高值区域的投影 → 重建后高值区域**偏高** → **反 cupping** (凸起)
- 真实束硬化残差应该是 cupping (凹陷)

**问题 2: 应该在 BHC 之前模拟，或直接扰动 BHC 系数**

MAR_SynCode 的做法：束硬化是 `pkev2kvp` 多色模拟的自然结果，BHC 只是后处理校正。要模拟"束硬化伪影"，正确做法是：
- **方案 A**: 跳过 BHC 或使用不匹配的 BHC 系数 → 自然产生束硬化残余
- **方案 B**: 在 `pkev2kvp` 阶段修改能谱参数 (如用 80kVp 替代 120kVp) → 改变硬化程度
- **方案 C**: 人为加大骨成分的衰减系数比值 → 增强多色效应

**问题 3: `p_bone_kev` 是单能参考投影，与当前 `poly_sinogram` 域不一致**

`p_bone_kev` 是 70keV 单能下骨的前向投影，`sinogram` 是多色+噪声+BHC 后的投影。两者在值域和物理含义上不同，直接用 `p_bone_kev` 做权重缺乏严格的物理对应关系。

**修复建议**:

```python
# 方案 A: 扰动 BHC 系数 (推荐，改动最小)
def _apply_to_sinogram(self, base, sinogram, severity):
    # 用 pre-BHC 的带噪声投影
    proj_kvp_noise = base["proj_kvp_noise"]  # 需要在 _prepare_base_state 中保留
    
    # 使用不匹配的 BHC 系数: 缩放为 (1-delta)
    delta = sample(severity)  # mild: 0.05-0.10, severe: 0.20-0.40
    perturbed_bhc = self.phy.para_bhc * (1.0 - delta)
    result = apply_bhc(proj_kvp_noise, perturbed_bhc)
    return result, {...}
```

```python
# 方案 B: 不做 BHC (直接用多色投影)
def _apply_to_sinogram(self, base, sinogram, severity):
    proj_kvp_noise = base["proj_kvp_noise"]
    # 完全跳过 BHC → 最大束硬化
    # 或部分 BHC → 中间态
    blend = severity_to_blend(severity)  # 1.0=无BHC, 0.0=完全BHC
    result = blend * proj_kvp_noise + (1-blend) * sinogram
    return result, {...}
```

---

### 2.4 散射伪影 (Scatter) ⚠️ 模型方向正确，注入点有物理瑕疵

**当前实现**:
```python
primary = exp(-sinogram)        # 视为透射率
blurred = gaussian_filter(primary, sigma=(2.0, sigma_bins))
scatter = scatter_ratio * blurred
measured = primary + scatter     # 散射叠加
result = -log(measured)          # 回到投影域
```

**物理评判**: **模型框架正确，但输入域不严格**。

散射的物理本质：
```
真实测量 = I₀ × [exp(-∫μdl) + S(x,θ)]
S(x,θ) ≈ ratio × blur(exp(-∫μdl))     ← 散射近似为初级辐射的低频平滑版本
```

**问题**: `sinogram` 是 BHC 后的投影，`exp(-BHC(P))` ≠ `I/I₀` (原始透射率)。BHC 的三阶多项式变换改变了值域：
```
P_bhc = c₁P + c₂P² + c₃P³
exp(-P_bhc) ≠ exp(-P)
```

因此 `primary = exp(-sinogram)` 不是真实的透射率分布，而是经过非线性变换后的近似。

**严重程度**: **中等**。对于 BHC 系数接近线性（c₁≈1, c₂≈0, c₃≈0）的情况，误差较小。从 `para_bhc` 的典型值 `[0.80, 0.023, -0.0008]` 看，确实以线性项为主，非线性修正约 2-3%，因此实际偏差可控。

**修复建议**:
```python
# 在 _prepare_base_state 中保留 pre-BHC 投影
base["proj_kvp_noise"] = proj_kvp_noise  # BHC 之前的噪声投影

# scatter 仿真使用 pre-BHC 投影
def _apply_to_sinogram(self, base, sinogram, severity):
    proj_raw = base["proj_kvp_noise"]
    primary = np.exp(-proj_raw)
    blurred = gaussian_filter(primary, sigma=(2.0, sigma_bins))
    scatter = scatter_ratio * blurred
    measured = np.clip(primary + scatter, 1e-12, None)
    contaminated_proj = -np.log(measured)
    # 重新 BHC
    result = apply_bhc(contaminated_proj, self.phy.para_bhc)
    return result, {...}
```

---

### 2.5 截断伪影 (Truncation) ⚠️ 模型过于简化

**当前实现**: 对 sinogram 两端各 `width` 列做 cosine 渐变衰减

**物理评判**: **方向正确但过于简化**。

真实截断伪影的成因：
```
患者身体超出探测器 FOV
 → 部分射线路径的衰减积分 ∫μdl 被截断（高估透射率）
 → sinogram 边缘值偏小（非零但低于真实值）
 → FBP 重建时频域截断 → 亮边 + 低频偏移 (DC cupping)
```

**问题 1: cosine 窗不是物理正确的截断模型**

真实截断中：
- sinogram 边缘**不会衰减到 0**，而是衰减到**部分路径的积分**
- 截断宽度由 FOV 与体模大小决定，不是百分比

当前 `ramp = 0.5*(1 - cos(...))` 从 0 渐变到 1：
```
result[:, :width] *= ramp       # 左侧从 0 渐变到 1
result[:, -width:] *= ramp[::-1]  # 右侧从 1 渐变到 0
```
这导致最边缘 bin 的值 → 0，而真实截断不会归零（除非完全在 FOV 外）。

**问题 2: 缺少 FOV 外扩展体模型**

真正的截断仿真应该是：
1. 将图像零填充/扩展到更大尺寸
2. 对扩展图像做正投影（更宽的 sinogram）
3. 截取中央 641 bins 作为"截断后的 sinogram"
4. FBP 重建

**修复建议**:
```python
# 方案 A: 物理正确的截断 (需要更大的正投影)
def _apply_to_sinogram(self, base, sinogram, severity):
    # 将 μ 图像嵌入更大的画布 (模拟体模大于 FOV)
    mu = base["mu_image"]
    padded = np.pad(mu, pad_width, mode='edge')  # 边缘扩展
    # 用更宽的探测器做正投影
    full_sino = extended_geometry.forward(padded)
    # 截取中央 bins 作为截断 sinogram
    center = full_sino.shape[1] // 2
    half_det = sinogram.shape[1] // 2
    result = full_sino[:, center-half_det:center+half_det+1]
    return result, {...}

# 方案 B: 改进的现象学模型 (简单修复)
def _apply_to_sinogram(self, base, sinogram, severity):
    # 不归零，而是保留一个最小值 (模拟部分路径积分)
    min_fraction = 0.3  # 边缘至少保留 30% 的原始值
    ramp = min_fraction + (1-min_fraction) * 0.5*(1 - cos(linspace(0, pi, width)))
    result[:, :width] *= ramp
    result[:, -width:] *= ramp[::-1]
```

---

### 2.6 低剂量噪声 (Low-Dose) ❌ 注入点不正确

**当前实现**:
```python
# 在 BHC 后的 sinogram 上操作
transmission = exp(-sinogram)   # sinogram 是 BHC 后的！
expected = transmission * photon_num_low + scatter
actual = Poisson(expected)
actual += Gaussian(0, electronic_sigma)
result = -log(actual / photon_num_low)
```

**MAR_SynCode 正确做法**:
```python
# 在正投影后、BHC 前操作
proj_kvp = pkev2kvp(...)
N = Poisson(I₀_low * exp(-proj_kvp) + scatter)
P_noisy = -log(N / I₀_low)
P_bhc = apply_bhc(P_noisy)      # 噪声投影再做 BHC
```

**问题**:

1. **BHC 后加噪声违反物理时序**: 真实中噪声发生在探测器采集时（BHC 前），BHC 会对噪声进行非线性变换（放大/压缩），BHC 前后加噪声的结果不同。
2. **`exp(-BHC(P))` ≠ 原始透射率**: 与 scatter 相同的问题。
3. **基线已有满剂量噪声**: `_prepare_base_state` 已在 I₀=2e7 下加了 Poisson 噪声，再在 BHC 后重新加低剂量噪声相当于**叠加了两次噪声**。

**修复建议**:
```python
def _apply_to_sinogram(self, base, sinogram, severity):
    # 直接使用 pre-BHC、pre-noise 的干净多色投影
    proj_kvp_clean = base["proj_kvp"]  # 需要在 _prepare_base_state 中保留
    
    # 低剂量 Poisson
    photon_num_low = self.phy.config.photon_num * dose_fraction
    noisy = add_poisson_noise(proj_kvp_clean, photon_num_low, scatter, self.rng)
    
    # 电子噪声
    noisy += self.rng.normal(0, electronic_sigma / photon_num_low, noisy.shape)
    
    # BHC
    result = apply_bhc(noisy, self.phy.para_bhc)
    return result, {...}
```

---

### 2.7 稀疏角 (Sparse-View) ⚠️ 实现策略不同，物理近似

**当前实现**: 从完整 640 views sinogram 中均匀抽取 + 线性插值填充缺失行

**MAR_SynCode 正确做法**: 以稀疏角度集做正投影 → 加噪声 → 稀疏角 FBP

**问题**:
1. 当前方法用**插值填充**缺失视角，相当于假设缺失视角的投影可由相邻视角线性外推。这**减弱**了稀疏角伪影的严重程度，因为真正的 FBP 在缺失角度上会产生更强的条纹。
2. 更关键的是：缺失视角的噪声特性与真实采集不同。当前方法的"缺失视角"是从**同一次满剂量采集**插值得到的，噪声已经被平滑；而真实稀疏角采集中**不存在**这些视角的数据。

**修复建议**: 设置 `result[missing_views] = 0` (不插值)，或使用带权重的 FBP。更彻底的做法是为稀疏角构建独立的 `CTGeometry`：
```python
sparse_geo = CTGeometry(CTGeometryConfig(num_angles=num_views, ...))
sparse_sino = sparse_geo.forward(base["mu_image"])
# 加噪声
sparse_sino_noisy = add_poisson_noise(sparse_sino, ...)
# 稀疏角 FBP
result = sparse_geo.fbp(sparse_sino_noisy)
```

---

### 2.8 有限角 (Limited-Angle) ⚠️ 同上

**当前实现**: 保留连续角度范围内的视角，其余置零 + cosine 边界过渡

**问题**: 与稀疏角相同 — 应使用独立的有限角正投影 + FBP，而不是对满角度 sinogram 做掩膜。

---

## 三、基线 Pipeline 缺陷

### 3.1 缺少高斯电子噪声

**MAR_SynCode**: `P_final = P_noisy + N(0, σ²)`，σ = 0.01

**ctagent `_prepare_base_state`**: 只有 Poisson 噪声，无 Gaussian 电子噪声

**影响**: 轻微。电子噪声量级远小于 Poisson 噪声 (σ=0.01 vs Poisson 标准差 ~0.01-0.1)，但为完整性应补上。

### 3.2 `_prepare_base_state` 未保留中间态

当前 `_prepare_base_state` 只返回最终的 `poly_sinogram` (BHC 后)，不保留：
- `proj_kvp`: 多色投影（BHC 前，无噪声）
- `proj_kvp_noise`: 带噪声的多色投影（BHC 前）

这导致所有子类仿真器**被迫在 BHC 后操作**，无法在正确的物理注入点插入退化。

**修复**: 在 `_prepare_base_state` 返回字典中增加这两个中间态。

---

## 四、混合退化物理顺序问题

### 4.1 当前 `CompositeArtifactSimulator` 的做法

```python
sinogram = base["poly_sinogram"].copy()  # BHC 后
for gen, severity in recipe:
    sinogram, meta = gen._apply_to_sinogram(base, sinogram, severity)
ma_ct = self.geo.fbp(sinogram)
```

所有伪影**顺序叠加在 BHC 后的 sinogram 上**，然后一次 FBP。

### 4.2 物理正确的混合退化顺序

参考 MAR_SynCode 的规范，正确的注入顺序为：

```
HU → μ
 ↓ 组织分解
 ↓ 正投影 (角度方案在此决定: 稀疏角/有限角/全角度)
 ↓ pkev2kvp 多色转换 (束硬化在此自然产生)
 ↓ 散射叠加 (在透射域, BHC 前)
 ↓ Poisson 噪声 (I₀ 决定剂量)
 ↓ 电子噪声 (高斯)
 ↓ 探测器异常 (ring artifact — 增益/偏置/漂移)
 ↓ BHC (可选扰动系数来模拟 BH 残差)
 ↓ 运动效应 (sinogram 行偏移) — BHC 前后均可
 ↓ FBP 重建 (使用对应角度方案)
 ↓ 截断效应 (更准确的做法是在正投影时处理)
```

### 4.3 当前混合退化中的问题

| 组合 | 问题 | 严重程度 |
|------|------|----------|
| Ring + Scatter | 两者都在 BHC 后操作，ring 的增益偏差影响了 scatter 的 `exp(-)` 计算 | 低 |
| BH + Low-Dose | BH 用的是 BHC 后非线性项，Low-Dose 又重新取 exp 加噪，两者物理域不一致 | **高** |
| Scatter + Low-Dose | 两者都取 `exp(-sino)` 但含义不同 (一个加散射，一个加噪声)，叠加后物理不自洽 | **高** |
| Sparse + Ring | 先抽稀视角再加 ring，但 ring 应在所有视角上都存在 | 中 |
| Truncation + 任意 | cosine 衰减可能破坏其他伪影在边缘的特征 | 低 |

### 4.4 禁止的操作顺序 (来自 MAR_SynCode 规范)

| 错误操作 | 当前状态 | 说明 |
|----------|----------|------|
| ✗ 先重建再叠加噪声 | ✅ 未违反 | 所有噪声在 sinogram 域 |
| ✗ BHC 在正投影之前 | ✅ 未违反 | BHC 在噪声之后 |
| ✗ 多次独立加噪声 | ❌ **违反** | LowDose 在基线 Poisson 之上再加一次 |
| ✗ exp(-BHC(P)) 当作透射率 | ❌ **违反** | Scatter 和 LowDose 都这样做 |

---

## 五、修复优先级与行动计划

### P0 (必须修复 — 物理错误)

| # | 问题 | 修复方案 | 涉及文件 |
|---|------|----------|----------|
| 1 | `_prepare_base_state` 缺少中间态 | 返回 `proj_kvp` 和 `proj_kvp_noise` | `ct_artifact_simulator.py` |
| 2 | BeamHardening 建模方向错误 | 改为 BHC 系数扰动或跳过 BHC 的 blend | `ct_artifact_simulator.py` |
| 3 | LowDose 在 BHC 后重复加噪 | 改为在 `proj_kvp` 上重新加噪 → BHC | `ct_artifact_simulator.py` |
| 4 | Scatter 用 BHC 后投影取 exp | 改为在 `proj_kvp_noise` 上叠加散射 → BHC | `ct_artifact_simulator.py` |

### P1 (建议修复 — 提升物理保真度)

| # | 问题 | 修复方案 | 涉及文件 |
|---|------|----------|----------|
| 5 | 基线缺少高斯电子噪声 | 在 Poisson 后添加 N(0, σ²) | `ct_artifact_simulator.py` |
| 6 | SparseView/LimitedAngle 用插值而非独立投影 | 构建稀疏角 Geometry 做独立正投影+FBP | `ct_artifact_simulator.py` |
| 7 | Truncation 过于简化 (cosine→0) | 改为 `min_fraction + (1-min_fraction)*ramp` | `ct_artifact_simulator.py` |
| 8 | Motion 缺少旋转分量 | 增加视角偏移 (angle_shift) | `ct_artifact_simulator.py` |

### P2 (可选增强)

| # | 问题 | 修复方案 |
|---|------|----------|
| 9 | CompositeArtifactSimulator 无顺序约束 | 根据物理流程自动排序 recipe |
| 10 | Ring 缺少死探测器/卡死模式 | 增加 `dead`/`stuck` 模式 |
| 11 | Motion sigma=0.8 硬编码 | 改为随 severity 变化 |

---

## 六、修复后的理想物理链路

```
HU → μ → 组织分解 → 正投影(P_water, P_bone)
  ↓
pkev2kvp → proj_kvp (多色,干净)     ← 保留此中间态
  ↓
[可选] 散射叠加 (在透射域)          ← Scatter 在此注入
  ↓
Poisson噪声(I₀) + 高斯电子噪声     ← LowDose 改变 I₀
  ↓
proj_kvp_noise                       ← 保留此中间态
  ↓
[可选] 探测器异常 (ring)             ← Ring 在此注入
  ↓
BHC(para_bhc × perturbation)        ← BeamHardening 扰动 BHC 系数
  ↓
[可选] 运动偏移 (motion)             ← Motion 在此注入
  ↓
FBP → 重建图像
  ↓
[可选] 截断: 通过扩展 FOV 正投影实现
```

**混合退化时的合法顺序**:
```
Scatter → Poisson(I₀) → Ring → BHC(perturbation) → Motion → FBP
   ①          ②          ③          ④                ⑤       ⑥
```

---

## 七、与 MAR_SynCode 的参数对应

| 参数 | MAR_SynCode | ctagent | 一致性 |
|------|-------------|---------|--------|
| μ_water | 0.192 cm⁻¹ | 0.192 cm⁻¹ | ✅ |
| I₀ (标准剂量) | 2×10⁷ | 2×10⁷ | ✅ |
| 散射光子 | 20 | 20 | ✅ |
| 参考能量 | 70 keV | 70 keV | ✅ |
| 能谱范围 | 20-120 keV | 20-120 keV | ✅ |
| BHC 阶数 | 3 阶多项式 | 3 阶多项式 | ✅ |
| 图像尺寸 | 416×416 | 416×416 | ✅ |
| 探元数 | 641 (ODL) | 641 (ODL) | ✅ |
| 全角度数 | 640 | 640 | ✅ |
| 滤波器 | Ram-Lak | Ram-Lak | ✅ |
| 组织分解阈值 | 0.210 / 0.480 | HU→μ 计算 | ✅ (换算后一致) |
| PVE factor | 1/4 | 1/4 | ✅ (MAR 路径) |

物理常数和几何参数完全一致，问题主要在伪影注入的**操作流程顺序**上。

---

*文档版本: v1.0 | 日期: 2026-03-30 | 审查范围: ct_artifact_simulator.py 全部 9 类退化*
