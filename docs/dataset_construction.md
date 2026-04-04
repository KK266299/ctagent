# 退化数据集构建说明

## 概览

本项目采用 **两阶段流水线** 构建退化数据集：

```
阶段一：DICOM → 干净切片（Clean Dataset）
阶段二：干净切片 → 退化切片（Degraded Dataset）
```

最终产出配对的 **干净-退化** 图像对，用于监督式 CT 图像复原模型的训练与评估。

---

## 阶段一：干净数据集构建

**入口脚本：** `scripts/build_clean_dataset.py`

### 处理流程

```
DICOM 目录 → 扫描分组 → 提取切片 → HU 转换 → 窗宽窗位 → 归一化 → .npy + Manifest
```

### 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| DICOM 扫描器 | `dataset/dicom_scanner.py` | 递归扫描 `.dcm` 文件，按 `SeriesInstanceUID` 分组 |
| DICOM 读取器 | `dataset/dicom_reader.py` | 像素 → HU 转换、窗宽窗位归一化 |
| 切片导出器 | `dataset/ct_slice_exporter.py` | 将切片保存为 `.npy`，记录元数据 |
| Manifest 管理 | `dataset/manifest.py` | JSONL 格式的清单读写 |

### HU 转换与归一化

```python
# 1. 像素值 → HU（Hounsfield Units）
HU = pixel_value * RescaleSlope + RescaleIntercept  # 默认裁剪 [-1024, 3071]

# 2. 窗宽窗位归一化 → [0, 1]
normalized = clip((HU - (center - width/2)) / width, 0, 1)
# 默认：center=40（软组织窗）, width=400
```

### 配置文件

**`configs/data/dicom_dataset.yaml`**

```yaml
input:
  root_dir: "/home/liuxinyao/data"
  max_patients: null            # 设为整数可限制病人数（调试用）
filter:
  min_slices: 10                # 序列最少切片数
  description_contains: ""      # 按序列描述过滤
window:
  center: null                  # null 则使用 DICOM 头中的值
  width: null
output:
  clean_dir: "output/clean"
  manifest_path: "output/manifests/clean_manifest.jsonl"
  max_series: null
```

### 输出结构

```
output/clean/
├── {patient_id}/
│   ├── {series_uid_short}/
│   │   ├── slice_0000.npy      # float32, [0, 1]
│   │   ├── slice_0001.npy
│   │   └── ...
│   └── ...
└── ...

output/manifests/
└── clean_manifest.jsonl
```

### 运行命令

```bash
PYTHONPATH=. python scripts/build_clean_dataset.py --config configs/data/dicom_dataset.yaml
```

---

## 阶段二：退化数据集构建

**入口脚本：** `scripts/build_degraded_dataset.py`
**核心引擎：** `dataset/degradation_builder.py`

### 处理流程

```
Clean Manifest → 读取干净 .npy → 施加退化 → 保存退化 .npy → 记录 Manifest
```

### 四种退化类型

#### 1. 噪声（Noise）— 模拟低剂量 CT

添加高斯噪声：`degraded = image + N(0, σ²)`，结果裁剪至 `[0, 1]`。

| 严重等级 | σ (sigma) |
|---------|-----------|
| 1 | 0.02 |
| 2 | 0.05 |
| 3 | 0.08 |
| 4 | 0.12 |
| 5 | 0.18 |

#### 2. 模糊（Blur）— 模拟运动模糊 / 散焦

使用 `scipy.ndimage.gaussian_filter` 进行高斯模糊。

| 严重等级 | σ (sigma) |
|---------|-----------|
| 1 | 0.5 |
| 2 | 1.0 |
| 3 | 1.5 |
| 4 | 2.0 |
| 5 | 3.0 |

#### 3. 下采样（Downsample）— 模拟低分辨率 / 厚层重建

先缩小再放大（双线性插值），引入分辨率损失。

| 严重等级 | 缩放因子 |
|---------|---------|
| 1 | 1.5× |
| 2 | 2.0× |
| 3 | 3.0× |
| 4 | 4.0× |
| 5 | 6.0× |

```
原图 (H, W) → 缩小至 (H/factor, W/factor) → 放大回 (H, W)
```

#### 4. 伪影（Artifact）— 模拟条纹伪影

在随机行或列叠加亮线条：`degraded = image + streak_pattern`。

| 严重等级 | 条纹数 | 强度 |
|---------|--------|------|
| 1 | 2 | 0.03 |
| 2 | 4 | 0.05 |
| 3 | 6 | 0.08 |
| 4 | 8 | 0.12 |
| 5 | 12 | 0.18 |

### 配置文件

**`configs/data/degradation.yaml`**

```yaml
input:
  clean_manifest: "output/manifests/clean_manifest.jsonl"
  max_slices: null               # 设为整数可限制处理切片数
output:
  degraded_dir: "output/degraded"
  manifest_path: "output/manifests/degraded_manifest.jsonl"
degradations:
  - type: "noise"
    severities: [1, 2, 3, 4, 5]
  - type: "blur"
    severities: [1, 2, 3]
  - type: "downsample"
    severities: [1, 2, 3]
  - type: "artifact"
    severities: [1, 2, 3]
seed: 42                          # 随机种子，保证可复现
```

### 输出结构

```
output/degraded/
├── noise/
│   ├── severity_1/
│   │   ├── {slice_id}.npy
│   │   └── ...
│   ├── severity_2/
│   └── ...
├── blur/
│   ├── severity_1/
│   └── ...
├── downsample/
│   ├── severity_1/
│   └── ...
└── artifact/
    ├── severity_1/
    └── ...

output/manifests/
└── degraded_manifest.jsonl
```

### Manifest 记录格式

每条退化切片在 `degraded_manifest.jsonl` 中对应一行 JSON：

```json
{
  "slice_id": "patient123__series_uid_suffix__slice_0001",
  "patient_id": "patient123",
  "series_uid": "1.2.840.113619...",
  "degradation_type": "noise",
  "severity": 3,
  "params": {"sigma": 0.08},
  "clean_path": "output/clean/patient123/series_uid_suffix/slice_0001.npy",
  "degraded_path": "output/degraded/noise/severity_3/patient123__series_uid_suffix__slice_0001.npy"
}
```

### 运行命令

```bash
PYTHONPATH=. python scripts/build_degraded_dataset.py --config configs/data/degradation.yaml
```

---

## 训练用 Dataset 类

| 类名 | 文件 | 用途 |
|------|------|------|
| `CTDataset` | `src/datasets/ct_dataset.py` | 单张图像加载，支持 `.dcm`/`.nii`/`.png`/`.npy` |
| `PairedCTDataset` | `src/datasets/paired_dataset.py` | 配对加载退化+干净图像，用于复原训练 |

**`PairedCTDataset` 返回格式：**

```python
{
    "degraded": np.ndarray,      # 退化图像
    "clean": np.ndarray,         # 对应的干净图像
    "degraded_path": str,
    "clean_path": str,
}
```

---

## 辅助工具

### 退化模拟器（运行时增强）

`src/degradations/simulator.py` — 支持在训练时动态施加噪声、模糊、低分辨率退化。

### 退化检测器

`src/degradations/detector.py` — 分析图像退化类型，基于拉普拉斯 MAD 估计噪声水平，返回 `DegradationReport`。

### Toy 合成数据

`dataset/toy.py` — 生成 256×256 的合成 CT 仿体（phantom），包含背景、身体轮廓、器官和病灶，无需真实 DICOM 即可验证流水线。

### I/O 与窗宽窗位

- `src/io/readers.py` — 自动检测格式读取 CT 图像
- `src/io/windowing.py` — 预设窗位（soft_tissue / lung / bone / brain / liver）

---

## 完整流程示意

```
┌─────────────────────────────────────────────────────────┐
│                    DICOM 原始数据                         │
│              /home/liuxinyao/data/                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  阶段一：build_clean_dataset.py                          │
│  ┌────────────┐  ┌──────────┐  ┌───────────────────┐    │
│  │ DICOM 扫描  │→│ HU 转换   │→│ 窗宽窗位 → [0,1]  │    │
│  └────────────┘  └──────────┘  └───────────────────┘    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  output/clean/*.npy  +  clean_manifest.jsonl             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  阶段二：build_degraded_dataset.py                       │
│  ┌────────┐  ┌────────┐  ┌────────────┐  ┌──────────┐  │
│  │ 噪声    │  │ 模糊   │  │ 下采样      │  │ 伪影     │  │
│  │ σ=0.02 │  │ σ=0.5  │  │ factor=1.5 │  │ n=2,i=.03│  │
│  │  ...   │  │  ...   │  │   ...      │  │   ...    │  │
│  │ σ=0.18 │  │ σ=3.0  │  │ factor=6.0 │  │ n=12,i=.18│ │
│  └────────┘  └────────┘  └────────────┘  └──────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  output/degraded/{type}/severity_{n}/*.npy               │
│  +  degraded_manifest.jsonl                              │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  PairedCTDataset：加载配对数据用于训练                      │
│  { "degraded": ..., "clean": ..., ... }                  │
└──────────────────────────────────────────────────────────┘
```
