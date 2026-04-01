# CT-Agent 项目工作手册

## 1. 目录结构总览

| 名称 | 路径 | 说明 |
|------|------|------|
| 项目代码 | `/home/liuxinyao/project/ctagent` | 主代码仓库 |
| CQ500 原始数据 | `/home/liuxinyao/data/cq500` | CQ500 DICOM + `reads.csv` 诊断标签 |
| CQ500 处理后数据 | `/home/liuxinyao/data/cq500_processed` | 预处理后的 `.h5` μ 值切片 |
| BHX 标注 | `/home/liuxinyao/data/physionet.org/files/bhx-brain-bounding-box/1.1/` | PhysioNet Brain Hemorrhage Extended |
| SOP Index | `/home/liuxinyao/data/cq500_sop_index.json` | DICOM SOP UID → 处理后路径映射 |
| 实验结果 | `/home/liuxinyao/project/ctagent/results/` | 评估结果、标签等 |
| 日志输出 | `/home/liuxinyao/output/ctagent/` | 标准化日志 (eval / iqa) |
| 可视化临时输出 | `/home/liuxinyao/project/ctagent/try_output/` | 调试/可视化图片 |
| DnCNN 权重 | `/home/liuxinyao/project/ctagent/checkpoints/dncnn_ct.pth` | 训练好的深度降噪模型 |
| 物理参数 | `/home/liuxinyao/project/ctagent/data/mar_physics/` | CT 仿真材料参数 (.mat) |

### 1.1 处理后数据结构

```
/home/liuxinyao/data/cq500_processed/
├── CQ500CT0/
│   └── PLAIN_THIN/
│       ├── CT000000/
│       │   ├── gt.h5          # 干净 GT (μ 值, float64, 416×416)
│       │   ├── 0.h5           # 金属掩模 0 的退化结果
│       │   ├── 1.h5           # 金属掩模 1 的退化结果
│       │   └── ...            # 共 10 个掩模
│       ├── CT000005/
│       └── ...
├── CQ500CT10/
└── ...                        # 共 ~490 个 patient, ~22509 个 slice
```

### 1.2 日志输出约定

```
/home/liuxinyao/output/ctagent/
├── eval/                      # 评估类任务日志 + 输出
│   ├── restoration_rule_full/           # Rule 模式全量修复输出
│   │   └── evaluation_summary.json
│   ├── restoration_llm_test/            # LLM 模式测试修复输出
│   │   ├── evaluation_summary.json
│   │   └── *.png                        # 可视化对比图
│   ├── rule_restoration_full_all_slice_20260327_123252.log
│   ├── restoration_llm_test_20260328_043202.log
│   └── ...
└── iqa/                       # IQA 评估日志
```

**日志命名规则**: `{task_name}_{YYYYMMDD_HHMMSS}.log`

**命令尾部固定追加**: `2>&1 | tee /home/liuxinyao/output/ctagent/{eval|iqa}/{logname}.log`

### 1.3 结果目录结构

```
/home/liuxinyao/project/ctagent/results/
├── slice_labels/              # 逐 slice 标签
│   ├── slice_labels.csv       # CSV 格式 (22509 行)
│   ├── slice_labels.jsonl     # JSONL 格式
│   └── stats.json             # 统计摘要
├── cq500_api_eval/            # VLM API 诊断评估结果
│   ├── predictions.jsonl      # 逐 case 预测详情
│   ├── summary.csv            # 汇总指标
│   ├── summary.json           # 汇总指标 (JSON)
│   └── case_studies.json      # 典型 case 分析
├── cq500_iqa_eval/            # IQA 评估结果
└── formal_comparison/         # 正式对比实验结果
```

---

## 2. 配置文件

| 配置文件 | 用途 |
|----------|------|
| `configs/experiment/cq500_api_eval.yaml` | VLM API 诊断评估主配置 (含 LLM 模型、数据路径、窗位参数) |
| `configs/experiment/cq500_iqa_eval.yaml` | IQA 质量评估配置 |
| `configs/experiment/formal_comparison.yaml` | 正式对比实验配置 |
| `configs/experiment/api_guided.yaml` | API-guided 闭环修复配置 |
| `configs/experiment/full_pipeline.yaml` | 完整 pipeline 配置 |

### 2.1 LLM 模型配置 (cq500_api_eval.yaml)

```yaml
llm:
  provider: "openai"
  model: "qwen/qwen-2.5-vl-72b-instruct"   # OpenRouter 路由
  base_url: "https://openrouter.ai/api/v1"
  temperature: 0.1
  max_tokens: 1024
  timeout: 120
```

API Key 通过环境变量设置:
```bash
export OPENAI_API_KEY="sk-or-v1-你的OpenRouter密钥"
```

---

## 3. 数据存储格式

### 3.1 CT 切片 (gt.h5 / 0~9.h5)

| 字段 | 值 |
|------|-----|
| 格式 | HDF5 |
| dataset key | `image` |
| dtype | float64 |
| shape | (416, 416) |
| 值域 | μ 值 (线衰减系数), 约 [0, 0.5] |
| μ_water | 0.192 |
| HU 转换 | `HU = (μ / 0.192 - 1) × 1000` |

### 3.2 Slice Labels (slice_labels.csv)

```
patient_id, series, slice_name, sop_uid, bhx_coverage, lesion_present,
ICH, IPH, IVH, SDH, EDH, SAH, Fracture, CalvarialFracture, MassEffect, MidlineShift,
bhx_labels, strategy
```

- `bhx_coverage`: 该 patient 是否有 BHX 标注覆盖 (bool)
- `lesion_present`: 该 slice 是否有病灶 (0/1)
- 各 label 列: 二值 (0/1)
- `strategy`: `inherit_all` 或 `bhx_aware`

### 3.3 Slice Labels (slice_labels.jsonl)

每行一个 JSON 对象:
```json
{
  "patient_id": "CQ500CT0",
  "series": "PLAIN_THIN",
  "slice_name": "CT000000",
  "sop_uid": "1.2.276...",
  "bhx_coverage": true,
  "lesion_present": 0,
  "labels": {"ICH": 0, "IPH": 0, ...},
  "bhx_label_names": [],
  "strategy": "inherit_all"
}
```

### 3.4 API 评估预测 (predictions.jsonl)

每行一个 JSON, 包含:
```json
{
  "case_id": "CQ500CT0",
  "input_type": "clean|degraded|restored",
  "gt": {"ICH": 0, ...},
  "predictions": {"ICH": 0, ...},
  "confidence": {"ICH": 1.0, ...},
  "reasoning": "...",
  "api_latency_sec": 12.04,
  "usage": {"prompt_tokens": 1224, "completion_tokens": 280}
}
```

### 3.5 修复评估摘要 (evaluation_summary.json)

```json
[
  {
    "case": "CQ500CT0_PLAIN_THIN_CT000120",
    "artifact_type": "ring",
    "severity": "mild",
    "planner": "rule|llm",
    "detected": "noise(mild), artifact_ring(mild)",
    "tools": "ring_removal_polar→denoise_wavelet",
    "reasoning": "...",
    "skipped": false,
    "psnr_before": 51.36,
    "ssim_before": 0.9936,
    "psnr_after": 51.76,
    "ssim_after": 0.9942,
    "psnr_delta": 0.40,
    "ssim_delta": 0.0006
  }
]
```

### 3.6 可视化图片命名

```
{patient}_{series}_{slice}_{artifact}_{severity}_restore_{window}.png
```

示例: `CQ500CT0_PLAIN_THIN_CT000120_ring_severe_restore_brain.png`

---

## 4. 命令手册

> 所有命令在 tmux 中运行，conda 环境为 `llamafactory`。
> GPU 默认使用 GPU1 (`CUDA_VISIBLE_DEVICES=1`)。

### 4.1 前置: 构建 SOP Index (仅需一次)

```bash
PYTHONPATH=. python scripts/build_sop_index.py \
    --cq500-root /home/liuxinyao/data/cq500 \
    --processed-root /home/liuxinyao/data/cq500_processed \
    --output /home/liuxinyao/data/cq500_sop_index.json
```

### 4.2 生成逐 Slice 标签

```bash
PYTHONPATH=. python scripts/build_slice_labels.py \
    --config configs/experiment/cq500_api_eval.yaml \
    --strategy inherit_all \
    --output-dir results/slice_labels \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/build_slice_labels_$(date +%Y%m%d_%H%M%S).log
```

**可选 strategy**:
- `inherit_all`: 有病灶 → 继承全部 case-level 标签; 无病灶 → 全 0
- `bhx_aware`: 根据 BHX 标注的具体出血亚型精确映射

**输出**:
- `results/slice_labels/slice_labels.csv`
- `results/slice_labels/slice_labels.jsonl`
- `results/slice_labels/stats.json`

### 4.3 CT 伪影生成 & 可视化

```bash
PYTHONPATH=. bash -c '
eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/test_ct_artifact_generators.py \
    --output-dir try_output \
    --num-slices 2' \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/artifact_gen_$(date +%Y%m%d_%H%M%S).log
```

**输出**: `try_output/{patient}_{artifact}_severity_{window}.png`

### 4.4 伪影修复评估 — Rule 模式

**测试 (2 slices):**
```bash
PYTHONPATH=. bash -c '
eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_ct_artifact_restoration.py \
    --planner rule \
    --output-dir /home/liuxinyao/output/ctagent/eval/restoration_rule_test \
    --gpu 1 \
    --num-slices 2 \
    --save-images' \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/rule_restoration_test_$(date +%Y%m%d_%H%M%S).log
```

**全量 (所有 slice, ~22509):**
```bash
PYTHONPATH=. bash -c '
eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_ct_artifact_restoration.py \
    --planner rule \
    --output-dir /home/liuxinyao/output/ctagent/eval/restoration_rule_full \
    --gpu 1 \
    --all-slices \
    --no-save-images' \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/rule_restoration_full_$(date +%Y%m%d_%H%M%S).log
```

### 4.5 伪影修复评估 — LLM 模式

**测试 (2 slices):**
```bash
PYTHONPATH=. bash -c '
eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_ct_artifact_restoration.py \
    --planner llm \
    --llm-model qwen/qwen-2.5-vl-72b-instruct \
    --llm-base-url https://openrouter.ai/api/v1 \
    --output-dir /home/liuxinyao/output/ctagent/eval/restoration_llm_test \
    --gpu 1 \
    --num-slices 2 \
    --save-images' \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/restoration_llm_test_$(date +%Y%m%d_%H%M%S).log
```

**全量:**
```bash
PYTHONPATH=. bash -c '
eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_ct_artifact_restoration.py \
    --planner llm \
    --llm-model qwen/qwen-2.5-vl-72b-instruct \
    --llm-base-url https://openrouter.ai/api/v1 \
    --output-dir /home/liuxinyao/output/ctagent/eval/restoration_llm_full \
    --gpu 1 \
    --all-slices \
    --no-save-images' \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/restoration_llm_full_$(date +%Y%m%d_%H%M%S).log
```

> LLM 模式每个 (slice × 5 artifacts × 3 severities) = 15 次 API 调用。全量 22509 slices ≈ 337k 次调用, 注意费用。

### 4.6 VLM API 诊断评估 (Case-level)

```bash
PYTHONPATH=. python scripts/run_cq500_api_eval.py \
    --config configs/experiment/cq500_api_eval.yaml \
    --max-cases 50 \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/cq500_api_eval_$(date +%Y%m%d_%H%M%S).log
```

**参数**:
- `--max-cases N`: 限制 case 数量
- `--input-types clean`: 只评测干净图
- `--input-types clean degraded restored`: 三路对比

**输出**: `results/cq500_api_eval/{predictions.jsonl, summary.csv, summary.json}`

### 4.7 VLM API 诊断评估 (Per-slice)

```bash
PYTHONPATH=. python scripts/run_cq500_api_eval.py \
    --config configs/experiment/cq500_api_eval.yaml \
    --per-slice \
    --slice-labels results/slice_labels/slice_labels.csv \
    --bhx-only \
    --max-cases 10 \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/cq500_per_slice_eval_$(date +%Y%m%d_%H%M%S).log
```

**额外参数**:
- `--per-slice`: 启用逐 slice 评估 (1 API call / slice)
- `--slice-labels PATH`: 指定 slice 标签文件
- `--bhx-only`: 只评估 BHX 覆盖的 patient

### 4.8 DnCNN 训练

```bash
PYTHONPATH=. bash -c '
eval "$(conda shell.bash hook)" && conda activate llamafactory && \
CUDA_VISIBLE_DEVICES=1 python -u scripts/train_dncnn.py' \
    2>&1 | tee /home/liuxinyao/output/ctagent/eval/train_dncnn_$(date +%Y%m%d_%H%M%S).log
```

**输出**: `checkpoints/dncnn_ct.pth`

---

## 5. Pipeline 流程图

```
                        ┌────────────────────────────┐
                        │   CQ500 原始 DICOM 数据     │
                        └─────────────┬──────────────┘
                                      │ build_sop_index.py
                                      ▼
                        ┌────────────────────────────┐
                        │  cq500_processed (gt.h5)    │
                        └─────────────┬──────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ 伪影生成 & 修复   │  │ 逐 Slice 标签生成 │  │ VLM API 诊断评估  │
   │ (eval_ct_artifact │  │ (build_slice_    │  │ (run_cq500_api_  │
   │  _restoration.py) │  │  labels.py)      │  │  eval.py)        │
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            │                     │                      │
            ▼                     ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ Detect → Plan    │  │ slice_labels.csv │  │ predictions.jsonl│
   │ → Execute        │  │ slice_labels.jsonl│  │ summary.csv      │
   │                  │  │ stats.json       │  │ summary.json     │
   │ Planner 模式:     │  └──────────────────┘  └──────────────────┘
   │ · rule (规则)     │
   │ · llm  (LLM引导)  │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ evaluation_      │
   │ summary.json     │
   │ + *.png 可视化    │
   └──────────────────┘
```

### 5.1 修复 Pipeline 详细流程

```
退化图像 ──┬─── [Rule 模式] ──→ DegradationDetector ──→ RuleBasedPlanner ───┐
           │                    (阈值匹配)              (硬编码映射表)       │
           │                                                                │
           └─── [LLM 模式] ──→ AnalysisTool ──→ PlannerCaller ──→ LLM API ─┤
                                PerceptionTool    (构建 prompt)    (Qwen VL) │
                                StatisticsTool                               │
                                                                            ▼
                                                                      Plan (工具链)
                                                                            │
                                                                            ▼
                                                              RestorationTool.apply_chain
                                                              (逐步执行, PSNR/SSIM 安全回滚)
                                                                            │
                                                                            ▼
                                                                       修复后图像
```

---

## 6. 可用伪影类型 & 修复工具

### 6.1 伪影类型

| 类型 | 英文 | 严重度 | 仿真方式 | 可恢复性 |
|------|------|--------|----------|----------|
| 环状伪影 | ring | mild/moderate/severe | 极坐标空间径向条纹注入 | 高 |
| 运动伪影 | motion | mild/moderate/severe | 投影角度偏移 + 模糊 | 中 |
| 束硬化伪影 | beam_hardening | mild/moderate/severe | 多项式 BHC 反向注入 | 高 |
| 散射伪影 | scatter | mild/moderate/severe | 低频散射分量叠加 | 中-高 |
| 截断伪影 | truncation | mild/moderate/severe | FOV 截断 + 重建 | 中 |
| 低剂量噪声 | low_dose | mild/moderate/severe | Poisson 噪声重采样 | 高 |
| 稀疏角采样 | sparse_view | mild/moderate/severe | 均匀角度欠采样 | 低 (病态) |
| 有限角采样 | limited_angle | mild/moderate/severe | 角度范围截断 | 很低 (病态) |
| 焦点模糊 | focal_spot_blur | mild/moderate/severe | 探测器/焦点 PSF 模糊 | 中-高 |
| 金属伪影 | metal (MAR) | 10 个掩模 | 金属 mask 前向投影 | 中 |

### 6.2 修复工具清单

| 工具名 | 类型 | 适用伪影 |
|--------|------|----------|
| `ring_removal_polar` | 经典 | ring (mild) |
| `ring_removal_wavelet` | 经典 | ring (moderate/severe) |
| `motion_correction_tv` | 经典 | motion (moderate/severe) |
| `motion_correction_wiener` | 经典 | motion (mild) |
| `bhc_flatfield` | 经典 | beam_hardening (mild/moderate) |
| `bhc_polynomial` | 经典 | beam_hardening (severe) |
| `scatter_correction_detrend` | 经典 | scatter |
| `scatter_correction_clahe` | 经典 | scatter (severe, 辅助) |
| `truncation_correction_extrapolate` | 经典 | truncation |
| `truncation_correction_tv` | 经典 | truncation (severe, 辅助) |
| `denoise_dncnn` | 深度学习 | 通用降噪, low_dose, 后处理 (最佳 PSNR/SSIM) |
| `denoise_tv` | 经典 | sparse_view/limited_angle 条纹抑制, low_dose 强降噪 |
| `denoise_bilateral` / `nlm` / `wavelet` / `bm3d` | 经典 | 通用降噪 |
| `deblur_richardson_lucy` | 经典 | focal_spot_blur (RL 反卷积) |
| `clip_extreme` + `inpaint_biharmonic` | 经典 | metal 前置处理 |
| `sharpen_usm` / `enhance_laplacian` | 经典 | 模糊/低分辨率/focal_spot_blur |

---

## 7. 安全机制

| 机制 | 描述 |
|------|------|
| **Do-no-harm 门控** | 仅检测到 mild 通用退化(noise/blur) + 无特定伪影时，跳过修复 |
| **SSIM 安全回滚** | 单步 SSIM 下降 > 0.03 → 自动 revert |
| **PSNR 安全回滚** | 单步 PSNR 下降 > 2.0 dB → 自动 revert |
| **工具参数 clip** | LLM 输出的参数自动 clip 到合法范围 |
| **工具名校验** | 仅允许白名单中的工具名 |

---

## 8. 环境信息

| 项目 | 值 |
|------|-----|
| Conda 环境 | `llamafactory` |
| GPU | GPU 1 (`CUDA_VISIBLE_DEVICES=1`) |
| CT 仿真后端 | `astra_cuda` (ASTRA-CUDA + ODL) |
| LLM API | OpenRouter (`qwen/qwen-2.5-vl-72b-instruct`) |
| Workspace 文件 | `/home/liuxinyao/project/ctagent/ctagent.code-workspace` |
