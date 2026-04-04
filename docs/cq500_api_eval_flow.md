# CQ500 API Evaluation Pipeline - Flow Chart

## Overview

This document describes the end-to-end flow of `scripts/run_cq500_api_eval.py`,
which evaluates Vision Language Models (VLMs) on CT head scan diagnosis using the CQ500 dataset.

## Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                     1. CONFIGURATION                             │
│                                                                  │
│  CLI: python scripts/run_cq500_api_eval.py                      │
│         --config configs/experiment/cq500_api_eval.yaml          │
│         --max-cases N                                            │
│                          │                                       │
│                          ▼                                       │
│              Load YAML config file                               │
│    (data paths, LLM params, windowing, eval settings)            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
┌─────────────────┐ ┌───────────────┐ ┌──────────────────┐
│ 2. LOAD LABELS  │ │ 3. BUILD      │ │ 4. CREATE LLM    │
│                 │ │    MANIFEST   │ │    CLIENT         │
│ reads.csv       │ │               │ │                   │
│ (3 radiologist  │ │ Scan          │ │ Provider:         │
│  annotations)   │ │ cq500_        │ │  openai/anthropic │
│       │         │ │ processed/    │ │       │           │
│       ▼         │ │       │       │ │       ▼           │
│ Majority vote   │ │       ▼       │ │ create_client()   │
│ (≥2 agree → 1) │ │ Find gt.h5 &  │ │ → OpenAIClient or │
│       │         │ │ {mask}.h5     │ │   AnthropicClient │
│       ▼         │ │       │       │ │                   │
│ gt_labels       │ │       ▼       │ │                   │
│ (10 binary      │ │ Select middle │ │                   │
│  labels per     │ │ slices (up to │ │                   │
│  case)          │ │ 5 per case)   │ │                   │
│                 │ │       │       │ │                   │
│ 10 Labels:      │ │       ▼       │ │                   │
│ ICH, IPH, IVH,  │ │ Filter to     │ │                   │
│ SDH, EDH, SAH,  │ │ labeled cases │ │                   │
│ Fracture,       │ │       │       │ │                   │
│ CalvarialFx,    │ │       ▼       │ │                   │
│ MassEffect,     │ │ list[EvalCase]│ │                   │
│ MidlineShift    │ │               │ │                   │
└────────┬────────┘ └───────┬───────┘ └────────┬─────────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│               5. BATCH EVALUATION LOOP                           │
│                                                                  │
│  For each EvalCase (limited by --max-cases):                     │
│    For each input_type in [clean, degraded]:                     │
│                                                                  │
│    ┌──────────────────────────────────────────────────────┐      │
│    │  a) IMAGE ENCODING                                   │      │
│    │                                                      │      │
│    │  Load HDF5 file:                                     │      │
│    │    clean → gt.h5["image"]                            │      │
│    │    degraded → {mask_idx}.h5["ma_CT"]                 │      │
│    │         │                                            │      │
│    │         ▼                                            │      │
│    │  μ (linear attenuation) → HU conversion              │      │
│    │    HU = (μ / μ_water - 1.0) × 1000                  │      │
│    │         │                                            │      │
│    │         ▼                                            │      │
│    │  CT Windowing (center=40, width=80)                  │      │
│    │    normalized = (HU - center + width/2) / width      │      │
│    │         │                                            │      │
│    │         ▼                                            │      │
│    │  uint8 [0-255] → PIL Image → PNG → base64            │      │
│    └──────────┬───────────────────────────────────────────┘      │
│               │                                                  │
│               ▼                                                  │
│    ┌──────────────────────────────────────────────────────┐      │
│    │  b) VLM API CALL                                     │      │
│    │                                                      │      │
│    │  System Prompt: CQ500_DIAGNOSIS_SYSTEM_PROMPT        │      │
│    │    "You are a neuroradiologist..."                    │      │
│    │    "Classify 10 findings as 0/1 with confidence"     │      │
│    │                                                      │      │
│    │  User Message:                                       │      │
│    │    [base64_img_1, base64_img_2, ..., text_prompt]    │      │
│    │         │                                            │      │
│    │         ▼                                            │      │
│    │  POST https://openrouter.ai/api/v1/chat/completions  │      │
│    │         │                                            │      │
│    │         ▼                                            │      │
│    │  LLMResponse {text, usage, finish_reason}            │      │
│    └──────────┬───────────────────────────────────────────┘      │
│               │                                                  │
│               ▼                                                  │
│    ┌──────────────────────────────────────────────────────┐      │
│    │  c) RESPONSE PARSING                                 │      │
│    │                                                      │      │
│    │  Extract JSON from response text                     │      │
│    │    (handles markdown code blocks, loose JSON)        │      │
│    │         │                                            │      │
│    │         ▼                                            │      │
│    │  Parse predictions: {label: 0 or 1}                  │      │
│    │  Parse confidence:  {label: 0.0~1.0}                 │      │
│    │  Parse reasoning:   free text                        │      │
│    │         │                                            │      │
│    │         ▼                                            │      │
│    │  CaseResult {case_id, input_type, gt, pred,          │      │
│    │              confidence, latency, error}             │      │
│    └──────────┬───────────────────────────────────────────┘      │
│               │                                                  │
│               ▼                                                  │
│    sleep(rate_limit_sec)  →  next iteration                      │
│                                                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  6. METRIC COMPUTATION                           │
│                                                                  │
│  For each input_type (clean / degraded):                         │
│                                                                  │
│    Filter valid results (no errors, parse_success=True)          │
│         │                                                        │
│         ▼                                                        │
│    Collect y_true, y_pred, y_prob arrays                         │
│         │                                                        │
│         ├─── Per-label metrics ──────────────────────┐           │
│         │    TP, TN, FP, FN                          │           │
│         │    Accuracy = (TP+TN) / Total              │           │
│         │    Precision = TP / (TP+FP)                │           │
│         │    Recall = TP / (TP+FN)                   │           │
│         │    F1 = 2×P×R / (P+R)                      │           │
│         │    Specificity = TN / (TN+FP)              │           │
│         │                                            │           │
│         ├─── Aggregate metrics ──────────────────────┤           │
│         │    Macro-F1 = mean(per-label F1)           │           │
│         │    Micro-F1 = global TP/FP/FN based F1     │           │
│         │    Mean Accuracy = mean(per-label acc)     │           │
│         │    AUROC (requires sklearn)                │           │
│         │    Mean Latency                            │           │
│         │                                            │           │
│         └────────────────────────────────────────────┘           │
│                                                                  │
│  If both clean & degraded available:                             │
│    Degraded Drop % = (clean_acc - degraded_acc) / clean_acc × 100│
│    (Measures impact of metal artifacts on model performance)     │
│                                                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                  7. OUTPUT & RESULTS                             │
│                                                                  │
│  results/cq500_api_eval/                                         │
│    │                                                             │
│    ├── summary.json      Aggregated metrics per input_type       │
│    ├── summary.csv       Tabular metrics for comparison          │
│    ├── predictions.jsonl Per-case predictions (one JSON/line)    │
│    └── case_studies.json Interesting cases for manual review     │
│                                                                  │
│  Console Output:                                                 │
│    [clean]    accuracy  macro_F1  micro_F1  auroc  latency       │
│    [degraded] accuracy  macro_F1  micro_F1  auroc  latency       │
│    Degraded drop: X%                                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Majority Voting**: 3 radiologists annotate each case; label = 1 if ≥2 agree
2. **Slice Sampling**: Selects up to 5 slices from the middle 50% of the series (most diagnostically relevant region)
3. **Dual Evaluation**: Tests both `clean` (original) and `degraded` (metal artifact) CT images to measure artifact robustness
4. **CT Windowing**: Uses brain window (center=40 HU, width=80 HU) for optimal soft tissue contrast
5. **Rate Limiting**: Configurable delay between API calls to respect provider limits

## Module Dependency Graph

```
scripts/run_cq500_api_eval.py
    │
    ├── configs/experiment/cq500_api_eval.yaml
    │
    ├── eval/cq500_labels.py        ← reads.csv
    │
    ├── eval/cq500_manifest.py      ← cq500_processed/ directory
    │
    ├── eval/cq500_api_eval.py      ← core evaluation logic
    │       │
    │       ├── llm/api_client.py   ← OpenAI / Anthropic API
    │       ├── llm/prompt_builder.py
    │       └── llm/response_parser.py
    │
    └── eval/metrics.py             ← multilabel classification metrics
```
