# 本地部署 Qwen 模型指南

## 环境现状

| 项目 | 值 |
|------|-----|
| GPU | 8× RTX 5090 32GB |
| Driver | 580.76.05 / CUDA 13.0 |
| conda 环境 | `llamafactory` |
| PyTorch | 2.7.0+cu128 |
| nvcc | 13.1（conda 已装 cuda-toolkit） |
| transformers | 5.2.0 |
| flash-attn | **未安装** |
| vllm | **未安装** |
| **空闲 GPU** | **GPU 2**（0MiB 占用），GPU 3 显存空闲但 util 100% 需确认 |

---

## 第一步：安装 flash-attention

```bash
conda activate llamafactory

# 关键：加 --no-cache-dir 解决跨文件系统链接报错
pip install flash-attn --no-build-isolation --no-cache-dir
```

编译耗时较长（约 10-30 分钟），耐心等待。

验证：
```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

> **如果编译失败**：可以尝试装预编译 wheel：
> ```bash
> # 去 https://github.com/Dao-AILab/flash-attention/releases 找匹配的 wheel
> # 需匹配：torch2.7, cu128, cp311, linux_x86_64
> pip install flash_attn-xxx.whl
> ```

---

## 第二步：安装 vLLM（推荐的推理服务框架）

```bash
pip install vllm --no-cache-dir
```

验证：
```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 第三步：下载 Qwen 模型

选择合适的模型大小（单卡 32GB 显存参考）：

| 模型 | 显存需求（fp16） | 单卡可跑 |
|------|----------------|---------|
| Qwen2.5-7B-Instruct | ~14GB | ✅ |
| Qwen2.5-14B-Instruct | ~28GB | ✅ 勉强 |
| Qwen2.5-72B-Instruct | ~144GB | ❌ 需多卡 |

```bash
# 推荐先用 7B 做最小验证
# 方式一：huggingface-cli（如果能访问 HF）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /home/liuxinyao/project/models/Qwen2.5-7B-Instruct

# 方式二：modelscope（国内推荐）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /home/liuxinyao/project/models/Qwen2.5-7B-Instruct
```

---

## 第四步：启动推理服务（二选一）

### 方案 A：vLLM 启动（推荐，OpenAI 兼容 API）

```bash
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model /home/liuxinyao/project/models/Qwen2.5-7B-Instruct \
    --served-model-name qwen2.5-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

### 方案 B：LLaMA-Factory API 模式

创建配置文件 `/home/liuxinyao/project/LLaMA-Factory/qwen_api.yaml`：

```yaml
model_name_or_path: /home/liuxinyao/project/models/Qwen2.5-7B-Instruct
template: qwen
infer_backend: vllm    # 或 huggingface
vllm_maxlen: 4096
```

启动：
```bash
CUDA_VISIBLE_DEVICES=2 llamafactory-cli api qwen_api.yaml
```

---

## 第五步：最小验证

```bash
# 测试 API 是否通
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}],
    "max_tokens": 100
  }'
```

如果返回 JSON 且包含模型回复内容，部署成功。

也可以用 Python 验证：
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")
resp = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=50,
)
print(resp.choices[0].message.content)
```

---

## 注意事项

1. **始终指定 `CUDA_VISIBLE_DEVICES=2`**，避免占用其他正在使用的 GPU
2. 启动前用 `nvidia-smi` 再次确认 GPU 2 仍然空闲
3. 如果后续要给 ctagent 项目调用，API 地址就是 `http://localhost:8000/v1`
4. 不要修改 LLaMA-Factory 源码（项目约定）
