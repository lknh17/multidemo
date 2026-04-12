# p08 推理部署

> **目标**: 掌握 LLM 推理部署全流程，对比 vLLM / SGLang / Ollama 三大框架的性能，理解 PagedAttention、Continuous Batching、RadixAttention 等核心优化技术。
>
> **前置条件**: 完成 p07 模型量化，已有可部署的模型。
>
> **预计耗时**: 3-5 小时（含部署、调优、压测）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 理解自回归推理的计算瓶颈（memory-bound vs compute-bound）
- 掌握 KV Cache 原理及 PagedAttention 的显存优化
- 使用 vLLM 部署 OpenAI 兼容 API 服务
- 使用 SGLang 体验 RadixAttention 的前缀复用加速
- 使用 Ollama 进行本地离线部署
- 进行多级并发压力测试，分析 TTFT / TPS / 吞吐量
- 理解 Continuous Batching 和 Speculative Decoding 原理

### 本模块代码文件概览

本模块从"训练好的模型"到"可用的 API 服务"：**三种框架部署 → 压力测试 → 性能调优 → API 调用**。通过实际部署和压测，理解推理优化的核心技术。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | 部署配置（端口、显存利用率、并发参数等） | 理解推理服务的关键参数 |
| `serve_vllm.py` | vLLM 服务部署脚本（PagedAttention + Continuous Batching）| 掌握高性能推理部署 |
| `serve_sglang.py` | SGLang 服务部署脚本（RadixAttention 前缀复用） | 了解前缀缓存加速 |
| `serve_ollama.sh` | Ollama 本地部署脚本（GGUF 格式，支持 CPU） | 掌握本地离线部署 |
| `benchmark_serving.py` | 多级并发压力测试，输出 TTFT/TPS/吞吐量指标 | 学会评估推理服务性能 |
| `inference.py` | API 客户端：流式推理、多轮对话、批量推理 | 掌握服务调用方式 |

> 💡 **建议学习顺序**: `serve_vllm.py` 启动服务 → `inference.py` 验证 → `benchmark_serving.py` 压测 → 再试 `serve_sglang.py` 和 `serve_ollama.sh` 对比

### 确认前置条件

```bash
cd p08_serving_deploy

# 检查 vLLM 安装
python -c "import vllm; print(f'vLLM 版本: {vllm.__version__}')"

# 检查模型存在
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',trust_remote_code=True); print(f'Tokenizer OK: {t.vocab_size}')"
```

---

## 2. 推理框架对比

| 特性 | vLLM | SGLang | Ollama |
|------|------|--------|--------|
| 核心优化 | PagedAttention | RadixAttention | llama.cpp (GGUF) |
| 目标场景 | 高吞吐服务 | 复杂 LLM 程序 | 本地/边缘部署 |
| 并行方式 | Tensor Parallel | TP + DP | CPU/GPU 混合 |
| 量化支持 | AWQ, GPTQ, FP8 | AWQ, GPTQ | GGUF (Q4/Q5/Q8) |
| API 兼容 | OpenAI | OpenAI | OpenAI |
| 多轮加速 | 前缀缓存 | 自动前缀复用 | 上下文缓存 |
| 易用性 | ★★★★ | ★★★ | ★★★★★ |
| 性能 | ★★★★★ | ★★★★★ | ★★★ |

---

## 3. vLLM 部署

### 3.1 启动服务

```bash
# 方式一：使用脚本启动
python serve_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct

# 方式二：直接使用 vLLM CLI
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --enable-chunked-prefill

# 方式三：多卡张量并行
python serve_vllm.py --tensor-parallel 2
```

### 3.2 配置说明

```python
# config.py 中的 vLLM 关键参数
gpu_memory_utilization = 0.90     # 显存利用率（越高吞吐越大，但留余量防 OOM）
block_size = 16                    # KV Cache 页大小
max_num_seqs = 256                 # 最大并发序列
enable_chunked_prefill = True      # 分块 prefill（降低首 token 延迟）
enable_prefix_caching = True       # 前缀缓存（多轮对话加速）
```

### 3.3 验证服务

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'
```

---

## 4. SGLang 部署

### 4.1 启动服务

```bash
# 使用脚本
python serve_sglang.py --model Qwen/Qwen2.5-0.5B-Instruct

# 直接使用 SGLang
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 8001 \
    --mem-fraction-static 0.88
```

### 4.2 RadixAttention 优势

RadixAttention 自动检测请求之间的**公共前缀**，复用已计算的 KV Cache：

| 场景 | 普通推理 | RadixAttention |
|------|---------|---------------|
| 多轮对话 | 每轮重算全部 KV | 只算新增部分 |
| Few-shot 推理 | 每次重算示例 | 示例 KV 共享 |
| 同 system prompt | 每次重算 | 自动复用 |

---

## 5. Ollama 本地部署

```bash
# 一键部署
bash serve_ollama.sh

# 自定义参数
bash serve_ollama.sh --gguf models/my-model.gguf --tag my-model --ctx 4096
```

Ollama 使用 GGUF 格式模型，适合：
- 开发调试（无需 GPU）
- 边缘设备部署
- 离线环境使用

---

## 6. 压力测试

### 6.1 运行测试

```bash
# 测试 vLLM 服务
python benchmark_serving.py --url http://localhost:8000/v1

# 测试 SGLang 服务
python benchmark_serving.py --url http://localhost:8001/v1

# 自定义并发级别
python benchmark_serving.py --concurrency 1,4,8,16,32 --num-requests 100
```

### 6.2 关键指标解读

| 指标 | 含义 | 优化方向 |
|------|------|---------|
| TTFT | 首 token 延迟 | chunked prefill, 前缀缓存 |
| TPS | 每秒生成 token 数 | 批处理优化, 硬件加速 |
| 吞吐量 | 每秒完成请求数 | continuous batching |
| P90 延迟 | 90% 请求的延迟上界 | 负载均衡, 队列调度 |

### 6.3 预期结果（0.5B 模型，单卡 4090）

| 并发 | TTFT | TPS | 吞吐量 |
|------|------|-----|--------|
| 1 | ~0.05s | ~80 | ~2 req/s |
| 4 | ~0.08s | ~70 | ~6 req/s |
| 16 | ~0.15s | ~50 | ~15 req/s |
| 32 | ~0.3s | ~35 | ~20 req/s |

---

## 7. API 客户端使用

```bash
# 流式推理
python inference.py --mode stream --prompt "解释什么是KV Cache"

# 多轮对话
python inference.py --mode chat

# 批量推理
python inference.py --mode batch --concurrency 8
```

### Python SDK 调用示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# 流式输出
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "什么是PagedAttention?"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## 8. 性能调优建议

### 8.1 显存优化

| 方法 | 效果 | 适用场景 |
|------|------|---------|
| 量化 (AWQ/GPTQ) | 显存减半 | 大模型部署 |
| PagedAttention | KV Cache 利用率 >95% | 高并发场景 |
| Chunked Prefill | 降低 TTFT | 长 prompt 场景 |
| Tensor Parallel | 显存按卡分摊 | 多卡环境 |

### 8.2 吞吐优化

- **增大 max_num_seqs**: 允许更多并发（但 TTFT 会增大）
- **增大 gpu_memory_utilization**: 更多显存给 KV Cache
- **启用 Continuous Batching**: 动态插入新请求
- **Speculative Decoding**: 用小模型加速生成

### 8.3 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| OOM | 显存不足 | 降低 max_model_len / 启用量化 |
| TTFT 过高 | prefill 计算量大 | 启用 chunked prefill |
| 吞吐瓶颈 | 显存被 KV Cache 占满 | 增大 swap_space / 用 PagedAttention |
| 精度下降 | 量化损失 | 使用更高位量化 (Q8 > Q4) |

---

## 小结

- [x] 使用 vLLM 部署了高性能推理服务
- [x] 使用 SGLang 体验了 RadixAttention 加速
- [x] 使用 Ollama 进行了本地离线部署
- [x] 完成了多级并发压力测试
- [x] 理解了 PagedAttention、Continuous Batching 等核心优化
- [x] 掌握了推理服务性能调优方法

**本模块完成！** 你已经掌握了从模型训练到部署上线的完整流程。
