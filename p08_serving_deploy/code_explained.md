# p08 代码详解 - 推理部署

> 逐文件解释推理部署模块的核心代码实现。

---

## 1. config.py — 三框架统一配置

### 配置架构设计

```python
@dataclass
class ServingConfig:          # 通用服务配置
@dataclass
class VLLMConfig:             # vLLM 专用配置
@dataclass
class SGLangConfig:           # SGLang 专用配置
@dataclass
class OllamaConfig:           # Ollama 专用配置
@dataclass
class BenchmarkConfig:        # 压力测试配置
```

**设计原则**：
- **通用参数**（端口、模型路径、生成参数）放在 `ServingConfig`
- **框架专用参数**（PagedAttention 块大小、RadixAttention 策略）放在各自配置
- **GPU Preset**：24G/48G 两套配置，自动适配显存

### VLLMConfig 关键参数

```python
gpu_memory_utilization: float = 0.90     # 显存利用率
# 0.90 = 给 KV Cache 保留 90% 的可用显存
# 越大 → 支持更多并发 → 但 OOM 风险增大
# 建议: 生产环境 0.85-0.90, 开发环境 0.80

block_size: int = 16                      # KV Cache 块大小
# 每个 block 存储 16 个 token 的 KV
# 16 是 vLLM 的默认值, 也可以设为 32 (减少页表开销)

max_num_seqs: int = 256                   # 最大并发序列
# continuous batching 的上限
# 24G 建议 64-128, 48G 建议 128-256

enable_chunked_prefill: bool = True       # 分块 prefill
# 将长 prompt 的 prefill 拆成多个 chunk
# 与 decode 请求混合调度, 降低长 prompt 的 TTFT
```

### BenchmarkConfig 并发级别

```python
concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32]
# 从低到高测试并发，观察性能拐点
# 拐点之前: 吞吐线性增长, 延迟基本不变
# 拐点之后: 吞吐增长放缓, 延迟显著增大
```

---

## 2. serve_vllm.py — vLLM 服务启动

### 命令构建逻辑

```python
def build_vllm_command(args) -> list:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model or config.model_path,
        "--port", str(args.port or config.vllm_port),
        "--host", "0.0.0.0",     # 监听所有网卡
    ]
```

**为什么用 subprocess 而非直接 import**：
- vLLM 的 server 启动会阻塞主进程
- 使用 subprocess 可以更好地控制进程生命周期
- 启动参数通过命令行传递，与直接 CLI 使用一致

### 关键参数映射

| 配置字段 | 命令行参数 | 作用 |
|----------|-----------|------|
| `tensor_parallel_size` | `--tensor-parallel-size` | 多卡并行 |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | 显存比例 |
| `enable_prefix_caching` | `--enable-prefix-caching` | 前缀缓存 |
| `enable_chunked_prefill` | `--enable-chunked-prefill` | 分块 prefill |
| `speculative_model` | `--speculative-model` | 投机解码 |

### Dry Run 模式

```python
if args.dry_run:
    print("  [Dry Run] 仅打印命令，不实际启动。")
    return
```

开发调试时先 `--dry-run` 确认命令正确，再实际启动。

---

## 3. serve_sglang.py — SGLang 服务启动

### SGLang 独有参数

```python
cmd.extend(["--schedule-policy", sglang_config.schedule_policy])
# lpm = Longest Prefix Match（默认）
# 优先调度与已有缓存匹配最长前缀的请求
# 最大化 RadixAttention 的缓存复用率

cmd.extend(["--mem-fraction-static", str(sglang_config.mem_fraction_static)])
# 静态显存占比: 模型参数 + 固定开销
# 剩余显存用于 KV Cache (动态分配)
```

### SGLang vs vLLM 调度差异

| 特性 | vLLM | SGLang |
|------|------|--------|
| 缓存策略 | 显式前缀缓存 | RadixAttention 自动管理 |
| 调度粒度 | 请求级 | token 级（更细粒度） |
| 数据并行 | 需外部负载均衡 | 内置 DP 支持 |

### 数据并行配置

```python
if dp > 1:
    cmd.extend(["--dp", str(dp)])
# SGLang 原生支持数据并行:
# dp=2 → 在 2 组 GPU 上各运行一个模型副本
# 配合 tp=2, 共需 4 张 GPU
# 总吞吐 ≈ 单副本 × dp
```

---

## 4. benchmark_serving.py — 异步压力测试

### 流式请求测量

```python
async def send_request(...) -> RequestResult:
    start_time = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    
    async for line in resp.content:
        # 解析 SSE 流
        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()  # 记录首 token 时间
            output_tokens += 1
    
    result.ttft = first_token_time - start_time          # TTFT
    result.tps = output_tokens / result.total_time        # TPS
```

**测量精度**：
- 使用 `time.perf_counter()` 而非 `time.time()`（纳秒级精度）
- TTFT 测量的是**客户端收到**首 token 的时间（包含网络延迟）
- TPS 是端到端的速率（包含网络传输时间）

### 并发控制

```python
semaphore = asyncio.Semaphore(concurrency)

async def bounded_request(prompt):
    async with semaphore:
        return await send_request(...)
```

**Semaphore** 限制同时活跃的请求数：
- `concurrency=4` → 最多 4 个请求同时在飞
- 确保测量的是固定并发下的性能
- 避免一次性发送所有请求导致服务过载

### 百分位计算

```python
cr.p50_ttft = sorted(ttfts)[len(ttfts) // 2]
cr.p90_ttft = sorted(ttfts)[int(len(ttfts) * 0.9)]
cr.p99_ttft = sorted(ttfts)[min(int(len(ttfts) * 0.99), len(ttfts) - 1)]
```

排序后直接取位置，简单高效。生产环境通常使用 `numpy.percentile()` 或 `t-digest` 算法。

---

## 5. inference.py — OpenAI SDK 调用

### 流式推理核心

```python
stream = client.chat.completions.create(
    model=model,
    messages=[...],
    stream=True,            # 关键: 启用流式
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

**流式 vs 非流式**：
- 非流式：等全部生成完才返回（用户等待时间 = 总生成时间）
- 流式：每生成一个 token 就返回（用户感知延迟 = TTFT）
- **推荐始终使用流式**，用户体验显著更好

### 多轮对话上下文管理

```python
messages = [{"role": "system", "content": "..."}]  # 初始化

while True:
    user_input = input("🧑 你: ")
    messages.append({"role": "user", "content": user_input})
    
    # 发送包含完整历史的 messages
    stream = client.chat.completions.create(model=model, messages=messages, ...)
    
    assistant_content = "..."  # 收集完整回复
    messages.append({"role": "assistant", "content": assistant_content})
```

**注意**：每轮都发送完整的 messages 历史。服务端如果启用了前缀缓存（prefix caching），历史部分的 KV Cache 可以复用，不需要重算。

### 批量推理并发

```python
async def batch_inference_async(url, model, prompts, max_tokens, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [process_one(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
```

- `asyncio.gather` 并发发送所有请求
- `Semaphore` 控制同时在飞的请求数
- 适合离线批量处理场景（如数据标注、批量翻译）
