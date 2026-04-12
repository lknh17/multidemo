# p01 代码详解 - 逐文件逐函数解析

> 本文档是 p01 模块的代码参考手册，逐文件解释每个函数的实现细节、关键参数含义和常见修改方式。

---

## 1. config.py — GPU Preset 自动检测与全局配置

### 核心设计

`config.py` 是整个实践系列的**全局配置中心**。它用 Python `@dataclass` 组织配置项，支持 GPU 自动检测。

### `detect_gpu_preset()` 函数

```python
def detect_gpu_preset() -> str:
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    if vram_gb >= 40:
        return "GPU_48G"
    else:
        return "GPU_24G"
```

**工作原理**:
- `torch.cuda.get_device_properties(0)` 获取第 0 号 GPU 的属性
- `total_mem` 返回显存总量（单位：字节），除以 $1024^3$ 转为 GB
- 40GB 为分界线：RTX 4090 (24G) 走 `GPU_24G`，A6000 (48G) 走 `GPU_48G`

**为什么用 40 而不是 32？** 因为 A100 40GB 版本的实际显存是 ~39.4 GB，用 40 作为分界可以把它正确归为 48G 类别。

### `EnvConfig` 数据类

```python
@dataclass
class EnvConfig:
    base_model_name: str = "Qwen/Qwen2.5-0.5B"
    hf_mirror: str = "https://hf-mirror.com"
    use_mirror: bool = True
```

**HF Mirror 机制**: 
- `hf-mirror.com` 是 HuggingFace Hub 的国内镜像
- 通过设置 `HF_ENDPOINT` 环境变量，所有 `from_pretrained` 和 `snapshot_download` 自动走镜像
- `__post_init__` 中自动设置，用户无需手动操作

**常见修改**:
- 海外服务器设 `use_mirror=False`
- 要用更大模型，改 `base_model_name` 为 `"Qwen/Qwen2.5-1.5B"`
- 自定义缓存路径，改 `model_cache_dir`

### `VRAMEstimatorConfig` 和 `BenchmarkConfig`

这两个配置类用 `field(default_factory=lambda: [...])` 来定义列表默认值。

**为什么不能直接写 `list = [...]`？** 因为 Python `@dataclass` 中，可变对象（list/dict）不能作为默认值（会被所有实例共享）。`default_factory` 确保每个实例创建独立的列表。

---

## 2. setup.sh — 一键安装脚本

### 脚本结构

```bash
set -e  # 遇到任何错误立即退出
```

`set -e` 是 shell 脚本的最佳实践。如果某个 `pip install` 失败，脚本会立即停止，而不是继续执行后续步骤导致环境不完整。

### PyTorch 安装的 `--index-url`

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**为什么要指定 index-url？**
- 默认的 PyPI 上只有 CPU 版 PyTorch
- 要安装 CUDA 版，必须指定 PyTorch 的官方索引
- `cu118` 表示 CUDA 11.8 版本（兼容性最好）

**版本锁定策略**: `torch==2.4.0` 用精确版本号，其他包用 `>=` 最低版本。PyTorch 版本变化可能导致 API 不兼容，所以锁定精确版本；其他包向后兼容性好，用最低版本即可。

### Flash Attention 的特殊处理

```bash
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "⚠️ Flash Attention 安装失败"
}
```

- `--no-build-isolation`: Flash Attention 的编译需要链接已安装的 PyTorch，禁用构建隔离才能找到 PyTorch 的 CUDA 头文件
- `2>/dev/null`: 隐藏编译过程中的大量 warning
- `|| { ... }`: 如果安装失败，打印警告但不退出脚本（Flash Attention 是可选的）

**常见安装失败原因**:
1. GPU 架构太老（需要 SM 8.0+，即 Ampere 以上）
2. 缺少编译工具（`gcc`、`nvcc`）
3. CUDA 版本不匹配

---

## 3. download_model.py — 模型下载与镜像加速

### `snapshot_download` vs `from_pretrained`

```python
from huggingface_hub import snapshot_download
local_path = snapshot_download(repo_id=model_name, cache_dir=cache_dir, resume_download=True)
```

| 方法 | 作用 | 显存需求 | 断点续传 |
|------|------|---------|---------|
| `snapshot_download` | 只下载文件到本地 | 0（不加载模型） | ✅ |
| `from_pretrained` | 下载并加载到内存 | 模型大小 | ❌ |

**推荐使用 `snapshot_download` 下载**，因为：
- 7B 模型的权重约 14GB，`from_pretrained` 会直接加载到内存/显存，可能 OOM
- `snapshot_download` 只下载文件，不占运行内存
- 支持 `resume_download=True` 断点续传

### `show_model_info()` — 不加载权重获取模型信息

```python
cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
```

`AutoConfig` 只加载 `config.json`（几 KB），不加载模型权重（几 GB），可以在没有 GPU 的情况下查看模型配置。

**参数量估算公式**（基于配置推算）:

$$\text{参数量} \approx L \times (4d^2 + 3d \times d_{ff}) + V \times d$$

其中 $L$ 是层数，$d$ 是隐藏维度，$d_{ff}$ 是 FFN 中间维度，$V$ 是词表大小。

---

## 4. verify_env.py — 环境验证与兼容性检查

### `CheckResult` 类设计

```python
class CheckResult:
    def __init__(self, name, status, detail, suggestion=""):
```

每项检查返回一个 `CheckResult`，包含：
- `status`: `"PASS"` / `"WARN"` / `"FAIL"` 三级
- `suggestion`: 失败时的修复命令

这种设计让检查结果可以统一收集、汇总统计。

### GPU 架构检测

```python
capability = torch.cuda.get_device_capability(0)  # 返回 (major, minor)
```

`capability` 是 GPU 的计算能力版本号，决定了支持哪些特性：

| Capability | 架构 | 代表 GPU | bf16 | Flash Attention |
|-----------|------|---------|------|-----------------|
| 7.0 | Volta | V100 | ❌ | ❌ |
| 7.5 | Turing | T4/RTX 2080 | ❌ | ❌ |
| 8.0 | Ampere | A100 | ✅ | ✅ |
| 8.6 | Ampere | RTX 3090/A6000 | ✅ | ✅ |
| 8.9 | Ada | RTX 4090 | ✅ | ✅ |
| 9.0 | Hopper | H100 | ✅ | ✅ |

### 库版本检查

```python
from packaging.version import Version
if Version(version) < Version(min_version):
```

使用 `packaging.version.Version` 做版本比较，比字符串比较准确。例如 `"4.9.0" < "4.45.0"` 字符串比较会错误地认为 `"4.9"` 更大。

---

## 5. vram_estimator.py — 显存估算公式实现

### 核心：四部分显存分别估算

```python
param_mem = estimate_model_memory(num_params_billion, dtype)       # 参数
grad_mem = estimate_gradient_memory(num_params_billion, dtype)     # 梯度
optim_mem = estimate_optimizer_memory(num_params_billion, optimizer) # 优化器
act_mem = estimate_activation_memory(...)                           # 激活值
```

每个函数独立计算一部分，最后加总。这种设计让用户可以清楚看到每部分的贡献。

### 激活值估算的两种模式

```python
if gradient_checkpointing:
    activation_bytes = 2.0 * B * S * H * math.sqrt(L) * bytes_per_elem
else:
    activation_bytes = 34.0 * B * S * H * L * bytes_per_elem
```

- **无 Checkpointing**: 系数 34 来自 Megatron-LM 论文的分析，包括每层的 QKV 投影、注意力矩阵、FFN 中间结果等
- **有 Checkpointing**: 只保存 $\sqrt{L}$ 个 checkpoint 点的激活值，其余反向传播时重新计算

**常见修改**: 如果估算值和实际差距较大，调整系数。不同模型架构（如 MoE）的激活值模式不同。

### ZeRO Stage 调整

```python
def estimate_zero_memory(param_memory, gradient_memory, optimizer_memory, zero_stage, num_gpus):
    if zero_stage == 3:
        return {"params": param_memory / n, "gradients": gradient_memory / n, "optimizer": optimizer_memory / n}
```

ZeRO 的分片是**除以 GPU 数量**。单卡时 `num_gpus=1`，除以 1 等于没分片。所以 ZeRO 在单卡上的收益来自**内存管理优化**而非分片。

---

## 6. gpu_benchmark.py — GPU 性能测试实现

### 矩阵乘法 TFLOPS 计算

```python
flops_per_op = 2.0 * M * N * K
tflops = flops_per_op * test_iters / elapsed / 1e12
```

**为什么是 2×M×N×K？**

矩阵 $C_{M \times N} = A_{M \times K} \times B_{K \times N}$，输出矩阵每个元素需要：
- K 次乘法
- K-1 次加法 ≈ K 次加法

总共约 $2 \times M \times N \times K$ 次浮点运算。

### 预热的重要性

```python
# 预热（GPU 需要"暖机"才能达到最高频率）
for _ in range(warmup_iters):
    torch.matmul(A, B)
torch.cuda.synchronize()
```

**为什么需要预热？**
1. GPU 有**动态频率调节**：空闲时降频省电，计算时升频
2. CUDA 的 **JIT 编译**：首次运行某个 kernel 需要编译
3. **缓存预热**：GPU 的 L2 cache 需要时间填充

**`torch.cuda.synchronize()`**: GPU 操作是异步的，必须调用 `synchronize()` 等待所有 CUDA 操作完成，否则测量的时间不准确。

### Flash Attention 对比测试

```python
from torch.nn.functional import scaled_dot_product_attention
out = scaled_dot_product_attention(q, k, v, is_causal=True)
```

PyTorch 2.0+ 内置了 `scaled_dot_product_attention`，它会自动选择最优的后端：
1. Flash Attention（最快，需要支持的 GPU）
2. Memory-Efficient Attention（中等）
3. 数学实现（最慢，兼容性最好）

---

## 7. baseline_eval.py — 基线测评设计

### 评测 Prompt 设计思路

```python
EVAL_PROMPTS = {
    "中文知识": [...],
    "对话能力": [...],
    "数学推理": [...],
    "代码生成": [...],
    "安全性": [...],
}
```

评测 prompt 覆盖 5 个维度，每个维度 2-3 个 prompt。设计原则：

1. **固定不变**: 贯穿 p01-p10 全流程使用同一套 prompt
2. **简单明确**: 每个 prompt 的预期答案是确定的（便于对比）
3. **覆盖关键能力**: 知识/对话/推理/代码/安全 五大维度
4. **包含"故意刁难"**: 安全性 prompt 测试模型是否会生成不当内容

### 生成参数设计

```python
outputs = model.generate(
    max_new_tokens=256,      # 最多生成 256 个 token
    temperature=0.7,         # 适中的随机性
    top_p=0.9,              # 核采样
    do_sample=True,         # 开启采样
    pad_token_id=tokenizer.eos_token_id,  # 防止 padding 警告
)
```

**`pad_token_id=tokenizer.eos_token_id`**: 很多模型的 tokenizer 没有设置 `pad_token_id`，生成时会报 warning。将 `pad_token_id` 设为 `eos_token_id` 是常见的 workaround。

### 结果保存为 JSON

```python
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
```

`ensure_ascii=False`: 必须设置，否则中文会被编码为 `\uXXXX` 转义序列，人类不可读。

**常见修改**:
- 增加更多评测维度：在 `EVAL_PROMPTS` 中添加新类别
- 调整生成参数：修改 `temperature` 等参数
- 增加自动打分：接入 GPT-4 API 进行自动评估
