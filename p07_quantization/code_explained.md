# p07 代码详解 - 模型量化

> 逐文件解释量化模块的核心代码实现，帮助你理解每个函数的设计思路。

---

## 1. config.py — 多方法统一配置

### 配置架构

```python
QuantBaseConfig     # 公共配置（模型、校准数据、输出）
├── GPTQConfig      # GPTQ 特有参数（bits, group_size, desc_act）
├── AWQConfig       # AWQ 特有参数（bits, zero_point, version）
├── GGUFConfig      # GGUF 特有参数（quant_types, llama_cpp_path）
├── BnBConfig       # bitsandbytes 特有参数（quant_type, double_quant）
└── BenchmarkConfig # 评测参数（warmup, runs, 报告路径）
```

**设计要点**:
- `QuantBaseConfig` 管理所有方法共享的参数（模型名、校准数据、种子等）
- 每种方法有独立的 `dataclass`，参数名与库的 API 一一对应
- `create_config_24g/48g` 提供针对不同显存的 preset
- 校准样本数默认 128，这是 GPTQ/AWQ 论文中推荐的值

---

## 2. quantize_gptq.py — GPTQ 量化实现

### 校准数据准备

```python
def prepare_calibration_data(tokenizer, dataset_name, n_samples, seq_length):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]
    
    calibration_data = []
    for text in selected_texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_length)
        if tokens.input_ids.shape[1] >= seq_length // 2:
            calibration_data.append(tokens.input_ids)
```

**关键细节**:
1. 过滤过短文本（< 50 字符），避免空白和标题行
2. 只保留长度 >= seq_length/2 的样本，确保校准数据有足够的上下文
3. 返回 `input_ids` tensor 列表，这是 `auto-gptq` 要求的格式

### 量化核心流程

```python
quantize_config = BaseQuantizeConfig(
    bits=4,           # 4-bit 量化
    group_size=128,   # 每 128 权重一组
    desc_act=True,    # 按激活值排序（更精确）
    sym=True,         # 对称量化
    damp_percent=0.01 # Hessian 正则化
)

model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
model.quantize(calibration_data)  # 执行逐层量化
model.save_quantized(output_dir)  # 保存量化模型
```

- `damp_percent=0.01`: 给 Hessian 对角线加阻尼 $H' = H + 0.01 \cdot \text{diag}(H)$，防止数值不稳定
- `desc_act=True` 让量化从最重要的列开始，通常 PPL 降低 0.1-0.3

---

## 3. quantize_awq.py — AWQ 量化实现

### AWQ 的简洁 API

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_name)
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,      # 非对称量化
        "q_group_size": 128,     # 分组大小
        "w_bit": 4,              # 4-bit
        "version": "GEMM",       # 使用 GEMM 内核
    },
    calib_data="wikitext",       # 校准数据集
    n_samples=128,
    seqlen=512,
)
```

**AWQ vs GPTQ API 差异**:
- AWQ 自动处理校准数据（传数据集名即可），GPTQ 需手动准备
- AWQ 的 `version="GEMM"` 表示用 GEMM 矩阵乘法内核，适合 batch >= 1
- AWQ 的 `version="GEMV"` 用 GEMV 内核，适合 batch=1 的低延迟场景

---

## 4. quantize_gguf.py — GGUF 转换流程

### 三步转换

```python
# Step 1: HF 模型 → GGUF FP16
subprocess.run([
    sys.executable, "convert_hf_to_gguf.py",
    model_name,
    "--outfile", "model-fp16.gguf",
    "--outtype", "f16",
])

# Step 2: GGUF FP16 → 量化 GGUF
subprocess.run([
    "llama-quantize",       # llama.cpp 编译的二进制
    "model-fp16.gguf",      # 输入
    "model-q4_k_m.gguf",   # 输出
    "Q4_K_M",              # 量化级别
])
```

**为什么分两步**:
1. `convert_hf_to_gguf.py` 将 HuggingFace 格式（safetensors）转为 GGUF 的张量布局
2. `llama-quantize` 是 C++ 实现的量化器，直接操作二进制数据，速度极快

**批量量化**: `--all-types` 参数会生成所有级别（Q2_K → Q8_0），方便对比选择。

---

## 5. quantize_bnb.py — bitsandbytes 零校准量化

### 配置构造

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 模式
    bnb_4bit_quant_type="nf4",              # NF4 数据类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化为 bf16
    bnb_4bit_use_double_quant=True,         # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",                       # 自动分配 GPU/CPU
)
```

**核心理解**:
- bitsandbytes 在 `from_pretrained` 时实时量化，不需要校准数据
- `bnb_4bit_compute_dtype=bf16` 表示推理时先反量化到 bf16 再计算，质量更好
- `device_map="auto"` 让 accelerate 自动将层分配到可用设备
- 双重量化额外节省 ~0.4 bit/param（对 7B 模型约节省 350MB）

### NF4 vs FP4 选择

```python
# NF4（推荐）: 假设权重近似正态分布
bnb_4bit_quant_type = "nf4"

# FP4: 不做分布假设，均匀量化
bnb_4bit_quant_type = "fp4"
```

实测 NF4 在大多数 LLM 上 PPL 低 0.2-0.5，因为 LLM 权重确实近似正态分布。

---

## 6. benchmark.py — 多维度评测

### 困惑度计算

```python
def compute_perplexity(model, tokenizer, n_samples=256):
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * seq_len
        total_tokens += seq_len
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)  # PPL = e^(avg_loss)
```

**PPL 解读**: PPL=15 表示模型在每个 token 位置平均在 15 个选项中犹豫。FP16 基线 ~14，4-bit 量化后 ~15，差距越小说明量化质量越好。

### 推理速度测量

```python
def measure_inference_speed(model, tokenizer, warmup=3, runs=10):
    # Warmup: 前几次推理包含编译/缓存开销，不计入
    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=64)
    
    torch.cuda.synchronize()  # 确保 GPU 操作完成
    
    # 正式测量
    start = time.time()
    for _ in range(runs):
        outputs = model.generate(**inputs, max_new_tokens=64)
        total_tokens += generated_length
    torch.cuda.synchronize()
    
    return total_tokens / elapsed  # tokens/second
```

**关键**: `torch.cuda.synchronize()` 必须调用，否则 GPU 异步执行会导致时间测量不准。

### 对比表格

Benchmark 最终生成标准化的对比表格，包含：PPL（质量）、tokens/s（速度）、GB（大小）、VRAM（显存），让用户一目了然选择最适合的方法。
