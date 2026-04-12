# p07 模型量化

> **目标**: 掌握大模型量化的四大主流方法（GPTQ/AWQ/GGUF/bitsandbytes），理解其原理差异，通过实际 benchmark 对比选择最优方案。
>
> **前置条件**: 完成 p06 MLLM 训练，拥有可量化的模型。
>
> **预计耗时**: 2-3 小时（含全部量化 + 评测）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 理解量化的数学原理（对称/非对称、分组量化）
- 掌握 GPTQ 基于 Hessian 的逐层最优量化
- 掌握 AWQ 基于激活值感知的权重保护策略
- 学会将模型转换为 GGUF 格式供 llama.cpp / Ollama 使用
- 使用 bitsandbytes NF4 实现零校准快速量化
- 通过 Benchmark 对比四种方法的质量、速度、大小、显存

### 本模块代码文件概览

本模块围绕"如何让大模型跑得更轻更快"展开：**四种量化方法实操 → 统一 Benchmark → 推理质量对比**。每种量化方法对应一个独立脚本，方便单独学习和对比。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | 量化全局配置（模型路径、校准样本数、量化参数） | 理解量化配置项 |
| `quantize_gptq.py` | GPTQ 量化：基于 Hessian 的逐层最优量化 | 掌握经典 PTQ 方法 |
| `quantize_awq.py` | AWQ 量化：激活值感知的权重保护策略 | 理解"重要权重保护"思路 |
| `quantize_gguf.py` | GGUF 转换：支持 llama.cpp / Ollama 的格式 | 学会制作本地可运行模型 |
| `quantize_bnb.py` | bitsandbytes 量化：NF4/FP4 零校准快速量化 | 掌握最简单的量化方式 |
| `benchmark.py` | 统一对比四种方法的困惑度、速度、显存、模型大小 | 数据驱动的方法选型 |
| `inference.py` | 对比各量化版本的实际生成质量 | 直观感受量化影响 |

> 💡 **建议学习顺序**: 先分别跑 `quantize_gptq.py` → `quantize_awq.py` → `quantize_gguf.py` → `quantize_bnb.py`，然后用 `benchmark.py` 统一对比，最后用 `inference.py` 看实际效果

### 确认前置条件

```bash
cd p07_quantization
pip install -r requirements.txt

# 验证依赖
python -c "import auto_gptq; print(f'auto-gptq: {auto_gptq.__version__}')"
python -c "import awq; print('autoawq: OK')"
python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')"
```

---

## 2. 为什么需要量化

大模型的部署瓶颈在 **显存** 和 **计算**：

| 模型大小 | FP16 显存 | 4-bit 显存 | 压缩比 |
|----------|----------|-----------|--------|
| 0.5B     | ~1 GB    | ~0.3 GB   | ~3.3x  |
| 7B       | ~14 GB   | ~4 GB     | ~3.5x  |
| 70B      | ~140 GB  | ~35 GB    | ~4x    |

量化将权重从 FP16 (16-bit) 压缩到 4-bit 甚至 2-bit，使得消费级显卡也能运行大模型。

---

## 3. GPTQ 量化

GPTQ 是最经典的 PTQ（训练后量化）方法，基于二阶信息（Hessian 矩阵）逐层优化量化误差。

```bash
# 基础 4-bit GPTQ 量化
python quantize_gptq.py

# 自定义参数
python quantize_gptq.py --bits 4 --group-size 128
python quantize_gptq.py --bits 3 --group-size 64   # 更激进的压缩
```

**预期输出**:
```
GPTQ 量化
  模型:       Qwen/Qwen2.5-0.5B
  量化位数:   4-bit
  分组大小:   128
  校准样本:   128
  量化耗时:   ~60s
  模型大小:   ~0.32 GB
```

### GPTQ 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `bits` | 4 | 量化位数（2/3/4/8） |
| `group_size` | 128 | 每 128 个权重共享一组 scale/zero |
| `desc_act` | True | 按激活值大小排序后量化（更精确） |
| `sym` | True | 对称量化（范围 [-max, max]） |

---

## 4. AWQ 量化

AWQ 的核心思想：**不是所有权重同等重要，保护激活值大的通道**。

```bash
# AWQ 4-bit 量化
python quantize_awq.py

# 使用 GEMV 内核（适合 batch=1 推理）
python quantize_awq.py --version GEMV
```

**预期输出**:
```
AWQ 量化
  量化位数:   4-bit
  内核版本:   GEMM
  量化耗时:   ~45s
  模型大小:   ~0.30 GB
```

### AWQ vs GPTQ 对比

| 特性 | GPTQ | AWQ |
|------|------|-----|
| 校准数据需求 | 中（128条） | 少（更快） |
| 量化质量 | 高 | 高（略优） |
| 推理速度 | 快 | 略快（GEMM内核） |
| 易用性 | 成熟 | 新兴 |

---

## 5. GGUF 量化

GGUF 是 llama.cpp 生态的标准格式，支持 CPU 推理和混合 CPU+GPU 推理。

```bash
# 默认 Q4_K_M 量化
python quantize_gguf.py

# 生成所有量化级别
python quantize_gguf.py --all-types

# 指定量化级别
python quantize_gguf.py --quant-type Q5_K_M
```

### GGUF 量化级别

| 级别 | 位数 | 相对大小 | 质量 | 推荐场景 |
|------|------|---------|------|---------|
| Q2_K | ~2.5 | 最小 | 较差 | 极限压缩 |
| Q3_K_M | ~3.5 | 小 | 中等 | 移动端 |
| Q4_K_M | ~4.5 | 中 | 好 | **日常使用（推荐）** |
| Q5_K_M | ~5.5 | 较大 | 很好 | 质量优先 |
| Q6_K | ~6.5 | 大 | 优秀 | 接近原始 |
| Q8_0 | 8 | 较大 | 极好 | 几乎无损 |

> **在 Ollama 中使用**: `ollama create mymodel -f Modelfile`

---

## 6. bitsandbytes 量化

bitsandbytes 最大的优势：**无需校准数据，加载即量化**。

```bash
# NF4 量化（推荐）
python quantize_bnb.py

# FP4 量化
python quantize_bnb.py --quant-type fp4

# 8-bit 量化
python quantize_bnb.py --8bit

# 禁用双重量化
python quantize_bnb.py --no-double-quant
```

### NF4 vs FP4

| 特性 | NF4 | FP4 |
|------|-----|-----|
| 数据类型 | Normal Float 4 | Float Point 4 |
| 假设 | 权重近似正态分布 | 无假设 |
| 精度 | 更好 | 略差 |
| 推荐 | ✅ 推荐 | 特殊场景 |

**双重量化**: 对量化参数（scale 和 zero point）再做一次量化，额外节省 ~0.4 bits/param。

---

## 7. 全方法 Benchmark

```bash
# 运行完整 benchmark
python benchmark.py

# 指定评测方法
python benchmark.py --methods fp16 gptq awq bnb

# 跳过困惑度（加速评测）
python benchmark.py --skip-perplexity
```

**预期输出**:
```
=========================================================================
量化方法对比 Benchmark 结果
=========================================================================
方法                   位数     大小(GB)  VRAM(GB)      PPL  速度(tok/s)   加载(s)
----------------------------------------------------------------------------------
FP16 (基线)            16-bit     1.00G     1.02G     14.20        45.2     2.1s
GPTQ                   4-bit      0.32G     0.42G     14.85        62.3     1.5s
AWQ                    4-bit      0.30G     0.40G     14.72        65.8     1.3s
bitsandbytes (NF4)     4-bit      0.30G     0.38G     15.10        42.1     3.2s
=========================================================================
```

---

## 8. 推理对比

```bash
# 对比所有方法的输出质量
python inference.py

# 指定方法
python inference.py --methods fp16 gptq awq bnb

# 自定义 prompt
python inference.py --prompt "请解释量化的原理"
```

观察重点：
- 4-bit 量化后语义是否改变
- 不同方法对中文生成的影响
- 推理速度差异

---

## 9. 小结与选型建议

### 四种方法的选型建议

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| GPU 服务器部署 | GPTQ / AWQ | 最快推理速度，质量好 |
| 消费级显卡推理 | AWQ + GEMM | 速度快，质量优 |
| CPU / 边缘设备 | GGUF (Q4_K_M) | llama.cpp 生态，支持 CPU |
| LoRA 微调 + 量化 | bitsandbytes | 与 PEFT 完美配合，QLoRA |
| 快速实验 | bitsandbytes | 零校准，最简单 |
| Ollama / LM Studio | GGUF | 标准格式 |

### 完成清单

- [x] 理解对称/非对称量化的数学原理
- [x] GPTQ 量化并验证质量
- [x] AWQ 量化并对比
- [x] GGUF 多级别量化
- [x] bitsandbytes NF4 / FP4 量化
- [x] 完成 Benchmark 对比
- [x] 推理输出质量对比

**下一步**: 进入 [p08 部署推理](../p08_serving/README.md)，将量化后的模型部署为 API 服务。
