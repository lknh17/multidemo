# p07 原理详解 - 模型量化

> 本文档详细讲解大模型量化的理论基础：从浮点数表示到四种主流量化方法的数学原理，帮助你深入理解"用更少的 bit 表示权重"背后的技术细节。

---

## 1. 什么是量化

量化（Quantization）是将模型权重从高精度（如 FP16, 16-bit）映射到低精度（如 INT4, 4-bit）的过程。本质上是一种**有损压缩**：用更少的 bit 表示每个参数，换取更小的模型体积和更低的显存占用。

**核心 trade-off**: 压缩率 ↔ 模型质量

| 精度 | 每参数 bit | 相对大小 | 精度损失 |
|------|-----------|---------|---------|
| FP32 | 32 bit | 基准 (1x) | 无 |
| FP16 / BF16 | 16 bit | 0.5x | 极小 |
| INT8 | 8 bit | 0.25x | 小 |
| INT4 | 4 bit | 0.125x | 中等 |
| INT2 | 2 bit | 0.0625x | 较大 |

**量化的两大类**:
- **QAT (Quantization-Aware Training)**: 训练时就考虑量化误差，质量最好但需要重新训练
- **PTQ (Post-Training Quantization)**: 训练后直接量化，不需要重训，本模块聚焦于此

---

## 2. 浮点数与定点数表示

### IEEE 754 浮点数

FP16 的二进制表示：1 位符号 + 5 位指数 + 10 位尾数

$$\text{value} = (-1)^s \times 2^{e-15} \times (1 + \frac{m}{1024})$$

其中 $s$ 是符号位，$e$ 是指数（5 bits，偏移 15），$m$ 是尾数（10 bits）。

### BF16 (Brain Float 16)

BF16 重新分配了 bit：1 位符号 + 8 位指数 + 7 位尾数

$$\text{BF16 范围} = \pm 3.4 \times 10^{38}$$
$$\text{FP16 范围} = \pm 65504$$

BF16 范围更大（与 FP32 相同），但精度略低（7 bit 尾数 vs FP16 的 10 bit）。

### INT4 定点数

INT4 只有 4 个 bit，表示范围：
- 无符号: $[0, 15]$
- 有符号: $[-8, 7]$

要将 FP16 的权重映射到 INT4，需要**缩放因子** (scale) 和**零点** (zero point)。

---

## 3. 对称量化与非对称量化

### 对称量化 (Symmetric)

假设权重分布以 0 为中心，量化公式：

$$x_q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}$$

反量化：

$$\hat{x} = x_q \times s$$

其中 $s$ 是缩放因子，$b$ 是量化位数。

**特点**: 零点固定为 0，计算简单，但如果权重不对称（偏向正或负），会浪费表示范围。

### 非对称量化 (Asymmetric)

允许零点不为 0，充分利用整数范围：

$$x_q = \text{round}\left(\frac{x - z}{s}\right), \quad s = \frac{x_{max} - x_{min}}{2^b - 1}, \quad z = x_{min}$$

反量化：

$$\hat{x} = x_q \times s + z$$

**特点**: 更灵活，能更好地适配非对称分布，但引入额外的零点参数。

### 量化误差

量化误差来源于 `round()` 操作：

$$\epsilon = x - \hat{x} = x - \left(\text{round}\left(\frac{x}{s}\right) \times s\right)$$

误差的上界为 $\frac{s}{2}$，即半个量化步长。位数越多，步长越小，误差越小。

---

## 4. 分组量化 (Group Quantization)

### 为什么需要分组

全局共享一个 scale 和 zero point 时，如果个别权重值特别大（离群值 / outlier），会拉大整个范围，导致其他权重的量化精度下降。

### 分组量化策略

将权重矩阵按行或按通道分成若干组（如 group_size=128），每组独立计算 scale 和 zero point：

$$\text{对于第 } g \text{ 组}: \quad s_g = \frac{\max(|W_g|)}{2^{b-1}-1}, \quad W_{q,g} = \text{round}\left(\frac{W_g}{s_g}\right)$$

### 分组大小的影响

| group_size | 额外参数 | 量化精度 | 推荐 |
|------------|---------|---------|------|
| 32 | 多 | 最好 | 对精度要求极高 |
| 64 | 中 | 好 | 平衡 |
| 128 | 少 | 好 | **默认推荐** |
| 256 | 很少 | 一般 | 极致压缩 |
| -1（全局） | 最少 | 最差 | 不推荐 |

group_size=128 时，每 128 个权重增加一个 FP16 的 scale（16 bit），平均每个权重额外增加 $\frac{16}{128} = 0.125$ bit，总计 4.125 bit/param。

---

## 5. GPTQ 算法原理

GPTQ (Generalized Post-Training Quantization) 基于 OBC/OBQ 框架，利用 **Hessian 矩阵的逆** 来最小化量化后的输出误差。

### 数学目标

对于权重矩阵 $W$ 和输入 $X$，量化后的目标是最小化输出误差：

$$\min_{\hat{W}} \|WX - \hat{W}X\|_2^2$$

### 逐列量化

GPTQ 逐列处理权重矩阵。对第 $i$ 列量化时：

1. **量化**: $\hat{w}_i = \text{quantize}(w_i)$
2. **计算误差**: $\delta_i = w_i - \hat{w}_i$
3. **补偿后续列**: $w_{j>i} \leftarrow w_{j>i} - \delta_i \cdot \frac{H_{ij}^{-1}}{H_{ii}^{-1}}$

其中 $H = X X^T$ 是 Hessian 矩阵（实际上是输入的二阶矩）。

### Hessian 的直觉

$H_{ii}^{-1}$ 表示第 $i$ 列权重的"重要性"。$H_{ii}^{-1}$ 越大，说明这个权重对输出的影响越大，量化误差的补偿就需要分摊到更多后续列。

### desc_act 排序

GPTQ 的 `desc_act=True` 表示按激活值（$H_{ii}$ 的对角线）**降序**处理列，先量化最重要的权重，让误差补偿有更多"空间"。

### 校准数据

Hessian $H = XX^T$ 需要校准数据来估计。通常 128 条文本就足够获得稳定的 Hessian 估计。

---

## 6. AWQ 算法原理

AWQ (Activation-aware Weight Quantization) 的核心观察：**1% 的显著权重（salient weights）决定了模型质量，保护它们比优化全部权重更有效。**

### 显著权重识别

通过前向传播统计每个通道的激活值幅度：

$$\text{saliency}_j = \mathbb{E}\left[\|X_{:,j}\|_2\right]$$

激活值大的通道对应的权重更重要。

### 缩放因子保护

AWQ 不直接跳过显著权重，而是对它们进行**缩放**后再量化：

$$\hat{W}_j = \text{quantize}(W_j \cdot s_j) / s_j$$

其中缩放因子 $s_j$ 与该通道的激活值相关：

$$s_j \propto \|X_{:,j}\|^{\alpha}, \quad \alpha \in [0, 1]$$

### 为什么缩放有效

量化误差与权重的绝对值成正比：

$$\text{error} \propto \frac{\max(|W|)}{2^b - 1}$$

缩放显著权重使其相对值变大，在量化后占据更多 INT4 表示，减少了重要通道的量化误差。

### 自动搜索最优 $\alpha$

AWQ 在 $[0, 1]$ 范围内搜索每一层的最优 $\alpha$，最小化量化后的均方误差：

$$\alpha^* = \arg\min_{\alpha} \|WX - \hat{W}(\alpha)X\|^2$$

---

## 7. GGUF 格式与量化原理

GGUF (GPT-Generated Unified Format) 是 llama.cpp 项目定义的模型格式，支持多种混合精度量化。

### K-Quant 策略

GGUF 的 K-Quant 使用**层内混合精度**：不同层（甚至同一层的不同部分）使用不同的量化位数。

```
Q4_K_M 的典型分配:
  - attention.wq, attention.wk: 4-bit (medium)
  - attention.wv, attention.wo: 4-bit (medium)
  - ffn.w1, ffn.w2, ffn.w3:    4-bit (medium)
  - output, embed:              6-bit (保护关键层)
```

### 各量化级别的编码方式

**Q4_0**: 最简单的 4-bit 量化
- 每 32 个权重一组
- 每组: 1 个 FP16 scale + 32 个 4-bit 权重
- 有效位数: $4 + \frac{16}{32} = 4.5$ bit/param

**Q4_K_M**: K-Quant medium
- 关键层（embed, output）使用 6-bit
- 其余层使用 4-bit
- 每组包含 scale + min 两个参数
- 有效位数: ~4.8 bit/param

**Q2_K**: 2-bit K-Quant
- 超级块（256 个权重）+ 子块
- 每个子块有独立的 scale
- 有效位数: ~2.6 bit/param

### GGUF 的优势

1. **跨平台**: 支持 CPU（x86/ARM）、GPU（CUDA/Metal）
2. **混合推理**: 部分层在 GPU，部分在 CPU
3. **格式统一**: 模型 + tokenizer + 元信息 = 一个文件
4. **生态丰富**: llama.cpp, Ollama, LM Studio, vLLM 等

---

## 8. bitsandbytes 量化原理

bitsandbytes 的核心创新是 **NF4 (Normal Float 4)** 数据类型。

### NF4 的设计动机

神经网络权重近似服从正态分布 $W \sim \mathcal{N}(0, \sigma^2)$。NF4 的 16 个量化点按照标准正态分布的分位数均匀分布：

$$q_i = \Phi^{-1}\left(\frac{i + 0.5}{16}\right), \quad i = 0, 1, ..., 15$$

其中 $\Phi^{-1}$ 是标准正态分布的分位数函数。

### NF4 量化点

```
NF4 的 16 个值（归一化后）:
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0]
```

每个权重被映射到最近的量化点。由于量化点按正态分布密度排列，**在权重密集的区域有更多量化点**，减少了量化误差。

### NF4 vs FP4

| 特性 | NF4 | FP4 (E2M1) |
|------|-----|-------------|
| 量化点分布 | 按正态分位数 | 均匀间隔 |
| 适合 | 正态分布权重 | 均匀分布权重 |
| 实际效果 | 更好 | 略差 |
| 理论依据 | 信息论最优 | 通用 |

### 双重量化 (Double Quantization)

量化引入的额外参数（每组一个 FP32 scale）也占空间。双重量化对这些 scale 再做一次量化：

$$\text{额外空间}: \frac{32}{g} \xrightarrow{\text{Double Quant}} \frac{8}{g} + \frac{32}{256g} \approx \frac{8.125}{g}$$

当 $g=64$ 时，从 0.5 bit/param 降低到 0.127 bit/param。

### bitsandbytes 的限制

1. **仅支持 NVIDIA GPU**: 依赖 CUDA 内核
2. **推理速度不如 GPTQ/AWQ**: 因为需要运行时反量化
3. **不生成独立模型文件**: 量化发生在加载时
4. **与 QLoRA 完美配合**: 4-bit 推理 + LoRA 微调

---

## 9. 四种方法对比总结

| 维度 | GPTQ | AWQ | GGUF | bitsandbytes |
|------|------|-----|------|-------------|
| **原理** | Hessian 逆矩阵 | 激活值感知缩放 | K-Quant 混合精度 | NF4 正态量化 |
| **需要校准** | ✅ 128条 | ✅ 128条 | ❌ | ❌ |
| **量化速度** | 中（分钟级） | 快（秒级） | 快（秒级） | 即时 |
| **推理速度** | ⚡ 很快 | ⚡⚡ 最快 | ⚡ 快（CPU也快） | 🐢 较慢 |
| **模型质量** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **支持位数** | 2/3/4/8 | 4 | 2~8（多级） | 4/8 |
| **硬件** | NVIDIA GPU | NVIDIA GPU | CPU+GPU | NVIDIA GPU |
| **生态** | HuggingFace | HuggingFace | llama.cpp/Ollama | HuggingFace+PEFT |
| **最佳场景** | GPU 服务部署 | GPU 高速推理 | 边缘/桌面部署 | QLoRA 训练 |

---

## 10. 校准数据与量化质量

### 校准数据的重要性

GPTQ 和 AWQ 都需要校准数据来估计权重的"重要性"。校准数据的选择影响量化质量：

$$H = \frac{1}{N}\sum_{i=1}^{N} X_i X_i^T$$

**最佳实践**:
- 使用与推理场景相似的数据（如部署为中文助手，就用中文数据校准）
- 128 条通常足够，更多不一定更好
- 多样性比数量更重要

### 量化的局限性

1. **离群值问题**: 个别极端权重会降低整体量化精度（GPTQ/AWQ 都在解决此问题）
2. **任务敏感性**: 某些任务（如数学推理）对量化更敏感
3. **2-bit 瓶颈**: 低于 3-bit 时质量下降显著，目前 4-bit 是最佳平衡点
4. **KV Cache 未量化**: 常规量化只压缩权重，KV Cache 仍是 FP16（占推理时显存的大头）

### 未来方向

- **KV Cache 量化**: 如 KIVI, Atom 等方法
- **混合精度量化**: 不同层不同位数（GGUF 的 K-Quant 已实现）
- **1-bit 量化 (BitNet)**: 微软提出的极致量化
- **量化感知训练 (QAT)**: 训练中加入量化模拟，如 AQLM
