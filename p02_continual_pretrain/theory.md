# p02 原理详解 - 继续预训练

> 本文档详细讲解继续预训练的理论基础：CLM 目标函数、数据处理策略、学习率调度、混合精度训练等。

---

## 1. 什么是继续预训练

继续预训练（Continual Pre-training）是在已训练的基座模型上，用**领域数据**继续训练，注入新知识。

| 类型 | 起点 | 数据 | 目标 | 训练量 |
|------|------|------|------|--------|
| 从零预训练 | 随机初始化 | TB 级通用语料 | 学习语言和世界知识 | 万亿 token |
| 继续预训练 | 已训练模型 | GB 级领域语料 | 注入领域知识 | 十亿 token |
| SFT | 已训练模型 | 万级指令数据 | 学会对话格式 | 百万 token |

**使用场景**: 基座模型在特定领域（如中文/医学/金融）知识不足时，用领域语料继续预训练。

---

## 2. 因果语言模型 (CLM) 目标函数

继续预训练使用和预训练完全相同的训练目标：**预测下一个 token**。

$$L = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_1, x_2, ..., x_{t-1}; \theta)$$

其中 $T$ 是序列长度，$x_t$ 是第 $t$ 个 token，$\theta$ 是模型参数。

**直觉理解**: 给定前面的所有 token，最大化下一个 token 的预测概率。模型在这个过程中被迫学习语言规律和世界知识。

**Cross Entropy Loss**: 实现上就是 `nn.CrossEntropyLoss`，将模型输出的 logits 和真实 token id 计算交叉熵。

**Teacher Forcing**: 训练时输入真实的前缀（而非模型自己的预测），让训练更稳定。

---

## 3. 数据处理：Tokenize 详解

### BPE (Byte Pair Encoding) 回顾

Qwen2.5 使用基于 BPE 的 tokenizer，核心思想是：
- 从字符级开始，逐步合并最频繁出现的相邻 pair
- 最终得到一个子词 (subword) 词表
- 常见词保留为完整 token，罕见词拆为多个子词

### Qwen2.5 Tokenizer 特点

- 词表大小: 151,936（支持中英文混合）
- 中文: 大部分常见汉字是单独 token
- 英文: 常见单词是完整 token，罕见词拆为子词
- 特殊 token: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>` 等

---

## 4. 数据处理：Packing vs Padding

### Padding 模式

```
样本1: [token token token PAD PAD PAD PAD PAD]  ← 40% 浪费
样本2: [token token token token token PAD PAD]    ← 28% 浪费
样本3: [token token PAD PAD PAD PAD PAD PAD]      ← 75% 浪费
```

- 每条数据独立 tokenize，短的补 pad
- pad 部分不参与 loss 计算但占用 GPU 计算资源
- 平均效率约 60%（取决于数据长度分布）

### Packing 模式（推荐）

```
样本: [文本A tokens | EOS | 文本B tokens | EOS | 文本C tokens]
         ← 整条序列都是有效 token，效率 ~98% →
```

- 多条文本拼接为一条长序列，用 EOS 分隔
- 几乎没有浪费（只有末尾可能有少量不足 max_length 的残余）
- 注意：因果注意力天然防止跨文档信息泄漏

---

## 5. 数据混合与配比

当使用多源数据时，混合比例很重要。

**常见策略**:
- **等比混合**: 每个数据源的样本数相同
- **按比例混合**: 高质量数据给更大权重（如 Wikipedia×3 + 网页×1）
- **温度采样**: $P(\text{source}_i) \propto n_i^{1/T}$，T 越小越均匀

**防遗忘技巧**: 在领域数据中混入少量通用数据（如 10-20%），可以缓解灾难性遗忘。

---

## 6. 学习率策略详解

### Warmup 的必要性

训练初期模型参数是随机的，大学习率可能导致不稳定。Warmup 让学习率从 0 线性增长到目标值。

$$lr_t = lr_{max} \times \frac{t}{T_{warmup}}, \quad t \leq T_{warmup}$$

### Cosine Decay（推荐）

Warmup 后学习率按余弦函数衰减，后期平滑下降：

$$lr_t = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})(1 + \cos(\frac{\pi \cdot (t - T_w)}{T - T_w}))$$

### Linear Decay

Warmup 后学习率线性下降到 0，比 cosine 更简单但后期下降更快。

### 学习率与 Batch Size 的关系

经验法则：batch size 加倍时，学习率也可以适当增大（线性缩放法则）。

$$lr \propto \sqrt{B}$$

---

## 7. 混合精度训练

### bf16 vs fp16

| 特性 | bf16 | fp16 |
|------|------|------|
| 指数位 | 8 位 | 5 位 |
| 尾数位 | 7 位 | 10 位 |
| 数值范围 | ±3.4×10³⁸ | ±65504 |
| 精度 | ~2 位有效数字 | ~3 位有效数字 |
| Loss Scaling | 不需要 | 需要 |
| GPU 要求 | Ampere+ | Volta+ |

**推荐 bf16**: 数值范围大，不需要 Loss Scaling，训练更稳定。Qwen2.5 系列官方推荐 bf16。

### Loss Scaling（fp16 专用）

fp16 的数值范围小，梯度容易 underflow（变成 0）。Loss Scaling 在反向传播前将 loss 乘以一个大数（如 1024），让梯度"放大"，避免 underflow，然后在更新参数前除回来。

---

## 8. Gradient Checkpointing 原理

### 标准训练

前向传播保存每层的激活值，反向传播时直接使用：
- 内存: $O(L)$（L 层各保存一份）
- 计算: 1 次前向 + 1 次反向

### Gradient Checkpointing

只保存部分层（checkpoint 层）的激活值，其余在反向传播时重新计算：
- 内存: $O(\sqrt{L})$
- 计算: 1 次前向 + 1 次反向 + 部分前向（约 30% 额外计算）

**HuggingFace 实现**: `model.gradient_checkpointing_enable()` 一行代码搞定。

---

## 9. 灾难性遗忘

### 为什么会遗忘

继续预训练时，模型参数被领域数据"拉"向新知识的方向，原有通用知识对应的参数被覆盖。

### 缓解策略

1. **低学习率**: 用比从零预训练小 10-100 倍的学习率（如 2e-5 而非 1e-4）
2. **数据混合**: 混入 10-20% 的通用数据
3. **Replay Buffer**: 定期在旧数据上"复习"
4. **EWC/正则化**: 对重要参数的变化施加惩罚
5. **LoRA**: 不修改原始参数，避免遗忘（但学习能力也受限）

---

## 10. 训练监控指标

### Loss 曲线

- **正常**: 快速下降后趋于平缓
- **异常**: NaN（学习率太大）、不下降（学习率太小/数据问题）、震荡（batch 太小）

### Perplexity

$$PPL = e^{L}$$

直觉：模型在"每个 token 上平均有多少个选择"。PPL=10 表示模型平均在 10 个候选 token 中犹豫。越低越好。

### Gradient Norm

监控梯度范数可以检测训练稳定性。梯度突然爆大通常预示训练即将崩溃。

### 学习率曲线

确认学习率按预期变化（warmup → decay），排除配置错误。
