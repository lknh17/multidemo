# Transformer 基础原理

> 本文从零推导 Transformer 的核心组件，帮助你彻底理解自注意力机制、位置编码和前馈网络的数学本质。

## 目录

1. [为什么需要 Transformer？](#1-为什么需要-transformer)
2. [自注意力机制 (Self-Attention)](#2-自注意力机制-self-attention)
3. [多头注意力 (Multi-Head Attention)](#3-多头注意力-multi-head-attention)
4. [位置编码 (Positional Encoding)](#4-位置编码-positional-encoding)
5. [前馈网络 (Feed-Forward Network)](#5-前馈网络-feed-forward-network)
6. [残差连接与层归一化](#6-残差连接与层归一化)
7. [完整 Transformer Block](#7-完整-transformer-block)
8. [编码器与解码器](#8-编码器与解码器)

---

## 1. 为什么需要 Transformer？

### RNN 的局限性

在 Transformer 之前，序列建模主要依赖 RNN（LSTM/GRU）：

- **串行计算**：必须按时间步逐个处理，无法并行化 → 训练速度慢
- **长距离依赖衰减**：信息需要逐步传递，距离越远衰减越严重
- **梯度消失/爆炸**：反向传播路径过长

### Transformer 的核心思想

**"Attention Is All You Need"** (Vaswani et al., 2017)

核心创新：**用注意力机制完全替代递归结构**。

- 每个位置可以直接"看到"序列中所有其他位置 → 解决长距离依赖
- 所有位置可以并行计算 → 大幅提升训练速度
- 通过多层堆叠，逐步抽取更高层次的语义表示

---

## 2. 自注意力机制 (Self-Attention)

### 2.1 直觉理解

想象你在读一句话："**这只猫很可爱，它正在睡觉**"。

当你处理"它"这个词时，大脑会自动关联到"猫"——这就是"注意力"：**根据当前 token 的信息，去关注序列中其他相关的 token**。

### 2.2 Query-Key-Value 机制

每个 token 的表示向量 \(x_i \in \mathbb{R}^{d_{model}}\) 会被线性变换为三个向量：

\[
Q_i = x_i W^Q, \quad K_i = x_i W^K, \quad V_i = x_i W^V
\]

其中 \(W^Q, W^K, W^V \in \mathbb{R}^{d_{model} \times d_k}\) 是可学习的权重矩阵。

**含义**：
- **Query (查询)**：当前 token "想要找什么信息"
- **Key (键)**：每个 token "有什么信息可以提供"
- **Value (值)**：每个 token "实际提供的信息内容"

### 2.3 注意力分数计算

**步骤 1**：计算 Query 和所有 Key 的点积（衡量相关性）

\[
\text{score}(i, j) = Q_i \cdot K_j^T
\]

**步骤 2**：缩放（Scaled）

\[
\text{score}(i, j) = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}
\]

**为什么要除以 \(\sqrt{d_k}\)？**

假设 Q 和 K 的每个元素都是均值 0、方差 1 的独立随机变量，则点积的方差为 \(d_k\)。当 \(d_k\) 较大时，点积值会很大，导致 Softmax 后梯度极小（进入饱和区）。除以 \(\sqrt{d_k}\) 可以把方差重新拉回到 1。

**步骤 3**：Softmax 归一化

\[
\alpha_{ij} = \text{Softmax}_j\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) = \frac{\exp(\text{score}(i,j))}{\sum_{k=1}^{n} \exp(\text{score}(i,k))}
\]

**步骤 4**：加权求和 Value

\[
\text{Attention}(Q, K, V)_i = \sum_{j=1}^{n} \alpha_{ij} V_j
\]

### 2.4 矩阵形式（批量计算）

将所有 token 的 Q、K、V 堆叠成矩阵：

\[
Q = X W^Q \in \mathbb{R}^{n \times d_k}, \quad K = X W^K \in \mathbb{R}^{n \times d_k}, \quad V = X W^V \in \mathbb{R}^{n \times d_v}
\]

\[
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
\]

计算复杂度：\(O(n^2 \cdot d)\)，其中 \(n\) 是序列长度，\(d\) 是维度。

---

## 3. 多头注意力 (Multi-Head Attention)

### 3.1 为什么需要多头？

单一注意力头只能学习一种"关注模式"。但语言中存在多种关系：
- 语法关系（主语-谓语）
- 语义关系（代词-指代对象）
- 位置关系（相邻词的搭配）

多头注意力让模型同时学习多种不同的关注模式。

### 3.2 数学公式

将 \(d_{model}\) 维度拆成 \(h\) 个头，每个头的维度为 \(d_k = d_{model} / h\)：

\[
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\]

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
\]

其中：
- \(W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}\)：每个头的投影矩阵
- \(W^O \in \mathbb{R}^{d_{model} \times d_{model}}\)：输出投影矩阵
- Concat 操作将 h 个头的输出拼接回 \(d_{model}\) 维度

### 3.3 参数量分析

每个头有 Q、K、V 三个投影矩阵 + 一个输出投影矩阵：
- 每头参数：\(3 \times d_{model} \times d_k = 3 \times d_{model} \times (d_{model}/h)\)
- h 个头总参数：\(3 \times d_{model}^2\)
- 输出投影：\(d_{model}^2\)
- **总计**：\(4 \times d_{model}^2\)

注意：这与单头（\(d_k = d_{model}\)）的参数量完全相同！多头并没有增加参数量，但提升了表达能力。

---

## 4. 位置编码 (Positional Encoding)

### 4.1 为什么需要位置编码？

自注意力是一个**集合操作（Set Operation）**：它对输入的排列顺序不敏感。

也就是说，如果打乱输入序列的顺序，注意力的输出不会改变（除了顺序也跟着变了）。但语言是有序的："猫追狗" ≠ "狗追猫"。

因此，需要显式注入位置信息。

### 4.2 正弦位置编码（原始 Transformer）

\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

其中 \(pos\) 是位置索引，\(i\) 是维度索引。

**为什么这样设计？**

1. **不同频率**：每个维度对应不同的正弦频率，低维度变化快（捕捉局部位置），高维度变化慢（捕捉全局位置）
2. **相对位置可表示**：\(PE_{pos+k}\) 可以用 \(PE_{pos}\) 的线性变换表示
   \[
   \begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix} \begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}
   \]
3. **不增加可学习参数**：是确定性的函数，不需要训练

### 4.3 可学习位置编码

另一种方式是直接学习位置嵌入：

\[
PE = \text{Embedding}(pos) \in \mathbb{R}^{max\_len \times d_{model}}
\]

这是一个可训练的嵌入矩阵，通过训练数据学到位置模式。

**对比**：
| 特性 | 正弦编码 | 可学习编码 |
|------|----------|------------|
| 外推能力 | 好（可处理训练中未见过的长度） | 差（最大长度固定） |
| 参数量 | 0 | \(max\_len \times d_{model}\) |
| 表达力 | 受函数形式限制 | 更灵活 |
| 主流用法 | 原始 Transformer | GPT、BERT、ViT |

### 4.4 旋转位置编码 (RoPE)

**RoPE (Rotary Position Embedding)** 是目前大语言模型的主流选择（Qwen、LLaMA 都在用）。

核心思想：**不是把位置信息加到 embedding 上，而是对 Q 和 K 施加与位置相关的旋转**。

\[
f(x, pos) = x \cdot e^{i \cdot pos \cdot \theta}
\]

在实数域实现为 2D 旋转矩阵：

\[
R_{\theta, pos} = \begin{pmatrix} \cos(pos \cdot \theta) & -\sin(pos \cdot \theta) \\ \sin(pos \cdot \theta) & \cos(pos \cdot \theta) \end{pmatrix}
\]

**RoPE 的优势**：
- Q·K 的内积自然包含了相对位置信息
- 优秀的外推能力（配合 NTK-aware 等策略可支持超长序列）
- 不增加额外参数

我们会在 v03（LLM 版本）中详细实现 RoPE。

---

## 5. 前馈网络 (Feed-Forward Network)

### 5.1 结构

每个 Transformer 层中，自注意力之后接一个逐位置的前馈网络（Position-wise FFN）：

\[
\text{FFN}(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2
\]

通常 \(W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}\)，\(W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}\)，其中 \(d_{ff} = 4 \times d_{model}\)。

### 5.2 "逐位置"的含义

FFN 对序列中每个位置**独立**应用相同的变换。也就是说：
- 位置 1 的 FFN 和位置 2 的 FFN 共享参数
- 但它们互不影响
- 位置之间的信息交互完全由注意力层负责

### 5.3 激活函数的选择

| 激活函数 | 公式 | 使用场景 |
|----------|------|----------|
| ReLU | \(\max(0, x)\) | 原始 Transformer |
| GELU | \(x \cdot \Phi(x)\) | BERT, GPT-2 |
| SwiGLU | \(\text{Swish}(x W_1) \odot (x W_3)\) | LLaMA, Qwen |

**SwiGLU** 是目前大模型的主流选择：

\[
\text{SwiGLU}(x) = (\text{Swish}(x W_{gate}) \odot (x W_{up})) W_{down}
\]

其中 \(\text{Swish}(x) = x \cdot \sigma(x)\)，\(\odot\) 是逐元素乘法。

SwiGLU 使用了门控机制（Gate），让网络学习"让多少信息通过"，实验表明它比 ReLU/GELU 更优。

### 5.4 FFN 的参数量

对于标准 FFN：\(2 \times d_{model} \times d_{ff} = 2 \times d_{model} \times 4d_{model} = 8d_{model}^2\)

这通常是 Transformer 层中参数量最大的部分（多头注意力是 \(4d_{model}^2\)）！

---

## 6. 残差连接与层归一化

### 6.1 残差连接 (Residual Connection)

\[
\text{output} = \text{SubLayer}(x) + x
\]

**为什么需要残差连接？**
- 梯度直通：梯度可以通过恒等映射直接传播到底层，缓解梯度消失
- 降低优化难度：网络只需要学习"残差"（与输入的差异），而不是完整的变换
- 允许更深的网络：没有残差连接，Transformer 很难训练超过 6 层

### 6.2 层归一化 (Layer Normalization)

\[
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta
\]

其中 \(\mu, \sigma\) 是在**特征维度**上计算的均值和标准差，\(\gamma, \beta\) 是可学习的缩放和偏移参数。

**为什么用 LayerNorm 而不是 BatchNorm？**
- BatchNorm 在 batch 维度归一化 → 依赖 batch 中的其他样本
- LayerNorm 在特征维度归一化 → 每个样本独立计算
- 序列任务中 batch size 可能很小，BatchNorm 统计不稳定

### 6.3 Pre-Norm vs Post-Norm

```
Post-Norm (原始 Transformer):
    x → SubLayer → Add → LayerNorm → 输出

Pre-Norm (现代做法):
    x → LayerNorm → SubLayer → Add → 输出
```

**Pre-Norm 的优势**：
- 训练更稳定（梯度更平滑）
- 可以省去 warmup（或用更短的 warmup）
- GPT、LLaMA、Qwen 等现代大模型都用 Pre-Norm

### 6.4 RMSNorm（更轻量的归一化）

\[
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
\]

- 去掉了均值中心化步骤，只做缩放
- 计算更快，效果几乎不变
- LLaMA、Qwen 等模型使用 RMSNorm

---

## 7. 完整 Transformer Block

一个 Transformer Block = 多头注意力 + FFN + 残差 + 归一化

### Pre-Norm 版本（推荐，现代主流）：

```
Input x
  │
  ├──────────────────────┐
  │                      │ (残差连接)
  ▼                      │
LayerNorm                │
  │                      │
  ▼                      │
Multi-Head Attention     │
  │                      │
  ▼                      │
  + ◄────────────────────┘
  │
  ├──────────────────────┐
  │                      │ (残差连接)
  ▼                      │
LayerNorm                │
  │                      │
  ▼                      │
Feed-Forward Network     │
  │                      │
  ▼                      │
  + ◄────────────────────┘
  │
  ▼
Output
```

---

## 8. 编码器与解码器

### 8.1 编码器 (Encoder)

- 输入可以看到所有位置（双向注意力）
- 适用于理解任务（分类、NER、相似度计算）
- 代表模型：BERT

### 8.2 解码器 (Decoder)

- 使用因果掩码，每个位置只能看到它之前的位置
- 适用于生成任务（文本生成、翻译的目标端）
- 代表模型：GPT 系列

### 8.3 编码器-解码器 (Encoder-Decoder)

- 编码器处理输入，解码器生成输出
- 解码器中有额外的 Cross-Attention 层，Query 来自解码器，Key/Value 来自编码器
- 适用于序列到序列任务（翻译、摘要）
- 代表模型：原始 Transformer、T5、BART

### 8.4 本版本的实现

在本版本中，我们实现一个 **Encoder-Decoder** 架构的 Mini Transformer，用于序列排序任务（输入乱序数字，输出排好序的数字），以直观演示 Transformer 的工作方式。

---

## 总结

| 组件 | 作用 | 关键公式 |
|------|------|----------|
| Self-Attention | 建模序列内部的依赖关系 | \(\text{Softmax}(QK^T/\sqrt{d_k})V\) |
| Multi-Head | 学习多种注意力模式 | 将维度拆成 h 个头并行计算 |
| 位置编码 | 注入位置信息 | 正弦/可学习/RoPE |
| FFN | 非线性变换，增加表达力 | \(\text{Act}(xW_1)W_2\) |
| 残差连接 | 缓解梯度消失，便于优化 | \(y = f(x) + x\) |
| 层归一化 | 稳定训练，加速收敛 | 在特征维度上标准化 |

下一步：在 `model.py` 中从零实现这些组件！
