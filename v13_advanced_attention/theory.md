# 高级注意力机制与长上下文深入原理

> 本文深入解析现代大模型使用的注意力变体——GQA、MQA、Sliding Window、Flash Attention，以及长上下文扩展技术 NTK-aware RoPE 和 YaRN。

## 1. 从 MHA 到 GQA/MQA 的演进

### 1.1 MHA (Multi-Head Attention) 回顾

标准多头注意力为每个头分配独立的 Q、K、V 投影：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

KV Cache 大小：$2 \times n_{heads} \times L \times d_k$，随头数线性增长。

### 1.2 MQA (Multi-Query Attention)

**核心思想**：所有 Query head 共享**同一组** Key 和 Value。

$$Q_i = XW_i^Q, \quad K = XW^K, \quad V = XW^V$$

- KV Cache 大小降为 $2 \times L \times d_k$（与头数无关）
- 推理速度大幅提升（KV Cache 是推理瓶颈）
- 缺点：训练表达力下降，质量略有损失

### 1.3 GQA (Grouped Query Attention)

**核心思想**：MHA 和 MQA 的折中。将 Query head 分成 $G$ 组，每组共享一对 KV head。

$$Q_i = XW_i^Q, \quad K_g = XW_g^K, \quad V_g = XW_g^V$$

其中第 $i$ 个 Query head 使用第 $g = \lfloor i / (n_{heads}/G) \rfloor$ 组的 KV。

| 方法 | KV Head 数 | KV Cache | 质量 | 使用模型 |
|------|-----------|----------|------|----------|
| MHA | n_heads | $2n_hLd_k$ | 最高 | 原始 Transformer |
| GQA | G (1<G<n_h) | $2GLd_k$ | 接近 MHA | Qwen2, LLaMA2-70B |
| MQA | 1 | $2Ld_k$ | 略低 | PaLM, Falcon |

**Qwen2 实际配置**：n_heads=32, GQA groups=8，KV Cache 降为 MHA 的 1/4。

## 2. Sliding Window Attention

**核心思想**：每个 token 只关注其前后 $w$ 个 token，而非全序列。

$$\text{Attn}(i, j) = \begin{cases} \text{normal} & \text{if } |i - j| \le w \\ -\infty & \text{otherwise} \end{cases}$$

- 复杂度从 $O(n^2)$ 降为 $O(n \cdot w)$
- 通过层层堆叠，信息仍可传播到全局（第 $L$ 层可感知 $L \times w$ 的距离）
- **Mistral** 的策略：部分层用 Sliding Window，部分层用 Full Attention

## 3. Flash Attention

**关键洞察**：标准注意力的瓶颈不在计算量，而在**内存访问**。

### 3.1 标准注意力的内存问题

$$S = QK^T \in \mathbb{R}^{N \times N}$$

需要实现整个 $N \times N$ 的注意力矩阵——当 $N$ 很大时，HBM（GPU 高带宽内存）的读写成为瓶颈。

### 3.2 Flash Attention 的 Tiling 策略

- 将 Q、K、V 分块（tile），每个块足够小以放入 SRAM（片上缓存）
- 在 SRAM 中完成 Softmax + 加权求和
- **Online Softmax**：边计算边更新 softmax 统计量，无需先存储完整 $S$ 矩阵
- 这**不是近似**！是精确的注意力计算，只是改变了计算顺序

### 3.3 复杂度对比

| 方面 | 标准 Attention | Flash Attention |
|------|---------------|-----------------|
| FLOPs | $O(N^2d)$ | $O(N^2d)$（相同） |
| HBM 读写 | $O(N^2 + Nd)$ | $O(N^2d^2/M)$ |
| 额外内存 | $O(N^2)$ | $O(N)$ |

其中 $M$ 是 SRAM 大小。Flash Attention 将 HBM 访问量减少了 $O(N/M)$ 倍。

## 4. 长上下文扩展

### 4.1 RoPE 外推问题

RoPE 在训练时只见过 $[0, L_{train})$ 范围的位置。推理时遇到 $pos > L_{train}$ 会导致注意力分数异常。

### 4.2 NTK-aware RoPE Scaling

核心思想：调整 RoPE 的频率基数 $\theta$，使高频分量保持不变（局部位置不变），低频分量压缩（全局位置适应）。

$$\theta'_i = \theta \cdot \alpha^{2i/d}$$

其中 $\alpha = (L_{target} / L_{train})^{d/(d-2)}$。

### 4.3 YaRN

YaRN 在 NTK 基础上进一步：
1. 对不同频率分量应用不同的缩放策略
2. 低频（变化慢的维度）→ 更多缩放
3. 高频（变化快的维度）→ 不缩放
4. 加入注意力分数温度校正

### 4.4 Dynamic NTK

推理时根据实际输入长度动态调整 $\alpha$，无需重新训练。

## 5. Attention Sink (StreamingLLM)

**发现**：大模型注意力中，**第一个 token**（无论内容是什么）始终获得异常高的注意力权重——这是 Softmax 的数学特性决定的。

**利用**：保留前几个 token（sink tokens）+ 最近的 $w$ 个 token，即可实现无限长度的流式推理。

## 总结

| 技术 | 核心目标 | 现代应用 |
|------|---------|---------|
| GQA | 减少 KV Cache，加速推理 | Qwen2, LLaMA2 |
| MQA | KV Cache 最小化 | PaLM, Falcon |
| Sliding Window | 降低注意力复杂度 | Mistral |
| Flash Attention | 减少内存访问 | 几乎所有现代模型 |
| NTK/YaRN | 长上下文外推 | Qwen2, Code LLaMA |
| Attention Sink | 无限长度推理 | StreamingLLM |
