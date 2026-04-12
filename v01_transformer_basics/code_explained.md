# v01 代码解释：Mini Transformer 实现详解

## 文件结构

| 文件 | 作用 |
|------|------|
| `model.py` | Transformer 所有核心组件的从零实现 |
| `dataset.py` | 数字排序任务的数据生成与加载 |
| `train.py` | 完整训练流程：Teacher Forcing + Warmup Cosine LR |
| `inference.py` | 两种解码策略：贪心解码 + Beam Search |
| `config.py` | 超参数配置 |

## 1. model.py 核心模块解析

### 1.1 scaled_dot_product_attention

这是整个 Transformer 最核心的函数。

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
```

**维度变化**：`[B,H,Lq,dk] x [B,H,dk,Lk] → [B,H,Lq,Lk]`

得到的 `scores[i][h][q][k]` 表示第 i 个样本、第 h 个头中，query 位置 q 对 key 位置 k 的注意力分数。

### 1.2 MultiHeadAttention 的高效实现

关键技巧：用**一个大矩阵**一次性完成所有头的投影，然后通过 reshape 和 transpose 分出多个头。

```python
Q = self.W_q(query)  # [B, L, d_model]
Q = Q.view(B, -1, n_heads, d_k).transpose(1, 2)  # [B, n_heads, L, d_k]
```

这比为每个头分别创建投影矩阵更高效，因为 GPU 擅长大矩阵运算。

### 1.3 位置编码的实现

`register_buffer` 确保位置编码表随模型一起保存/加载，但不参与梯度更新。

### 1.4 Encoder vs Decoder

- **Encoder**：`self_attention(x, x, x)` — Q=K=V=x，所有位置互相可见
- **Decoder**：多了因果掩码（只能看到过去）和 Cross-Attention（Q 来自解码器，K/V 来自编码器）

## 2. 训练流程关键点

### Teacher Forcing

解码器每步的输入是真实的上一个 token，而非模型预测的 token。这加速了收敛但引入了 Exposure Bias。

### 学习率 Warmup

Transformer 对学习率敏感。Warmup 阶段让 LR 从 0 线性增到目标值，避免训练初期发散。

### 梯度裁剪

`clip_grad_norm_` 在梯度的 L2 范数超过阈值时等比缩放，是训练 Transformer 的标配技巧。

## 3. 推理：贪心 vs Beam Search

| 策略 | 每步选择 | 复杂度 | 效果 |
|------|---------|--------|------|
| 贪心 | 概率最高的 1 个 | O(L) | 可能局部最优 |
| Beam Search | 维护 K 个候选 | O(K*L) | 更可能全局最优 |

对于排序任务，贪心通常就够了。但在文本生成中，Beam Search 能显著提升质量。
