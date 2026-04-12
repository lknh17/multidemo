# v13 代码解释：高级注意力机制实现详解

## 文件结构

| 文件 | 作用 |
|------|------|
| `model.py` | GQA/MQA/Sliding Window Attention 从零实现 |
| `attention_variants.py` | 独立的注意力变体集合，含 Flash Attention 模拟 |
| `dataset.py` | 长序列语言建模数据集 |
| `train.py` | 三种注意力的对比训练实验 |
| `inference.py` | KV Cache 推理 + 速度/显存 Benchmark |
| `config.py` | 超参数配置 |

## 1. GQA 核心实现

```python
# 关键区别：KV heads 数 < Query heads 数
self.n_kv_heads = n_kv_heads  # 例如 8
self.n_heads = n_heads          # 例如 32
self.n_groups = n_heads // n_kv_heads  # 每组 4 个 Q head 共享 1 对 KV

# KV 投影维度更小
self.W_k = nn.Linear(d_model, n_kv_heads * d_k)  # 而非 n_heads * d_k
self.W_v = nn.Linear(d_model, n_kv_heads * d_k)

# Forward 中需要 expand KV 以匹配 Q 的头数
K = K.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)  # 组内广播
K = K.reshape(B, self.n_heads, L, self.d_k)  # 合并组维度
```

**维度变化**：Q `[B, 32, L, dk]`，K 先 `[B, 8, L, dk]` → expand → `[B, 32, L, dk]`

## 2. Sliding Window 实现

通过构造带状掩码实现局部注意力：
```python
def create_sliding_window_mask(seq_len, window_size):
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + 1)  # 因果：只看过去
        mask[i, start:end] = True
    return mask
```

## 3. NTK-aware RoPE

```python
# 动态调整频率基数
alpha = (target_len / train_len) ** (d_model / (d_model - 2))
theta_new = theta * alpha ** (2 * torch.arange(0, d_model, 2) / d_model)
```

## 4. KV Cache 对比

MHA/GQA/MQA 的 KV Cache 大小差异是推理效率的关键：
- MHA: `2 × n_heads × L × d_k` = `2 × 32 × 4096 × 64` = 16M
- GQA(G=8): `2 × 8 × 4096 × 64` = 4M（节省 75%）
- MQA: `2 × 1 × 4096 × 64` = 0.5M（节省 97%）
