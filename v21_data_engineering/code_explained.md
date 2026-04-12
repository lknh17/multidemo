# V21 代码解释：多模态数据工程实现详解

## 文件结构

| 文件 | 作用 |
|------|------|
| `config.py` | 数据过滤 / 去重 / 增强 / 课程学习 / 平衡等全流程配置 |
| `data_ops.py` | 核心数据操作：MinHash/SimHash 去重、质量评分、数据平衡 |
| `model.py` | 增强训练器（CutMix/MixUp/RandAug）、课程调度器、合成数据生成 |
| `dataset.py` | 合成数据集：图像 + 文本，支持不平衡分布 |
| `train.py` | 三种训练模式：augmented / curriculum / balanced |
| `inference.py` | 四组实验：去重 / 增强 / 课程 / 质量 |

## 1. data_ops.py 核心模块解析

### 1.1 MinHashDedup — MinHash 去重

```python
# 1. Shingling: 将文本切分为 n-gram 集合
shingles = {hash("abc"), hash("bcd"), hash("cde"), ...}

# 2. 签名计算：k 个哈希函数的最小值
sig[i] = min{h_i(s) : s ∈ shingles}

# 3. LSH 分桶加速：将签名分成 b 个 band
bucket_id = hash(sig[start:end])  # 每个 band 一个桶

# 4. 候选对精确比较
jaccard_est = mean(sig_A[i] == sig_B[i])
```

**关键实现细节**：
- 使用 `h(x) = (a*x + b) % p` 作为哈希函数族（a, b 随机，p 为大素数）
- LSH 的 band 数量影响候选对阈值：`P = 1 - (1 - J^r)^b`
- 时间复杂度从 O(N²) 降到近似 O(N)

### 1.2 SimHashDedup — SimHash 去重

```python
# 1. 随机超平面映射
projections = planes @ vector  # [hash_bits]
hash_bits = (projections >= 0)  # 二值化

# 2. Hamming 距离 → 余弦相似度
theta = π * hamming_dist / num_bits
cosine_sim = cos(theta)
```

**与 MinHash 的区别**：
- MinHash 估计 **Jaccard 相似度**（适合集合数据，如文本 n-gram）
- SimHash 估计 **余弦相似度**（适合向量数据，如嵌入特征）

### 1.3 QualityScorer — 质量评分

```python
# 多维度加权评分
Q(x) = 0.4 * S_clip + 0.3 * S_resolution + 0.2 * S_aspect_ratio + 0.1 * S_blur

# 分辨率评分：几何均值归一化
S_res = min(1, sqrt(H*W) / 224)

# 宽高比评分：对数正态惩罚
S_ar = exp(-(log(W/H))² / (2σ²))
```

### 1.4 DataBalancer — 数据平衡

```python
# 逆频率权重
w_c = N / (C * n_c)

# 有效样本数权重（更平滑）
w_c = (1 - β) / (1 - β^n_c)

# 过采样：复制少数类至最大类样本数
n_repeat = max_count // n_c
```

## 2. model.py 增强训练器

### 2.1 MixUp 实现

```python
# Beta 分布采样混合比例
lam = np.random.beta(alpha, alpha)  # α=0.2 → 大部分接近 0 或 1

# 像素级混合
mixed = lam * x_i + (1-lam) * x_j

# 标签混合（one-hot → soft label）
mixed_label = lam * y_onehot_i + (1-lam) * y_onehot_j
```

**注意**：使用 soft cross-entropy loss: `-Σ y_soft * log_softmax(logits)`

### 2.2 CutMix 实现

```python
# 随机矩形：面积比 = 1-λ
cut_w = W * sqrt(1-lam)
cut_h = H * sqrt(1-lam)

# 替换区域
mixed[:, :, y1:y2, x1:x2] = x_j[:, :, y1:y2, x1:x2]

# 实际 λ（因为裁剪到边界后面积可能变化）
lam_actual = 1 - (x2-x1)*(y2-y1) / (H*W)
```

### 2.3 CurriculumScheduler 课程调度

```python
# 节奏函数：控制使用数据比例
fraction = pacing_function(epoch)
# linear: λ₀ + (1-λ₀) * t/T_w
# root:   λ₀ + (1-λ₀) * sqrt(t/T_w)

# 按难度排序，取前 fraction 比例
sorted_by_difficulty = argsort(difficulties)
selected = sorted_by_difficulty[:n * fraction]
```

## 3. 训练流程关键点

### 3.1 Augmented Training

每个 batch 随机选择一种增强策略 → 计算 soft CE loss → 标准梯度更新。

关键：soft label 的 loss 不能用 `nn.CrossEntropyLoss`（它只接受 hard label），要手动计算。

### 3.2 Curriculum Training

每隔几个 epoch 重新评估样本难度（因为模型在变好，之前的难样本可能变简单了）。

### 3.3 Balanced Training

使用 `nn.CrossEntropyLoss(weight=class_weights)` 自动为少数类放大梯度。

## 4. 推理实验解读

| 实验 | 观察指标 | 预期现象 |
|------|---------|---------|
| 去重 | 压缩率 | 阈值越低压缩越多 |
| 增强 | 验证准确率 | Mixed > 单一 > 无增强 |
| 课程 | 训练曲线 | 前期课程快，后期趋同 |
| 质量 | 分布统计 | 近似正态，阈值可滤除低质量 |
