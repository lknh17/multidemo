# 多模态数据工程 — 数学原理

> 本文详细推导数据去重（MinHash / SimHash）、数据增强（CutMix / MixUp）、课程学习、数据质量评分的核心数学公式。

## 目录

1. [数据去重 — MinHash 与 Jaccard 相似度](#1-数据去重--minhash-与-jaccard-相似度)
2. [数据去重 — SimHash 与角距离](#2-数据去重--simhash-与角距离)
3. [数据增强 — MixUp](#3-数据增强--mixup)
4. [数据增强 — CutMix](#4-数据增强--cutmix)
5. [课程学习 — 节奏函数](#5-课程学习--节奏函数)
6. [数据质量评分](#6-数据质量评分)
7. [重要性采样与数据平衡](#7-重要性采样与数据平衡)

---

## 1. 数据去重 — MinHash 与 Jaccard 相似度

### 1.1 Jaccard 相似度

给定两个集合 \(A\) 和 \(B\)，Jaccard 相似度定义为：

\[
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

- \(J \in [0, 1]\)：0 表示无交集，1 表示完全相同
- 直接计算 \(J\) 需要 \(O(|A| + |B|)\)，当数据量巨大时两两比较不可行

### 1.2 MinHash 估计

MinHash 利用随机哈希函数来**近似估计** Jaccard 相似度。

**核心定理**：对于随机哈希函数 \(h\)：

\[
\Pr[\min_{a \in A} h(a) = \min_{b \in B} h(b)] = J(A, B)
\]

即：两个集合的最小哈希值相等的概率，恰好等于它们的 Jaccard 相似度。

**实际操作**：使用 \(k\) 个不同的哈希函数 \(h_1, \ldots, h_k\)，每个集合的 MinHash 签名为：

\[
\text{sig}(A) = [\min_{a \in A} h_1(a), \min_{a \in A} h_2(a), \ldots, \min_{a \in A} h_k(a)]
\]

Jaccard 相似度的无偏估计：

\[
\hat{J}(A, B) = \frac{1}{k} \sum_{i=1}^{k} \mathbf{1}[\text{sig}_i(A) = \text{sig}_i(B)]
\]

### 1.3 LSH（Locality-Sensitive Hashing）加速

将长度为 \(k\) 的签名分成 \(b\) 个 band，每个 band 有 \(r = k/b\) 行。

两个文档成为候选对的概率：

\[
P(\text{candidate}) = 1 - (1 - J^r)^b
\]

通过调整 \(b\) 和 \(r\)，可以控制阈值附近的陡峭程度。

---

## 2. 数据去重 — SimHash 与角距离

### 2.1 SimHash 原理

SimHash 将高维向量映射为固定长度的二进制串，用于近似余弦相似度。

对于向量 \(v \in \mathbb{R}^d\) 和随机超平面法向量 \(r_i\)：

\[
\text{SimHash}(v)_i = \begin{cases} 1 & \text{if } v \cdot r_i \geq 0 \\ 0 & \text{if } v \cdot r_i < 0 \end{cases}
\]

### 2.2 角距离近似

两个 SimHash 签名的 Hamming 距离与原始向量的角距离成正比：

\[
\frac{d_H(\text{sig}(u), \text{sig}(v))}{b} \approx \frac{\theta(u, v)}{\pi}
\]

其中 \(\theta(u, v) = \arccos\left(\frac{u \cdot v}{\|u\| \|v\|}\right)\) 是两个向量的夹角。

余弦相似度的近似：

\[
\cos(\theta) \approx \cos\left(\pi \cdot \frac{d_H}{b}\right)
\]

---

## 3. 数据增强 — MixUp

### 3.1 核心公式

给定两个训练样本 \((x_i, y_i)\) 和 \((x_j, y_j)\)：

\[
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
\]
\[
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
\]

其中混合比例 \(\lambda\) 采样自 Beta 分布：

\[
\lambda \sim \text{Beta}(\alpha, \alpha), \quad \alpha > 0
\]

### 3.2 Beta 分布性质

Beta 分布的概率密度函数：

\[
f(\lambda; \alpha, \beta) = \frac{\lambda^{\alpha-1}(1-\lambda)^{\beta-1}}{B(\alpha, \beta)}
\]

当 \(\alpha = \beta\)（对称 Beta 分布）：
- \(\alpha < 1\)：U 形分布（偏向 0 或 1，混合程度小）
- \(\alpha = 1\)：均匀分布
- \(\alpha > 1\)：钟形分布（偏向 0.5，混合程度大）

**推荐值**：\(\alpha = 0.2\)（大多数时候接近原始样本，偶尔强混合）

### 3.3 为什么 MixUp 有效？

MixUp 相当于在输入空间中**线性插值**，鼓励模型在训练样本之间学习**线性**的预测行为。

- 正则化效果：减小对抗扰动的敏感度
- 标签平滑：软标签比硬标签提供更多的梯度信息
- 训练稳定性：减少过拟合，提升泛化能力

---

## 4. 数据增强 — CutMix

### 4.1 核心公式

CutMix 不是在像素级混合，而是用一个矩形区域替换：

\[
\tilde{x} = \mathbf{M} \odot x_i + (\mathbf{1} - \mathbf{M}) \odot x_j
\]
\[
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
\]

其中 \(\mathbf{M} \in \{0, 1\}^{H \times W}\) 是二值掩码，\(\lambda\) 表示保留区域的面积比。

### 4.2 随机矩形区域

混合比例同样从 Beta 分布采样：\(\lambda \sim \text{Beta}(\alpha, \alpha)\)

矩形中心 \((c_x, c_y)\) 均匀随机，矩形宽高为：

\[
r_w = W \sqrt{1 - \lambda}, \quad r_h = H \sqrt{1 - \lambda}
\]

保证裁剪区域面积比恰好为 \(1 - \lambda\)：

\[
\frac{r_w \cdot r_h}{W \cdot H} = 1 - \lambda
\]

### 4.3 CutMix vs MixUp vs Cutout

| 方法 | 图像操作 | 标签操作 | 信息保留 |
|------|---------|---------|---------|
| Cutout | 遮挡矩形区域 | 不变 | 丢失信息 |
| MixUp | 全局像素混合 | 混合 | 全部保留但模糊 |
| CutMix | 矩形区域替换 | 混合 | 两张图信息都保留 |

---

## 5. 课程学习 — 节奏函数

### 5.1 核心思想

课程学习模仿人类的学习过程：**先学简单样本，再逐步引入困难样本**。

关键组件：
1. **难度评分函数** \(d(x)\)：衡量样本的难度
2. **节奏函数** \(\lambda(t)\)：控制在训练进度 \(t\) 时使用多少比例的数据

### 5.2 常见节奏函数

给定训练进度 \(t \in [0, 1]\)：

**线性节奏**（Linear Pacing）：

\[
\lambda(t) = \lambda_0 + (1 - \lambda_0) \cdot \min\left(1, \frac{t}{T_w}\right)
\]

**根号节奏**（Root Pacing）：

\[
\lambda(t) = \lambda_0 + (1 - \lambda_0) \cdot \min\left(1, \sqrt{\frac{t}{T_w}}\right)
\]

**阶梯节奏**（Step Pacing）：

\[
\lambda(t) = \lambda_0 + (1 - \lambda_0) \cdot \min\left(1, \left\lfloor \frac{t \cdot K}{T_w} \right\rfloor / K\right)
\]

其中 \(\lambda_0\) 是初始数据比例，\(T_w\) 是 warmup 比例，\(K\) 是阶梯数。

### 5.3 难度评分

常用的样本难度衡量方式：

\[
d_{\text{loss}}(x, y) = \mathcal{L}(f_\theta(x), y)
\]

\[
d_{\text{conf}}(x, y) = 1 - p_\theta(y | x)
\]

\[
d_{\text{entropy}}(x) = -\sum_{c} p_\theta(c | x) \log p_\theta(c | x)
\]

---

## 6. 数据质量评分

### 6.1 多维度质量评分

综合质量评分由多个维度加权组成：

\[
Q(x) = w_{\text{clip}} \cdot S_{\text{clip}}(x) + w_{\text{res}} \cdot S_{\text{res}}(x) + w_{\text{ar}} \cdot S_{\text{ar}}(x) + w_{\text{blur}} \cdot S_{\text{blur}}(x)
\]

**CLIP 质量评分**（基于图文匹配度）：

\[
S_{\text{clip}}(x) = \cos(f_{\text{img}}(x), f_{\text{text}}(\text{prompt}))
\]

**分辨率评分**（归一化到 [0, 1]）：

\[
S_{\text{res}}(x) = \min\left(1, \frac{\sqrt{H \cdot W}}{R_{\text{target}}}\right)
\]

**宽高比评分**：

\[
S_{\text{ar}}(x) = \exp\left(-\frac{(\log(W/H))^2}{2\sigma^2}\right)
\]

### 6.2 质量加权训练

将质量评分作为样本权重参与训练：

\[
\mathcal{L}_{\text{weighted}} = \frac{\sum_{i} Q(x_i) \cdot \mathcal{L}(f_\theta(x_i), y_i)}{\sum_{i} Q(x_i)}
\]

---

## 7. 重要性采样与数据平衡

### 7.1 类别不平衡问题

当类别分布不均衡时，模型倾向于偏向多数类。

**逆频率权重**：

\[
w_c = \frac{N}{C \cdot n_c}
\]

其中 \(N\) 是总样本数，\(C\) 是类别数，\(n_c\) 是类别 \(c\) 的样本数。

**有效样本数权重**（Effective Number of Samples）：

\[
w_c = \frac{1 - \beta}{1 - \beta^{n_c}}, \quad \beta \in [0, 1)
\]

当 \(\beta \to 0\) 退化为等权重，\(\beta \to 1\) 退化为逆频率权重。

### 7.2 重要性采样

通过调整采样概率实现数据平衡：

\[
P(x_i \text{ sampled}) \propto w_{c(x_i)} = \frac{1}{n_{c(x_i)}^{\gamma}}
\]

\(\gamma = 0\)：均匀采样（不平衡），\(\gamma = 1\)：完全平衡，\(\gamma = 0.5\)：折中。

### 7.3 过采样策略

对少数类进行过采样，使每个类别的样本数接近最大类别：

\[
n_c^{\text{new}} = \max_k n_k
\]

少数类样本重复 \(\lceil \max_k n_k / n_c \rceil\) 次。结合数据增强可减少过拟合风险。

---

## 总结

| 技术 | 核心思想 | 关键公式 |
|------|---------|---------|
| MinHash | 随机哈希估计 Jaccard | \(\Pr[\min h(A) = \min h(B)] = J(A,B)\) |
| SimHash | 随机超平面映射 | \(d_H / b \approx \theta / \pi\) |
| MixUp | 全局线性插值 | \(\tilde{x} = \lambda x_i + (1-\lambda) x_j\) |
| CutMix | 矩形区域替换 | \(\tilde{x} = M \odot x_i + (1-M) \odot x_j\) |
| 课程学习 | 先易后难 | \(\lambda(t)\) 节奏函数 |
| 质量评分 | 多维度加权 | \(Q(x) = \sum w_d \cdot S_d(x)\) |
| 重要性采样 | 逆频率加权 | \(w_c = N / (C \cdot n_c)\) |

下一步：在 `data_ops.py` 和 `model.py` 中实现这些算法！
