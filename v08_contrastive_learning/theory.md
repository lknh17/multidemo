# 对比学习深入原理

## 1. 对比学习家族

| Loss | 公式特点 | 适用场景 |
|------|---------|---------|
| InfoNCE | softmax over batch negatives | 大 batch 对比学习 (CLIP) |
| Triplet Loss | max(0, d(a,p) - d(a,n) + margin) | 人脸识别、小 batch |
| Circle Loss | 自适应加权正负样本对 | 更精细的相似度学习 |
| NT-Xent | InfoNCE + 温度 + 正则化 | SimCLR 自监督 |

## 2. InfoNCE Loss 深入

$$\mathcal{L} = -\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)}$$

### 梯度分析
对正样本的梯度：$\frac{\partial\mathcal{L}}{\partial s_{ii}} = p_{ii} - 1$（推动正样本相似度增大）
对负样本的梯度：$\frac{\partial\mathcal{L}}{\partial s_{ij}} = p_{ij}$（推动负样本相似度减小）

**关键洞察**：困难负样本（$p_{ij}$ 大的）获得更大的梯度 → 自动关注困难样本。

### 温度系数 τ 的作用
- τ → 0：只关注最困难的负样本（可能不稳定）
- τ → ∞：均匀对待所有负样本（信号弱）
- 实践中 τ ∈ [0.05, 0.1] 效果较好

## 3. Hard Negative Mining

- **Random**: 随机采样负样本
- **Semi-Hard**: 选择比正样本远但在 margin 内的负样本
- **Hardest**: 选择最接近 anchor 的负样本（训练不稳定但信息量大）
- **In-batch**: 用 batch 内其他样本作负样本（CLIP 方式，简单高效）

## 4. 广告场景对比学习

- Anchor: 广告创意（图片+文案）
- Positive: 匹配的搜索 Query
- Negative: 不匹配的 Query（同行业但不同商品 > 随机 > 同商品不同表达）
- 相关性标签: 0-4 分级（无关/弱相关/相关/强相关/完全匹配）
