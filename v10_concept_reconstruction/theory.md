# 商业概念重构 Loss 原理

## 1. 为什么需要概念重构

对比学习学的是"相似/不相似"的二元关系。但广告场景有丰富的**结构化商业语义**：
- 行业（电商/游戏/教育/金融）
- 品牌（Nike/Apple/...）
- 商品属性（颜色/材质/价格区间）
- 用户意图（购买/了解/比较）

**概念重构 Loss** 的目标：从 embedding 中能**重构**出这些商业属性，迫使 embedding 编码商业语义。

## 2. 重构学习目标

给定 embedding $z$，通过重构头预测商业属性：

$$\hat{y}_{industry} = \text{MLP}_{industry}(z)$$
$$\hat{y}_{brand} = \text{MLP}_{brand}(z)$$
$$\hat{y}_{attr} = \text{MLP}_{attr}(z)$$

重构 Loss:
$$\mathcal{L}_{recon} = \lambda_1 \text{CE}(\hat{y}_{industry}, y_{industry}) + \lambda_2 \text{BCE}(\hat{y}_{attr}, y_{attr}) + ...$$

## 3. 与对比 Loss 的互补性

- **对比 Loss**：学习"什么是相似的"（全局语义空间）
- **重构 Loss**：学习"包含什么信息"（局部属性空间）

联合训练：
$$\mathcal{L}_{total} = \alpha \mathcal{L}_{contrastive} + \beta \mathcal{L}_{recon} + \gamma \mathcal{L}_{deepstack}$$

## 4. 动态 Loss 权重

使用 **Uncertainty Weighting** 或 **GradNorm** 自动平衡多任务 Loss，避免手动调参。

Uncertainty Weighting:
$$\mathcal{L} = \sum_i \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log\sigma_i$$

其中 $\sigma_i$ 是可学习的不确定性参数。
