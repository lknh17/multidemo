# DeepStack 深层融合原理

## 1. 核心问题

标准对比学习只使用模型**最后一层**的 embedding。但不同层包含不同语义：
- **浅层**：纹理、颜色、词形等低级特征
- **中层**：物体部件、短语语义
- **深层**：高级语义、抽象概念

只用最后一层会丢失低层信息，对广告细节（如logo、颜色搭配、文字排版）不利。

## 2. DeepStack 思想

**从多层提取特征，堆叠/融合后生成更丰富的 embedding。**

```
Layer 1 → feat_1 ─┐
Layer 4 → feat_4 ──┤→ Fusion → 多层 Embedding
Layer 8 → feat_8 ──┤
Layer 12 → feat_12 ┘
```

## 3. 融合策略

| 策略 | 描述 | 优缺点 |
|------|------|--------|
| Concatenation | 直接拼接 | 简单但维度大 |
| Weighted Sum | 可学习权重加权求和 | 维度不变，权重可解释 |
| Attention Fusion | 用注意力机制动态融合 | 最灵活但计算量大 |
| Gated Fusion | 门控机制选择性融合 | 平衡效果和效率 |

## 4. 多层 Loss

在多层分别施加对比学习 Loss：

$$\mathcal{L} = \sum_{l \in \text{layers}} w_l \cdot \mathcal{L}_{contrastive}^{(l)}$$

这迫使每层都学到有区分力的表示，而不仅依赖最后一层。

## 5. 接入对比学习

DeepStack + 对比学习 Loss 的完整流程：
1. 多模态模型前向，提取指定层的中间特征
2. 各层特征分别通过投影头得到 embedding
3. 各层分别计算 InfoNCE Loss
4. 可选：融合所有层特征得到最终 embedding
5. 最终 embedding 也计算一个 InfoNCE Loss
6. 总 Loss = Σ(层 Loss) + 融合 Loss
