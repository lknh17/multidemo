# 多模态 Embedding 生成与检索原理

## 1. Embedding 提取策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| [CLS] Token | 取 [CLS] 位置的输出 | ViT、BERT |
| Mean Pooling | 所有 token 的平均 | 通用，鲁棒性好 |
| Weighted Pooling | 按注意力权重加权平均 | 需要突出重要 token |
| Last Token | 取最后一个 token | GPT 类自回归模型 |

## 2. 后处理

- **L2 归一化**：映射到单位球面，余弦相似度 = 内积
- **降维 (PCA/SVD)**：减少存储和计算开销
- **白化 (Whitening)**：消除特征相关性，提升检索效果

## 3. 向量检索

| 方法 | 精度 | 速度 | 内存 |
|------|------|------|------|
| Brute Force | 精确 | O(N) | 原始大小 |
| IVF (Inverted File) | 近似 | O(√N) | 聚类中心+倒排 |
| HNSW | 高精度近似 | O(log N) | 额外图结构 |
| PQ (Product Quantization) | 中等 | O(N) | 压缩 8-32x |
| IVF+PQ | 近似 | O(√N) | 压缩+倒排 |

## 4. 评估指标

- **Recall@K**：Top-K 结果中包含正确答案的比例
- **NDCG**：考虑位置的排序质量
- **MRR**：第一个正确结果的位置倒数

## 5. 完整检索 Pipeline

```
离线: 广告库 → 编码 → Embedding → 构建 FAISS 索引
在线: Query → 编码 → Embedding → ANN 检索 → 重排 → 结果
```
